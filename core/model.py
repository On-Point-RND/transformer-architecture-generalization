import math

import torch
import torch.nn as nn
from torch.nn import functional as F

from optimizers import get_optimizer


class LayerNorm(nn.Module):
    """LayerNorm with an optional bias. PyTorch doesn't support bias=False."""

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class MLP(nn.Module):
    def __init__(self, config, layer_idx=0):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))


class CausalAttention(nn.Module):
    """Causal self-attention with the standard 1/sqrt(head_dim) scaling.

    Two paths, on purpose: with ``needs_scores = False`` this is exactly
    ``F.scaled_dot_product_attention`` (fused, never materialises the T x T
    score matrix). A variant that adds a bias or replaces the softmax sets
    ``needs_scores = True`` and gets the explicit path instead.
    """

    needs_scores = False

    def __init__(self, config, layer_idx=0):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.head_dim = config.n_embd // config.n_head

    def transform_qk(self, q, k):
        return q, k

    def add_bias(self, scores, q, k):
        """Add positional terms to the raw scores and return them.

        Returns the scores rather than a bias tensor on purpose: a variant that
        composes several terms controls the order of the additions, which is
        what keeps results bit-identical across refactors.
        """
        return scores

    def forward(self, x):
        b, t, c = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        shape = (b, t, self.n_head, self.head_dim)  # heads to their own axis
        q, k, v = (q.view(shape).transpose(1, 2), k.view(shape).transpose(1, 2),
                   v.view(shape).transpose(1, 2))
        q, k = self.transform_qk(q, k)
        y = self._mix_values(q, k, v, t, x.device)  # [B, n_head, T, head_dim]
        y = y.transpose(1, 2).contiguous().view(b, t, c)  # heads side by side
        return self.resid_dropout(self.c_proj(y))

    def _mix_values(self, q, k, v, t, device):
        if not self.needs_scores:
            return F.scaled_dot_product_attention(
                q, k, v, is_causal=True,
                dropout_p=self.dropout if self.training else 0.0,
            )
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = self.add_bias(scores, q, k)
        causal = torch.ones(t, t, device=device, dtype=torch.bool).tril()
        scores = scores.masked_fill(~causal, torch.finfo(scores.dtype).min)
        weights = self.attn_dropout(F.softmax(scores, dim=-1))
        return torch.matmul(weights, v)


class Block(nn.Module):
    """Pre-LN block: x + attn(ln_1(x)), then x + mlp(ln_2(x)).

    Both concrete sublayers are passed in; the block has no architecture hooks.
    """

    def __init__(self, config, attn, mlp):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = attn
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = mlp

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        return x + self.mlp(self.ln_2(x))


class Transformer(nn.Module):
    """Decoder stack assembled from explicit attention and MLP components."""

    def __init__(
        self,
        config,
        attention=CausalAttention,
        mlp=MLP,
        use_pos_embedding=True,
        n_block_applications=None,
        share_block_weights=False,
        residual_init_scaling=True,
        positional_encoding=None,
        positional_markers=("transformer.wpe",),
    ):
        super().__init__()
        assert config.vocab_size is not None and config.block_size is not None
        self.config = config
        self.positional_encoding = positional_encoding or (
            "wpe" if use_pos_embedding else "nope"
        )
        self.positional_markers = positional_markers
        n_blocks = config.n_layer if n_block_applications is None else n_block_applications
        token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        dropout = nn.Dropout(config.dropout)

        def make_block(layer_idx):
            return Block(config, attention(config, layer_idx), mlp(config, layer_idx))

        if share_block_weights:
            block = make_block(0)
            blocks = nn.ModuleList([block] * n_blocks)
        else:
            blocks = nn.ModuleList([make_block(i) for i in range(n_blocks)])
        final_norm = LayerNorm(config.n_embd, bias=config.bias)
        modules = dict(
            wte=token_embedding,
            drop=dropout,
            h=blocks,
            ln_f=final_norm,
        )
        if use_pos_embedding:
            modules["wpe"] = nn.Embedding(config.block_size, config.n_embd)
        self.transformer = nn.ModuleDict(modules)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        self.apply(self._init_weights)
        if residual_init_scaling:
            for name, parameter in self.named_parameters():
                if name.endswith("c_proj.weight"):
                    nn.init.normal_(parameter, mean=0.0,
                                    std=0.02 / math.sqrt(2 * config.n_layer))

    def param_report(self):
        def is_positional(name):
            return any(marker in name for marker in self.positional_markers)

        parameters = [(p.requires_grad, p.numel())
                      for name, p in self.named_parameters() if is_positional(name)]
        trainable = sum(size for trained, size in parameters if trained)
        frozen = sum(size for trained, size in parameters if not trained)
        frozen += sum(buffer.numel() for name, buffer in self.named_buffers()
                      if is_positional(name))
        return {
            "positional_encoding": self.positional_encoding,
            "positional_parameters": {
                "trainable": trainable,
                "frozen": frozen,
                "total": trainable + frozen,
            },
        }

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        _, t = idx.size()
        assert t <= self.config.block_size, (
            f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}")
        x = self.transformer.wte(idx)  # [B, T, n_embd]
        if "wpe" in self.transformer:
            pos = torch.arange(t, dtype=torch.long, device=idx.device)
            x = x + self.transformer.wpe(pos)
        x = self.transformer.drop(x)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        if targets is None:
            return self.lm_head(x[:, [-1], :]), None
        logits = self.lm_head(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1),
                               ignore_index=-1)
        return logits, loss

    def get_num_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding and "wpe" in self.transformer:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def crop_block_size(self, block_size):
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        if "wpe" in self.transformer:
            self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])

    def configure_optimizers(self, optimizer, device_type):
        params = {n: p for n, p in self.named_parameters() if p.requires_grad}
        if optimizer.decay == "all":
            groups = [{"params": list(params.values()),
                       "weight_decay": optimizer.weight_decay}]
        elif optimizer.decay == "matrices":
            groups = [{"params": [p for p in params.values() if p.dim() >= 2],
                       "weight_decay": optimizer.weight_decay},
                      {"params": [p for p in params.values() if p.dim() < 2],
                       "weight_decay": 0.0}]
        else:
            raise ValueError(f"optimizer.decay must be 'matrices' or 'all', "
                             f"got {optimizer.decay!r}")
        return get_optimizer(optimizer.name)(groups, optimizer, device_type)

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """Model flops utilisation against A100 bf16 peak (PaLM appendix B)."""
        cfg = self.config
        n = self.get_num_params()
        l, h, q, t = cfg.n_layer, cfg.n_head, cfg.n_embd // cfg.n_head, cfg.block_size
        flops = (6 * n + 12 * l * h * q * t) * t * fwdbwd_per_iter
        return flops / dt / 312e12

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """Complete idx [B, T] by feeding predictions back in, max_new_tokens times."""
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < values[:, [-1]]] = -float("inf")
            idx = torch.cat((idx, torch.multinomial(F.softmax(logits, dim=-1), 1)), dim=1)
        return idx
