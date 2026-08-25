import inspect
import math

import torch
import torch.nn as nn
from torch.nn import functional as F


class LayerNorm(nn.Module):
    """LayerNorm with an optional bias. PyTorch doesn't support bias=False."""

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class MLP(nn.Module):
    def __init__(self, config):
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

    # --- hooks ----------------------------------------------------------
    def transform_qk(self, q, k):
        return q, k

    def add_bias(self, scores, q, k):
        """Add positional terms to the raw scores and return them.

        Returns the scores rather than a bias tensor on purpose: a variant that
        composes several terms controls the order of the additions, which is
        what keeps results bit-identical across refactors.
        """
        return scores

    # --- forward --------------------------------------------------------
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

    The attention sublayer is passed in, so an architecture varies it through
    Transformer.build_attention without this class knowing about it.
    """

    def __init__(self, config, attn):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = attn
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        return x + self.mlp(self.ln_2(x))


class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None and config.block_size is not None
        self.config = config
        # Construction order is part of the numerics — see the module docstring.
        modules = dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([Block(config, self.build_attention(config, i))
                             for i in range(config.n_layer)]),
            ln_f=LayerNorm(config.n_embd, bias=config.bias),
        )
        if self.uses_pos_embedding(config):
            modules["wpe"] = nn.Embedding(config.block_size, config.n_embd)
        self.transformer = nn.ModuleDict(modules)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        self.apply(self._init_weights)
        for name, parameter in self.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.normal_(parameter, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    # --- construction hooks ---------------------------------------------
    def build_attention(self, config, layer_idx):
        return CausalAttention(config, layer_idx)

    def uses_pos_embedding(self, config):
        return True

    # --- reporting hook -------------------------------------------------
    def param_report(self):
        """What this architecture contributes positionally, for the run log.

        The training loop reads the mechanism name from here instead of guessing
        it from the config, so a new architecture is described by its own code.
        """
        wpe = self.transformer.wpe.weight.numel() if "wpe" in self.transformer else 0
        return {"positional_encoding": "wpe" if wpe else "nope",
                "positional_parameters": {"trainable": wpe, "frozen": 0, "total": wpe}}

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
            # inference-time mini-optimization: only the last position is needed
            return self.lm_head(x[:, [-1], :]), None
        logits = self.lm_head(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1),
                               ignore_index=-1)
        return logits, loss

    def get_num_params(self, non_embedding=True):
        """Parameter count; the position table is excluded by default.

        Token embeddings are counted because weight tying makes them the output
        layer as well.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding and "wpe" in self.transformer:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def crop_block_size(self, block_size):
        """Model surgery for loading a checkpoint trained with a longer context."""
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        if "wpe" in self.transformer:
            self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        """Decay every 2D tensor (matmuls + embeddings); leave biases/norms alone."""
        params = {n: p for n, p in self.named_parameters() if p.requires_grad}
        decay = [p for p in params.values() if p.dim() >= 2]
        nodecay = [p for p in params.values() if p.dim() < 2]
        groups = [{"params": decay, "weight_decay": weight_decay},
                  {"params": nodecay, "weight_decay": 0.0}]
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        extra = {"fused": True} if fused_available and device_type == "cuda" else {}
        return torch.optim.AdamW(groups, lr=learning_rate, betas=betas, **extra)

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
