"""Composable positional-encoding model with standard scaled dot-product attention.

Select this architecture with ``--model=positional`` and configure a single
mechanism or a ``+`` composition with ``--pos_encoding=...``.
"""

import inspect
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

from models.base import LayerNorm, MLP
from models.positional_encodings import PositionalAttention, parse_positional_spec


class CausalSelfAttention(nn.Module):
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
        if self.head_dim % 2 and parse_positional_spec(config.pos_encoding).qk in ("rope", "fope"):
            raise ValueError(f"rotary PE needs an even head_dim, got {self.head_dim}")
        self.positioning = PositionalAttention(config)

    def forward(self, x):
        b, t, c = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        q, k = self.positioning.transform_qk(q, k)

        # Use PyTorch's standard 1/sqrt(head_dim) attention scaling.
        if self.positioning.is_legacy_fast_path:
            y = F.scaled_dot_product_attention(
                q, k, v, is_causal=True,
                dropout_p=self.dropout if self.training else 0.0,
            )
        else:
            semantic_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            static_bias = self.positioning.static_bias(t, x.device, semantic_scores.dtype)
            context_bias = self.positioning.contextual_bias(q, semantic_scores, static_bias)
            scores = semantic_scores
            # CAPE consumes its base RPE as f(attention, B), matching Eq. (2),
            # while CoPE composes additively with an optional independent RPE.
            if static_bias is not None and self.positioning.spec.context != "cape":
                scores = scores + static_bias
            if context_bias is not None:
                scores = scores + context_bias
            causal = torch.ones(t, t, device=x.device, dtype=torch.bool).tril()
            scores = scores.masked_fill(~causal, torch.finfo(scores.dtype).min)
            weights = F.softmax(scores, dim=-1)
            weights = self.attn_dropout(weights)
            y = torch.matmul(weights, v)

        y = y.transpose(1, 2).contiguous().view(b, t, c)
        return self.resid_dropout(self.c_proj(y))


class Block(nn.Module):
    def __init__(self, config, layer_idx=0):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config, layer_idx)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        return x + self.mlp(self.ln_2(x))


@dataclass
class ModelConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True
    pos_encoding: str = "nope"
    rope_theta: float = 10000.0
    rpe_num_buckets: int = 32
    rpe_max_distance: int = 128
    cope_max_position: int = 64
    cape_hidden_dim: int = 32
    fope_train_length: int = 128
    fope_init_gain: float = 0.3


class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None and config.block_size is not None
        self.config = config
        self.positional_spec = parse_positional_spec(config.pos_encoding)
        modules = dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([Block(config, i) for i in range(config.n_layer)]),
            ln_f=LayerNorm(config.n_embd, bias=config.bias),
        )
        if self.positional_spec.embedding == "wpe":
            modules["wpe"] = nn.Embedding(config.block_size, config.n_embd)
        self.transformer = nn.ModuleDict(modules)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        self.apply(self._init_weights)
        for name, parameter in self.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.normal_(parameter, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))
        print(f"positional encoding: {self.positional_spec.canonical}")
        print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))
        counts = self.get_positional_param_counts()
        print(f"positional parameters: trainable={counts['trainable']:,}, frozen={counts['frozen']:,}")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_positional_param_counts(self):
        trainable = frozen = 0
        for name, parameter in self.named_parameters():
            if ".positioning." in name or name.startswith("transformer.wpe"):
                if parameter.requires_grad:
                    trainable += parameter.numel()
                else:
                    frozen += parameter.numel()
        for name, buffer in self.named_buffers():
            if ".positioning." in name:
                frozen += buffer.numel()
        return {"trainable": trainable, "frozen": frozen, "total": trainable + frozen}

    def get_num_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding and self.positional_spec.embedding == "wpe":
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def forward(self, idx, targets=None):
        _, t = idx.size()
        assert t <= self.config.block_size, (
            f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}")
        x = self.transformer.wte(idx)
        if self.positional_spec.embedding == "wpe":
            pos = torch.arange(t, dtype=torch.long, device=idx.device)
            x = x + self.transformer.wpe(pos)
        x = self.transformer.drop(x)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None
        return logits, loss

    def crop_block_size(self, block_size):
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        if self.positional_spec.embedding == "wpe":
            self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        params = {n: p for n, p in self.named_parameters() if p.requires_grad}
        decay = [p for p in params.values() if p.dim() >= 2]
        nodecay = [p for p in params.values() if p.dim() < 2]
        groups = [{"params": decay, "weight_decay": weight_decay},
                  {"params": nodecay, "weight_decay": 0.0}]
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        extra = {"fused": True} if fused_available and device_type == "cuda" else {}
        return torch.optim.AdamW(groups, lr=learning_rate, betas=betas, **extra)

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        cfg = self.config
        n = self.get_num_params()
        l, h, q, t = cfg.n_layer, cfg.n_head, cfg.n_embd // cfg.n_head, cfg.block_size
        flops = (6 * n + 12 * l * h * q * t) * t * fwdbwd_per_iter
        return flops / dt / 312e12

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < values[:, [-1]]] = -float("inf")
            idx = torch.cat((idx, torch.multinomial(F.softmax(logits, dim=-1), 1)), dim=1)
        return idx
