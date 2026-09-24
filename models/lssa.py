"""LSSA and its rectified LSSAR variant.

The variants share the weight formulas below, but their attention modules are
siblings rather than an inheritance chain: each forward path is visible here.
"""

from dataclasses import dataclass

import torch
from torch.nn import functional as F

from core.config import ModelConfig
from core.model import CausalAttention, Transformer
from models.positional import rope_cache, rotate_half


def lssa_attention(q, k, v, eps=1e-8):
    _, _, length, head_dim = q.shape
    q = F.normalize(q, p=2, dim=-1)
    k = F.normalize(k, p=2, dim=-1)
    n = torch.arange(1, length + 1, device=q.device, dtype=q.dtype).view(1, 1, length, 1)
    log_d = torch.log(torch.tensor(head_dim, device=q.device, dtype=q.dtype))
    scores = (log_d * torch.log(n)) * (q @ k.transpose(-2, -1))
    attention = F.softplus(scores)
    causal = torch.ones(length, length, device=q.device, dtype=torch.bool).tril()
    attention = attention.masked_fill(~causal, 0.0)
    attention = attention / (attention.sum(dim=-1, keepdim=True) + eps)
    return attention @ v, attention


def lssar_attention(q, k, v, p=15, eps=1e-8):
    _, attention = lssa_attention(q, k, v, eps=eps)
    length = q.size(-2)
    n = torch.arange(1, length + 1, device=q.device, dtype=attention.dtype).view(1, 1, length, 1)
    offset = torch.ones(1, 1, length, 1, device=q.device, dtype=attention.dtype)
    offset[:, :, :3, :] = 0.0
    attention = torch.relu(attention * n - offset).pow(p)
    attention = attention / (attention.sum(dim=-1, keepdim=True) + eps)
    return attention @ v, attention


@dataclass
class LSSAConfig(ModelConfig):
    name: str = "lssa"
    rope_theta: float = 10000.0


@dataclass
class LSSARConfig(ModelConfig):
    name: str = "lssar"
    rope_theta: float = 10000.0
    p: int = 15


class LSSAAttention(CausalAttention):
    def __init__(self, config, layer_idx=0):
        super().__init__(config, layer_idx)
        if self.head_dim % 2:
            raise ValueError(f"RoPE needs an even head_dim, got {self.head_dim}")
        self.rope_theta = config.rope_theta

    def transform_qk(self, q, k):
        cos, sin = rope_cache(q.size(-2), self.head_dim, self.rope_theta, q.device, q.dtype)
        return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin

    def _mix_values(self, q, k, v, t, device):
        _, attention = lssa_attention(q, k, v)
        return self.attn_dropout(attention) @ v


class LSSARAttention(CausalAttention):
    def __init__(self, config, layer_idx=0):
        super().__init__(config, layer_idx)
        if self.head_dim % 2:
            raise ValueError(f"RoPE needs an even head_dim, got {self.head_dim}")
        self.rope_theta = config.rope_theta
        self.p = config.p

    def transform_qk(self, q, k):
        cos, sin = rope_cache(q.size(-2), self.head_dim, self.rope_theta, q.device, q.dtype)
        return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin

    def _mix_values(self, q, k, v, t, device):
        _, attention = lssar_attention(q, k, v, p=self.p)
        return self.attn_dropout(attention) @ v


def build_lssa(config):
    return Transformer(
        config,
        attention=LSSAAttention,
        use_pos_embedding=False,
        positional_encoding="rope",
        positional_markers=(),
    )


def build_lssar(config):
    return Transformer(
        config,
        attention=LSSARAttention,
        use_pos_embedding=False,
        positional_encoding="rope",
        positional_markers=(),
    )
