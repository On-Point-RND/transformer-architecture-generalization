"""Attention design (ablation axes 1, 2 and 4).

Three axes share this file because they all modify one computation — the same
attention call — and the ablation composes them freely:

  axis 4, KV/projection design
      ``n_kv_head`` equal to n_head is MHA, 1 is MQA, a divisor in between is GQA.
      ``share_kv`` is the K=V ablation: one projection feeds both keys and values,
      forcing addresses and content into a single subspace.
  axis 2, attention pattern
      ``full``, or ``local``: a sliding window of ``window_size`` tokens plus
      ``n_global_tokens`` leading sink tokens every query can always reach.
  axis 1, softmax
      the exact softmax, or its second-order Taylor approximation.

With the defaults — MHA, full, standard — this is core.model.CausalAttention:
the fused projection is the same 3*n_embd matrix built in the same order, and
the forward takes the same fused kernel. That makes the ablation's own control
arm verifiable rather than merely asserted.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

from core.config import ModelConfig
from core.model import Transformer


@dataclass
class Config(ModelConfig):
    name: str = "attention_design"
    n_kv_head: int = 0          # 0 follows n_head (MHA)
    share_kv: bool = False      # K=V: the key projection is reused as the value one
    pattern: str = "full"       # 'full' | 'local'
    window_size: int = 0        # tokens visible behind a query under 'local'
    n_global_tokens: int = 0    # leading tokens every query can always see
    softmax: str = "standard"   # 'standard' | 'taylor'

    def __post_init__(self):
        kv = self.n_kv_head or self.n_head
        if self.n_head % kv:
            raise ValueError(f"n_head ({self.n_head}) must be divisible by "
                             f"n_kv_head ({kv})")
        if self.pattern not in ("full", "local"):
            raise ValueError(f"model.pattern must be full/local, got {self.pattern!r}")
        if self.pattern == "local" and self.window_size <= 0:
            raise ValueError("model.window_size is required when pattern is 'local'")
        if self.softmax not in ("standard", "taylor"):
            raise ValueError(f"model.softmax must be standard/taylor, got {self.softmax!r}")


def taylor_softmax(scores):
    """Second-order Taylor expansion of exp, renormalised.

    Masked positions arrive as -inf and are dropped outright: the polynomial is
    even, so a large negative score would otherwise come back as a large positive
    weight.
    """
    weights = (1.0 + scores + 0.5 * scores.pow(2)).clamp(min=1e-6)
    weights = torch.where(torch.isfinite(scores), weights, torch.zeros_like(weights))
    return weights / weights.sum(dim=-1, keepdim=True).clamp(min=1e-6)


class DesignedAttention(nn.Module):
    def __init__(self, config, layer_idx=0):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head or config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.share_kv = config.share_kv
        self.pattern = config.pattern
        self.window_size = config.window_size
        self.n_global_tokens = config.n_global_tokens
        self.softmax = config.softmax
        self.dropout = config.dropout
        kv_dim = self.n_kv_head * self.head_dim
        self.splits = ((config.n_embd, kv_dim) if config.share_kv
                       else (config.n_embd, kv_dim, kv_dim))
        self.c_attn = nn.Linear(config.n_embd, sum(self.splits), bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        b, t, c = x.size()
        parts = self.c_attn(x).split(self.splits, dim=2)
        q, k = parts[0], parts[1]
        v = k if self.share_kv else parts[2]
        q = q.view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(b, t, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.n_kv_head, self.head_dim).transpose(1, 2)
        y = self._mix_values(q, self._repeat_kv(k), self._repeat_kv(v), t, x.device)
        y = y.transpose(1, 2).contiguous().view(b, t, c)  # heads side by side
        return self.resid_dropout(self.c_proj(y))

    def _repeat_kv(self, x):
        """[B, n_kv_head, T, head_dim] -> [B, n_head, T, head_dim] by repeating groups."""
        groups = self.n_head // self.n_kv_head
        if groups == 1:
            return x
        b, h, t, d = x.shape
        return x.unsqueeze(2).expand(b, h, groups, t, d).reshape(b, h * groups, t, d)

    def _visibility(self, t, device, dtype):
        """0 where a query may attend, -inf where it may not."""
        pos = torch.arange(t, device=device)
        visible = pos[:, None] >= pos[None, :]  # causal
        if self.pattern == "local":
            near = pos[:, None] - pos[None, :] < self.window_size
            sink = pos[None, :] < self.n_global_tokens
            visible = visible & (near | sink)
        return torch.zeros(t, t, device=device, dtype=dtype).masked_fill(~visible, -torch.inf)

    def _mix_values(self, q, k, v, t, device):
        dropout_p = self.dropout if self.training else 0.0
        if self.softmax == "standard" and self.pattern == "full":
            return F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                                  dropout_p=dropout_p)
        bias = self._visibility(t, device, q.dtype)
        if self.softmax == "standard":
            return F.scaled_dot_product_attention(q, k, v, attn_mask=bias,
                                                  dropout_p=dropout_p)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        weights = self.attn_dropout(taylor_softmax(scores + bias))
        return torch.matmul(weights, v)


class Model(Transformer):
    def build_attention(self, config, layer_idx):
        return DesignedAttention(config, layer_idx)
