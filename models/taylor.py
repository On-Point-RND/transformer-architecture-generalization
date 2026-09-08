import math
from dataclasses import dataclass

import torch

from core.config import ModelConfig
from core.model import CausalAttention, Transformer

UNGUARDED_EPS = 1e-8  # denominator stabiliser used when clamping is off


def taylor_weights(scores, visible, n_taylor, clamp_min):
    keep = visible.to(scores.dtype)
    safe = scores * keep
    weights = torch.ones_like(safe)
    term = torch.ones_like(safe)
    for p in range(1, n_taylor):
        term = term * safe / p  # s^p / p! accumulated, no pow
        weights = weights + term
    if clamp_min > 0:
        weights = weights.clamp(min=clamp_min)
    weights = weights * keep
    total = weights.sum(dim=-1, keepdim=True)
    if clamp_min > 0:
        return weights / total.clamp(min=clamp_min)
    return weights / (total + UNGUARDED_EPS)


@dataclass
class Config(ModelConfig):
    name: str = "taylor"
    n_taylor: int = 3         
    clamp_min: float = 1e-6  

    def __post_init__(self):
        if self.n_taylor < 1:
            raise ValueError(f"n_taylor must be at least 1, got {self.n_taylor}")
        if self.clamp_min < 0:
            raise ValueError(f"clamp_min must be >= 0, got {self.clamp_min}")


class Attention(CausalAttention):
    needs_scores = True

    def __init__(self, config, layer_idx=0):
        super().__init__(config, layer_idx)
        self.n_taylor = config.n_taylor
        self.clamp_min = config.clamp_min

    def _mix_values(self, q, k, v, t, device):
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = self.add_bias(scores, q, k)
        visible = torch.ones(t, t, device=device, dtype=torch.bool).tril()
        weights = taylor_weights(scores, visible, self.n_taylor, self.clamp_min)
        return torch.matmul(self.attn_dropout(weights), v)


class Model(Transformer):
    def build_attention(self, config, layer_idx):
        return Attention(config, layer_idx)
