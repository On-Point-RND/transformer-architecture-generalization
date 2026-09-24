import math
from dataclasses import dataclass

import torch

from core.config import ModelConfig
from core.model import CausalAttention, Transformer

RELUMAX_EPS = 1e-12


def relumax_weights(scores, visible, degree, window):
    top = scores.masked_fill(~visible, torch.finfo(scores.dtype).min)
    top = top.max(dim=-1, keepdim=True).values
    inside = (1.0 + (scores - top) / window).clamp(min=0.0)
    weights = inside.pow(degree) * visible.to(scores.dtype)
    return weights / weights.sum(dim=-1, keepdim=True).clamp(min=RELUMAX_EPS)


@dataclass
class Config(ModelConfig):
    name: str = "relumax"
    degree: int = 2
    window: float = 5.0

    def __post_init__(self):
        if self.degree < 1:
            raise ValueError(f"degree must be at least 1, got {self.degree}")
        if self.window <= 0:
            raise ValueError(f"window must be positive, got {self.window}")


class Attention(CausalAttention):
    needs_scores = True

    def __init__(self, config, layer_idx=0):
        super().__init__(config, layer_idx)
        self.degree = config.degree
        self.window = config.window

    def _mix_values(self, q, k, v, t, device):
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = self.add_bias(scores, q, k)
        visible = torch.ones(t, t, device=device, dtype=torch.bool).tril()
        weights = relumax_weights(scores, visible, self.degree, self.window)
        return torch.matmul(self.attn_dropout(weights), v)


def build_model(config):
    return Transformer(config, attention=Attention)
