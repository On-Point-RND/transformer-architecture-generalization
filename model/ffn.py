"""FFN sublayer (axis 5 of the ablation): ReLU vs SwiGLU, with a helper
to pick a hidden width so both variants have matched parameter count --
SwiGLU has 3 weight matrices instead of ReLU's 2, so a naive "same
hidden_dim" comparison silently gives SwiGLU more parameters.
"""
from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .config import FFNConfig


def compute_iso_param_hidden_dim(kind: str, d_model: int, target_params: int) -> int:
    """ReLU FFN:   params ~ 2 * d_model * hidden_dim   (W1, W2)
    SwiGLU FFN: params ~ 3 * d_model * hidden_dim   (W_gate, W_up, W_down)
    """
    if kind == "relu":
        divisor = 2 * d_model
    elif kind == "swiglu":
        divisor = 3 * d_model
    else:
        raise ValueError(f"Unknown FFN kind: {kind!r}")
    return max(1, target_params // divisor)


class ReluFFN(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(d_model, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(F.relu(self.w1(x)))


class SwiGLUFFN(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int):
        super().__init__()
        self.w_gate = nn.Linear(d_model, hidden_dim, bias=False)
        self.w_up = nn.Linear(d_model, hidden_dim, bias=False)
        self.w_down = nn.Linear(hidden_dim, d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class FFN(nn.Module):
    def __init__(self, d_model: int, config: FFNConfig):
        super().__init__()
        self.config = config
        if config.kind == "relu":
            self.impl: nn.Module = ReluFFN(d_model, config.hidden_dim)
        elif config.kind == "swiglu":
            self.impl = SwiGLUFFN(d_model, config.hidden_dim)
        else:
            raise ValueError(f"Unknown FFN kind: {config.kind!r}")

    def forward(self, x: Tensor) -> Tensor:
        return self.impl(x)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
