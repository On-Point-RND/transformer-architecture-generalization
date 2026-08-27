"""FFN design (ablation axis 5): GELU, ReLU or SwiGLU at a matched parameter count.

The question is whether multiplicative gating buys anything on algorithmic tasks.
Comparing gated against ungated is only fair at equal parameter count, and that
is not the same hidden width: the ungated FFN spends 2*d*h on two matrices while
SwiGLU spends 3*d*h on three. Fixing the budget at the ungated 4*d width leaves
SwiGLU with 8*d/3, so the width is derived rather than configured.

``ffn: gelu`` is the model core/model.py already builds, kept here as the control
arm of the ablation.
"""

import math
from dataclasses import dataclass

import torch.nn as nn
from torch.nn import functional as F

from core.config import ModelConfig
from core.model import Transformer

ACTIVATIONS = {"gelu": F.gelu, "relu": F.relu}


@dataclass
class Config(ModelConfig):
    name: str = "ffn_gating"
    ffn: str = "gelu"      # 'gelu' | 'relu' | 'swiglu'
    ffn_hidden: int = 0    # 0 derives the iso-parameter width

    def __post_init__(self):
        if self.ffn not in ("gelu", "relu", "swiglu"):
            raise ValueError(f"model.ffn must be gelu/relu/swiglu, got {self.ffn!r}")


def hidden_dim(config):
    """Hidden width, chosen so every variant costs the same in parameters."""
    if config.ffn_hidden:
        return config.ffn_hidden
    if config.ffn == "swiglu":
        return math.floor(8 * config.n_embd / 3)  # 3*d*h == 2*d*(4*d)
    return 4 * config.n_embd


class UngatedMLP(nn.Module):
    """The standard two-matrix FFN; the activation is the ablated axis."""

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, hidden_dim(config), bias=config.bias)
        self.c_proj = nn.Linear(hidden_dim(config), config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
        self.activation = ACTIVATIONS[config.ffn]

    def forward(self, x):
        return self.dropout(self.c_proj(self.activation(self.c_fc(x))))


class GatedMLP(nn.Module):
    """SwiGLU: one branch gates the other before the down-projection."""

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, hidden_dim(config), bias=config.bias)
        self.c_gate = nn.Linear(config.n_embd, hidden_dim(config), bias=config.bias)
        self.c_proj = nn.Linear(hidden_dim(config), config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.c_proj(F.silu(self.c_gate(x)) * self.c_fc(x)))


class Model(Transformer):
    def build_mlp(self, config, layer_idx):
        return GatedMLP(config) if config.ffn == "swiglu" else UngatedMLP(config)
