"""Attention whose weights are a truncated exponential of the score.

Weight of key j for query i is ``sum_{p < n_taylor} s^p / p!`` at the ordinary
scaled score ``s = q.k/sqrt(head_dim)``, renormalised over the causal row. At
``n_taylor = 3`` this is exactly attention_design.taylor_softmax; higher values
carry the expansion further.

This is the same kernel as Symmetry-Aware Taylor Attention (arXiv:2602.00294),
which reaches it through an explicit monomial feature map. The two are
algebraically identical: the symmetry-aware map satisfies
``sum_m C_m phi_p(q)[m] phi_p(k)[m] = (q.k)^p``, so with its
``alpha_p = 1/(p! d^(p/2))`` the kernel collapses to the polynomial above. That
map buys linear complexity in T, but it stores ``[B, H, T, m, D]`` with
``m = sum_p C(D+p-1, p)``, so it only wins once ``m*D < T``: 1320 at head_dim 8
and n_taylor 4, three million at head_dim 64. Our blocks are 128-256, so the
score form is cheaper everywhere we run, and unlike the feature map its cost
does not depend on head_dim at all.

``n_taylor`` counts TERMS, not the top degree, following that paper's
convention: 3 -> 1 + s + s^2/2, 4 -> that + s^3/6, 5 -> that + s^4/24. An odd
top degree is not non-negative -- degree 3 goes negative below s = -1.60 -- and
a row can then sum to near zero and blow the weights up, so ``clamp_min``
floors both the weights and the row sum. Setting it to 0.0 removes that floor
and reproduces the unguarded formulation, sign flips included.
"""

import math
from dataclasses import dataclass

import torch

from core.config import ModelConfig
from core.model import CausalAttention, Transformer

UNGUARDED_EPS = 1e-8  # denominator stabiliser used when clamping is off


def taylor_weights(scores, visible, n_taylor, clamp_min):
    """Renormalised ``sum_{p < n_taylor} s^p / p!`` over each row's visible keys.

    Causality arrives as a boolean mask, not as a sentinel score: the core masks
    with ``finfo.min`` and attention_design with ``-inf``, and raising either to
    a power puts inf (or inf - inf = nan) into the graph.
    """
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
    n_taylor: int = 3         # terms, not degree: 3 -> 1 + s + s^2/2
    clamp_min: float = 1e-6   # 0.0 removes the floor: weights may go negative

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
