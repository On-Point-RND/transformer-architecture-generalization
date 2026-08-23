"""
All five positional encoding schemes: Learned absolute, Sinusoidal, RoPE, ALiBi, NoPE.
A single PositionalEncoding instance is created once per model and
shared across all layers

Two different injection points exist, and each scheme uses exactly one:
  - "additive" schemes (Learned absolute, Sinusoidal, NoPE-as-identity)
    are added to the token embedding once, before the first block.
  - "attention-integrated" schemes (RoPE, ALiBi) modify the attention
    computation itself, inside every layer -- RoPE rotates Q/K before
    the dot product, ALiBi adds a distance-based bias to the scores.
"""
from __future__ import annotations

import math
from abc import ABC
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor


# Base abstract class for all positional encodings
class PositionalEncoding(nn.Module, ABC):
    kind: str = "base"

    def embed(self, token_emb: Tensor, positions: Tensor) -> Tensor:
        """Additive schemes override this. Default: identity (NoPE and
        attention-integrated schemes both pass through unchanged here)."""
        return token_emb

    def rotate_qk(self, q: Tensor, k: Tensor, positions: Tensor) -> tuple[Tensor, Tensor]:
        """Only RoPE overrides this. q, k: (B, n_heads, T, head_dim)."""
        return q, k

    def attention_bias(self, seq_len: int, n_heads: int, device: torch.device) -> Optional[Tensor]:
        """Only ALiBi overrides this. Returns (n_heads, T, T) additive
        bias or None if this scheme adds no attention-level bias."""
        return None


# Model learns a unique vector for each position
class LearnedAbsolutePE(PositionalEncoding):
    kind = "learned_absolute"

    def __init__(self, d_model: int, max_seq_len: int):
        super().__init__()
        # Create a lookup table (Embedding) for positions
        self.pos_emb = nn.Embedding(max_seq_len, d_model)

    def embed(self, token_emb: Tensor, positions: Tensor) -> Tensor:
        # Add position vector to the word vector
        return token_emb + self.pos_emb(positions)


# Fixed mathematical formulas (sines and cosines)
class SinusoidalPE(PositionalEncoding):
    kind = "sinusoidal"

    def __init__(self, d_model: int, max_seq_len: int):
        super().__init__()
        pe = torch.zeros(max_seq_len, d_model)
        # Create a column of position numbers (0, 1, 2...)
        pos = torch.arange(max_seq_len).unsqueeze(1).float()
        # Calculate frequencies for sines
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe, persistent=False)  # not learned, not saved as a param

    def embed(self, token_emb: Tensor, positions: Tensor) -> Tensor:
        return token_emb + self.pe[positions]

# No Positional Encoding at all
class NoPE(PositionalEncoding):
    kind = "nope"
    # It inherits empty methods from the base class


# Rotates Q and K vectors like a compass needle
class RoPE(PositionalEncoding):
    kind = "rope"

    def __init__(self, head_dim: int, max_seq_len: int, base: float = 10000.0):
        super().__init__()
        assert head_dim % 2 == 0, "RoPE requires an even head_dim"
        # Calculate rotation angles
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        t = torch.arange(max_seq_len).float()
        freqs = torch.outer(t, inv_freq)  # (max_seq_len, head_dim/2)
        # Save Cosine and Sine of these angles
        self.register_buffer("cos", freqs.cos(), persistent=False)
        self.register_buffer("sin", freqs.sin(), persistent=False)

    def _rotate(self, x: Tensor, positions: Tensor) -> Tensor:
        # x: (B, n_heads, T, head_dim)
        cos = self.cos[positions].unsqueeze(1)  # (B, 1, T, head_dim/2)
        sin = self.sin[positions].unsqueeze(1)
        x1, x2 = x[..., 0::2], x[..., 1::2]
        # Mathematical 2D rotation formula
        rotated = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
        # Flatten back into a single vector
        return rotated.flatten(-2)

    def rotate_qk(self, q: Tensor, k: Tensor, positions: Tensor) -> tuple[Tensor, Tensor]:
        return self._rotate(q, positions), self._rotate(k, positions)


# Penalizes attention based on distance
class ALiBiPE(PositionalEncoding):
    kind = "alibi"

    def __init__(self, n_heads: int):
        super().__init__()
        # Standard geometric slope schedule from the ALiBi paper.
        def slopes(n):
            start = 2 ** (-8 / n)
            return [start ** (i + 1) for i in range(n)]

        # Save slopes to memory
        self.register_buffer("slopes", torch.tensor(slopes(n_heads)), persistent=False)

    def attention_bias(self, seq_len: int, n_heads: int, device: torch.device) -> Tensor:
        # Create column and row of indices
        i = torch.arange(seq_len, device=device).unsqueeze(1)
        j = torch.arange(seq_len, device=device).unsqueeze(0)
        distance = (i - j).clamp(min=0).float()  # causal: only j <= i matters
        return -self.slopes.to(device).view(n_heads, 1, 1) * distance.unsqueeze(0)


# Helper function to create the right class
def build_positional_encoding(
    kind: str, d_model: int, n_heads: int, head_dim: int, max_seq_len: int
) -> PositionalEncoding:
    if kind == "learned_absolute":
        return LearnedAbsolutePE(d_model, max_seq_len)
    if kind == "sinusoidal":
        return SinusoidalPE(d_model, max_seq_len)
    if kind == "nope":
        return NoPE()
    if kind == "rope":
        return RoPE(head_dim, max_seq_len)
    if kind == "alibi":
        return ALiBiPE(n_heads)
    raise ValueError(f"Unknown positional encoding kind: {kind!r}")
