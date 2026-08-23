"""Configuration dataclasses covering every ablation axis declared in
Section 3.4 of the paper:
  1. softmax approximation
  2. attention pattern (full / local-window, with optional global tokens)
  3. attention approximation -- out of scope here, see note in attention.py
  4. KV/projection design (MHA / GQA / MQA / K=V)
  5. FFN design (ReLU / SwiGLU)
  6. positional encoding (learned absolute / sinusoidal / RoPE / ALiBi / NoPE)
plus the looped/universal transformer baseline (Sec 1.3), which reuses
this same config for its single shared block.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional


@dataclass
class AttentionConfig:
    """Axes 2 and 4: attention pattern + KV/projection design.

    n_kv_heads == n_heads      -> MHA
    n_kv_heads == 1            -> MQA
    1 < n_kv_heads < n_heads   -> GQA
    share_kv = True            -> K=V ablation

    pattern="full"   -> standard causal attention, every past token visible
    pattern="local"  -> only the last `window_size` tokens are visible,
                         plus the first `n_global_tokens` tokens (if any),
                         which every query can always attend to regardless
                         of distance (a minimal "attention sink" analogue).
    """
    n_heads: int
    n_kv_heads: int
    share_kv: bool = False
    head_dim: int = 64
    pattern: Literal["full", "local"] = "full"
    window_size: Optional[int] = None   # required if pattern == "local"
    n_global_tokens: int = 0            # only used if pattern == "local"
    softmax_kind: Literal["standard", "taylor"] = "standard"

    def __post_init__(self) -> None:
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError(
                f"n_heads ({self.n_heads}) must be divisible by n_kv_heads ({self.n_kv_heads})"
            )
        if self.n_kv_heads > self.n_heads:
            raise ValueError("n_kv_heads cannot exceed n_heads.")
        if self.pattern == "local" and self.window_size is None:
            raise ValueError("window_size is required when pattern='local'")

    @property
    def variant_name(self) -> str:
        if self.n_kv_heads == self.n_heads:
            base = "mha"
        elif self.n_kv_heads == 1:
            base = "mqa"
        else:
            base = f"gqa{self.n_kv_heads}"
        if self.share_kv:
            base += "_kv"
        if self.pattern == "local":
            base += f"_local{self.window_size}"
        if self.softmax_kind != "standard":
            base += f"_{self.softmax_kind}"
        return base


@dataclass
class FFNConfig:
    """Axis 5: FFN design. hidden_dim should come from
    model.ffn.compute_iso_param_hidden_dim for fair ReLU-vs-SwiGLU
    comparisons (SwiGLU has 3 weight matrices instead of ReLU's 2)."""
    kind: Literal["relu", "swiglu"]
    hidden_dim: int


@dataclass
class ModelConfig:
    d_model: int
    n_layers: int
    vocab_size: int
    attention: AttentionConfig
    ffn: FFNConfig
    positional_encoding: Literal[
        "learned_absolute", "sinusoidal", "rope", "alibi", "nope"
    ] = "learned_absolute"
    max_seq_len: int = 2048
    dropout: float = 0.0

    @property
    def variant_name(self) -> str:
        return f"{self.attention.variant_name}_{self.ffn.kind}_{self.positional_encoding}"


@dataclass
class LoopConfig:
    """Extra config for the looped/universal transformer baseline
    (Sec 1.3): a single shared block applied k_iters times, with a
    learnable per-iteration timestep embedding."""
    k_iters: int


