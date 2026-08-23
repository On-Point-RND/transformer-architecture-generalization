"""Unified attention module covering MHA / GQA / MQA / K=V, attention
pattern (full/local), and softmax variant (standard/taylor), plus hooks
for RoPE/ALiBi via the shared PositionalEncoding instance.

Fast path: whenever softmax_kind == "standard", this dispatches to
F.scaled_dot_product_attention instead of the manual
matmul -> add-mask -> softmax -> matmul sequence. SDPA computes the
exact same math (it's just a fused kernel), but on CUDA it can pick a
FlashAttention/memory-efficient backend, which is both faster and
avoids ever materializing the full (B, H, T, T) score matrix in memory
-- the thing that was causing OOM on our longest KV sequences. The
manual path is kept only for softmax_kind == "taylor", since SDPA has
no hook for a custom score-to-weight transform.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .config import AttentionConfig
from .positional_encoding import PositionalEncoding


def _taylor_softmax(scores: Tensor, mask_bias: Tensor) -> Tensor:
    x = scores + mask_bias
    approx = (1.0 + x + 0.5 * x.pow(2)).clamp(min=1e-6)
    approx = torch.where(torch.isfinite(x), approx, torch.zeros_like(approx))
    denom = approx.sum(dim=-1, keepdim=True).clamp(min=1e-6)
    return approx / denom


def _build_pattern_mask(
    seq_len: int, pattern: str, window_size: int | None, n_global_tokens: int, device: torch.device
) -> Tensor:
    causal_forbidden = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=1)

    if pattern == "full":
        forbidden = causal_forbidden
    elif pattern == "local":
        i = torch.arange(seq_len, device=device).unsqueeze(1)
        j = torch.arange(seq_len, device=device).unsqueeze(0)
        too_far = (i - j) >= window_size
        forbidden = causal_forbidden | too_far
        if n_global_tokens > 0:
            is_global_key = torch.zeros(seq_len, dtype=torch.bool, device=device)
            is_global_key[:n_global_tokens] = True
            forbidden = forbidden & ~is_global_key.unsqueeze(0)
            forbidden = forbidden | causal_forbidden
    else:
        raise ValueError(f"Unknown attention pattern: {pattern!r}")

    mask = torch.zeros(seq_len, seq_len, device=device)
    mask.masked_fill_(forbidden, float("-inf"))
    return mask


class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model: int, config: AttentionConfig, pos_enc: PositionalEncoding):
        super().__init__()
        self.config = config
        self.pos_enc = pos_enc
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.head_dim
        self.n_rep = self.n_heads // self.n_kv_heads
        self._has_sdpa = hasattr(F, "scaled_dot_product_attention")
        if config.softmax_kind == "standard" and not self._has_sdpa:
            print("WARNING: F.scaled_dot_product_attention unavailable, falling back to manual attention "
                  "(requires PyTorch >= 2.0 for the fast path).")

        q_out = self.n_heads * self.head_dim
        kv_out = self.n_kv_heads * self.head_dim
        self.w_q = nn.Linear(d_model, q_out, bias=False)
        self.w_k = nn.Linear(d_model, kv_out, bias=False)
        self.w_v = None if config.share_kv else nn.Linear(d_model, kv_out, bias=False)
        self.w_o = nn.Linear(q_out, d_model, bias=False)

    def _repeat_kv(self, x: Tensor) -> Tensor:
        if self.n_rep == 1:
            return x
        b, h_kv, t, d = x.shape
        x = x.unsqueeze(2).expand(b, h_kv, self.n_rep, t, d)
        return x.reshape(b, h_kv * self.n_rep, t, d)

    def forward(self, x: Tensor, positions: Tensor, key_padding_mask: Tensor | None = None) -> Tensor:
        """key_padding_mask: optional (B, T) bool, True at PAD positions.
        Needed for batched generation, where prompts of different lengths
        get padded to a common width -- without this, real tokens would
        silently attend to PAD embeddings, something the model never saw
        during training (there, padding only ever appears after the end
        of a sequence, where causal masking already hides it for free)."""
        b, t, _ = x.shape
        device = x.device

        q = self.w_q(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.w_k(x).view(b, t, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v_src = self.w_k(x) if self.w_v is None else self.w_v(x)
        v = v_src.view(b, t, self.n_kv_heads, self.head_dim).transpose(1, 2)

        k = self._repeat_kv(k)
        v = self._repeat_kv(v)

        q, k = self.pos_enc.rotate_qk(q, k, positions)

        pattern_mask = _build_pattern_mask(
            t, self.config.pattern, self.config.window_size, self.config.n_global_tokens, device
        )
        alibi_bias = self.pos_enc.attention_bias(t, self.n_heads, device)
        mask_bias = pattern_mask.unsqueeze(0) if alibi_bias is None else pattern_mask.unsqueeze(0) + alibi_bias
        # mask_bias: (1, T, T) with no ALiBi, or (n_heads, T, T) with ALiBi --
        # both broadcast correctly against a (B, n_heads, T, T) score tensor.
        mask_bias = mask_bias.unsqueeze(0).to(q.dtype)  # (1, 1_or_H, T, T)

        if key_padding_mask is not None:
            # (B, T) -> (B, 1, 1, T): forbid attending TO a padded key,
            # for every query row and every head. Broadcasts against
            # mask_bias's (1, 1_or_H, T, T) to give (B, 1_or_H, T, T).
            pad_bias = torch.zeros(b, 1, 1, t, device=device, dtype=q.dtype)
            pad_bias.masked_fill_(key_padding_mask[:, None, None, :], float("-inf"))
            mask_bias = mask_bias + pad_bias

        if self.config.softmax_kind == "standard" and self._has_sdpa:
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask_bias, is_causal=False)
        elif self.config.softmax_kind in ("standard", "taylor"):
            scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            if self.config.softmax_kind == "standard":
                attn = F.softmax(scores + mask_bias, dim=-1)
            else:
                attn = _taylor_softmax(scores, mask_bias)
            out = attn @ v
        else:
            raise ValueError(f"Unknown softmax_kind: {self.config.softmax_kind!r}")

        out = out.transpose(1, 2).contiguous().view(b, t, self.n_heads * self.head_dim)
        return self.w_o(out)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
