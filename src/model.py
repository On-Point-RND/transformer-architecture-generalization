"""Decoder-only transformer (nanoGPT-style), implemented from scratch.

FIXED architecture choices for this subproject (do NOT vary — they belong to
other subprojects):
  - learned absolute positional embeddings
  - standard softmax (scaled dot-product) causal attention
  - standard MLP FFN with GELU
  - pre-LayerNorm

Derived defaults: d_ff = 4 * d_model; n_heads = d_model // 32 (see budgets.py
for the exact head-count rule used when d_model is not a multiple of 32).

A LOOPED variant is provided: a single shared transformer block applied T
times (weight-shared). T can be given at construction and overridden per
forward call.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def default_n_heads(d_model: int) -> int:
    """Largest divisor of d_model giving a head dimension closest to 32."""
    best_h, best_err = 1, float("inf")
    for h in range(1, d_model + 1):
        if d_model % h == 0:
            err = abs((d_model // h) - 32)
            if err < best_err:
                best_err, best_h = err, h
    return best_h


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=True)
        self.proj = nn.Linear(d_model, d_model, bias=True)
        self.attn_dropout = dropout
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        # (B, T, C) -> (B, n_heads, T, head_dim)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        y = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_dropout if self.training else 0.0,
            is_causal=True,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.proj(y))


class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff, bias=True)
        self.fc2 = nn.Linear(d_ff, d_model, bias=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.fc2(F.gelu(self.fc1(x))))


class Block(nn.Module):
    """Pre-LN transformer block."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, d_ff, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    """Decoder-only transformer.

    Args:
        n_layers:   number of transformer blocks L (distinct blocks when not
                    looped; ignored for block count when looped=True).
        d_model:    model width.
        n_heads:    attention heads (default: default_n_heads(d_model)).
        d_ff:       FFN hidden dim (default: 4 * d_model).
        vocab_size: token vocabulary size.
        max_seq_len:maximum sequence length (for the positional embedding).
        dropout:    dropout probability.
        looped:     if True, use a single shared block applied n_loops times.
        n_loops:    default number of iterations for the looped variant (T).
    """

    def __init__(
        self,
        n_layers: int,
        d_model: int,
        vocab_size: int,
        max_seq_len: int,
        n_heads: Optional[int] = None,
        d_ff: Optional[int] = None,
        dropout: float = 0.0,
        looped: bool = False,
        n_loops: int = 1,
    ):
        super().__init__()
        if n_heads is None:
            n_heads = default_n_heads(d_model)
        if d_ff is None:
            d_ff = 4 * d_model

        self.n_layers = n_layers
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.looped = looped
        self.n_loops = n_loops

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.drop = nn.Dropout(dropout)

        if looped:
            self.block = Block(d_model, n_heads, d_ff, dropout)
            self.blocks = None
        else:
            self.blocks = nn.ModuleList(
                [Block(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
            )
            self.block = None

        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        # Weight tying (nanoGPT-style).
        self.lm_head.weight = self.tok_emb.weight

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self, non_embedding: bool = False) -> int:
        """Total trainable parameters. With tied weights, lm_head adds none."""
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.pos_emb.weight.numel()
        return n

    @torch.no_grad()
    def hidden_states(self, input_ids: torch.Tensor):
        """Raw representation after each block (BEFORE final LN / head), one per
        layer. For an honest per-layer linear probe (logit lens)."""
        B, T = input_ids.shape
        pos = torch.arange(T, device=input_ids.device)
        x = self.tok_emb(input_ids) + self.pos_emb(pos)[None, :, :]
        x = self.drop(x)
        outs = []
        if self.looped:
            for _ in range(self.n_loops):
                x = self.block(x)
                outs.append(x)
        else:
            for blk in self.blocks:
                x = blk(x)
                outs.append(x)
        return outs  # list of (B, T, d_model)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        n_loops: Optional[int] = None,
        ignore_index: int = -100,
        stop_at_layer: Optional[int] = None,
    ):
        """stop_at_layer=k runs only the first k blocks (early-exit probe):
        embedding -> blocks[:k] -> final LN -> head. None = full depth. For a
        looped model it caps the number of iterations instead."""
        B, T = input_ids.shape
        assert T <= self.max_seq_len, f"seq len {T} > max_seq_len {self.max_seq_len}"
        pos = torch.arange(T, device=input_ids.device)
        x = self.tok_emb(input_ids) + self.pos_emb(pos)[None, :, :]
        x = self.drop(x)

        if self.looped:
            T_iter = self.n_loops if n_loops is None else n_loops
            if stop_at_layer is not None:
                T_iter = min(T_iter, stop_at_layer)
            for _ in range(T_iter):
                x = self.block(x)
        else:
            blocks = self.blocks if stop_at_layer is None else self.blocks[:stop_at_layer]
            for blk in blocks:
                x = blk(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=ignore_index,
            )
        return logits, loss


if __name__ == "__main__":
    # Quick self-test.
    torch.manual_seed(0)
    m = GPT(n_layers=4, d_model=128, vocab_size=15, max_seq_len=64)
    x = torch.randint(0, 15, (8, 29))
    logits, loss = m(x)
    print("logits", tuple(logits.shape), "params", m.num_params())
    ml = GPT(n_layers=1, d_model=128, vocab_size=15, max_seq_len=64,
             looped=True, n_loops=4)
    lg, _ = ml(x, n_loops=6)
    print("looped logits", tuple(lg.shape), "params", ml.num_params())
