"""A small, dependency-free Gated DeltaNet.

References:
    https://github.com/NVlabs/GatedDeltaNet
    https://github.com/fla-org/flash-linear-attention/blob/main/fla/layers/gated_deltanet.py
    https://arxiv.org/abs/2411.12537

The state is a fast-weight matrix for every head.  At each token we decay it,
measure its prediction error for the current key/value pair, and write that
error back with the delta rule::

    S = alpha * S
    S = S + beta * k outer (v - k @ S)
    y = q @ S

With unit-length keys, the transition has eigenvalue ``alpha * (1 - beta)``
in the key direction.  The usual ``beta = sigmoid(.)`` keeps it positive;
``allow_neg_eigval`` uses ``beta = 2 * sigmoid(.)`` and extends it to
``(-alpha, alpha)`` without changing the recurrence.
"""

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from core.config import ModelConfig
from core.model import Transformer


@dataclass
class Config(ModelConfig):
    name: str = "gated_deltanet"
    expand_k: float = 0.75
    expand_v: float = 1.5
    d_conv: int = 4
    conv_bias: bool = False
    allow_neg_eigval: bool = False
    a_init_min: float = 1.0
    a_init_max: float = 16.0
    dt_min: float = 0.001
    dt_max: float = 0.1
    dt_init_floor: float = 1e-4

    def __post_init__(self):
        for name in ("n_embd", "n_layer", "n_head", "d_conv"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"model.{name} must be a positive integer")
        if self.expand_k <= 0 or self.expand_v <= 0:
            raise ValueError("model.expand_k and model.expand_v must be positive")
        for name, width in (
            ("expand_k", self.n_embd * self.expand_k),
            ("expand_v", self.n_embd * self.expand_v),
        ):
            if not float(width).is_integer() or int(width) % self.n_head:
                raise ValueError(
                    f"model.n_embd * model.{name} must be an integer divisible by n_head"
                )
        if not 0 < self.a_init_min <= self.a_init_max:
            raise ValueError("require 0 < a_init_min <= a_init_max")
        if not 0 < self.dt_min <= self.dt_max or self.dt_init_floor <= 0:
            raise ValueError("require 0 < dt_min <= dt_max and dt_init_floor > 0")


def gated_delta_scan(q, k, v, decay, beta, initial_state=None, seq_idx=None):
    """Literal recurrent form of the gated delta rule.

    Shapes are ``q/k: [B,T,H,K]``, ``v: [B,T,H,V]`` and gates ``[B,T,H]``.
    Returning the final state keeps the same function useful for cached decode.
    """
    if q.shape[1] == 0:
        raise ValueError("Gated DeltaNet requires a nonempty sequence")
    dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
    with torch.autocast(device_type=q.device.type, enabled=False):
        q, k, v, decay, beta = (x.to(dtype) for x in (q, k, v, decay, beta))
        batch, length, nheads, key_dim = q.shape
        value_dim = v.shape[-1]
        state = (
            q.new_zeros(batch, nheads, key_dim, value_dim)
            if initial_state is None
            else initial_state.to(dtype)
        )
        outputs = []
        for t in range(length):
            if seq_idx is not None and t:
                reset = seq_idx[:, t] != seq_idx[:, t - 1]
                fresh = q.new_zeros(state.shape)
                state = torch.where(reset[:, None, None, None], fresh, state)
            state = state * decay[:, t, :, None, None]
            prediction = torch.einsum("bhk,bhkv->bhv", k[:, t], state)
            error = beta[:, t, :, None] * (v[:, t] - prediction)
            state = state + torch.einsum("bhk,bhv->bhkv", k[:, t], error)
            outputs.append(torch.einsum("bhk,bhkv->bhv", q[:, t], state))
        return torch.stack(outputs, dim=1), state


class GatedDeltaNet(nn.Module):
    """Causal ``[batch, time, width]`` mixer, written to expose the algorithm."""

    def __init__(self, config, layer_idx=0):
        super().__init__()
        self.layer_idx = layer_idx
        self.nheads = config.n_head
        self.key_dim = int(config.n_embd * config.expand_k)
        self.value_dim = int(config.n_embd * config.expand_v)
        self.head_k_dim = self.key_dim // self.nheads
        self.head_v_dim = self.value_dim // self.nheads
        self.d_conv = config.d_conv
        self.allow_neg_eigval = config.allow_neg_eigval

        self.q_proj = nn.Linear(config.n_embd, self.key_dim, bias=config.bias)
        self.k_proj = nn.Linear(config.n_embd, self.key_dim, bias=config.bias)
        self.v_proj = nn.Linear(config.n_embd, self.value_dim, bias=config.bias)
        self.q_conv = self._make_conv(self.key_dim, config)
        self.k_conv = self._make_conv(self.key_dim, config)
        self.v_conv = self._make_conv(self.value_dim, config)

        self.a_proj = nn.Linear(config.n_embd, self.nheads, bias=False)
        self.b_proj = nn.Linear(config.n_embd, self.nheads, bias=False)
        self.g_proj = nn.Linear(config.n_embd, self.value_dim, bias=False)
        self.o_norm = nn.RMSNorm(self.head_v_dim, eps=1e-5)
        self.c_proj = nn.Linear(self.value_dim, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

        self.A_log = nn.Parameter(
            torch.empty(self.nheads)
            .uniform_(config.a_init_min, config.a_init_max)
            .log()
        )
        dt = (
            torch.empty(self.nheads)
            .uniform_(math.log(config.dt_min), math.log(config.dt_max))
            .exp()
            .clamp_min(config.dt_init_floor)
        )
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))

    def _make_conv(self, width, config):
        return nn.Conv1d(
            width,
            width,
            config.d_conv,
            groups=width,
            padding=config.d_conv - 1,
            bias=config.conv_bias,
        )

    @staticmethod
    def _causal_conv(x, conv):
        x = conv(x.transpose(1, 2))[..., : x.shape[1]].transpose(1, 2)
        return F.silu(x)

    def _project(self, x):
        q = self._causal_conv(self.q_proj(x), self.q_conv)
        k = self._causal_conv(self.k_proj(x), self.k_conv)
        v = self._causal_conv(self.v_proj(x), self.v_conv)
        shape_k = (*q.shape[:2], self.nheads, self.head_k_dim)
        shape_v = (*v.shape[:2], self.nheads, self.head_v_dim)
        q = F.normalize(q.reshape(shape_k).float(), dim=-1, eps=1e-6).to(q.dtype)
        k = F.normalize(k.reshape(shape_k).float(), dim=-1, eps=1e-6).to(k.dtype)
        return q, k, v.reshape(shape_v)

    def _gates(self, x):
        dt = F.softplus(self.a_proj(x).float() + self.dt_bias.float())
        decay = torch.exp(-self.A_log.float().exp() * dt)
        beta = self.b_proj(x).float().sigmoid()
        if self.allow_neg_eigval:
            beta = 2.0 * beta
        return decay, beta

    def _finish(self, y, gate):
        y = self.o_norm(y).flatten(2).to(gate.dtype) * F.silu(gate)
        return self.dropout(self.c_proj(y))

    def forward(self, x, seq_idx=None):
        q, k, v = self._project(x)
        decay, beta = self._gates(x)
        y, _ = gated_delta_scan(q, k, v, decay, beta, seq_idx=seq_idx)
        return self._finish(y, self.g_proj(x))

    def allocate_inference_cache(self, batch_size, max_seqlen=None, dtype=None):
        device = self.c_proj.weight.device
        dtype = dtype or self.c_proj.weight.dtype
        conv_states = tuple(
            torch.zeros(batch_size, width, self.d_conv, device=device, dtype=dtype)
            for width in (self.key_dim, self.key_dim, self.value_dim)
        )
        recurrent_state = torch.zeros(
            batch_size,
            self.nheads,
            self.head_k_dim,
            self.head_v_dim,
            device=device,
            dtype=dtype,
        )
        return (*conv_states, recurrent_state)

    @staticmethod
    def _conv_step(x, state, conv):
        state.copy_(torch.roll(state, shifts=-1, dims=-1))
        state[:, :, -1] = x
        y = (state * conv.weight[:, 0]).sum(-1)
        if conv.bias is not None:
            y = y + conv.bias
        return F.silu(y)

    def step(self, x, cache):
        """Decode one token and update Q/K/V convolution and memory caches."""
        if x.shape[1] != 1:
            raise ValueError("GatedDeltaNet.step expects exactly one token")
        q_state, k_state, v_state, state = cache
        token = x[:, 0]
        q = self._conv_step(self.q_proj(token), q_state, self.q_conv)
        k = self._conv_step(self.k_proj(token), k_state, self.k_conv)
        v = self._conv_step(self.v_proj(token), v_state, self.v_conv)
        q = F.normalize(
            q.reshape(-1, self.nheads, self.head_k_dim).float(), dim=-1, eps=1e-6
        )
        k = F.normalize(
            k.reshape(-1, self.nheads, self.head_k_dim).float(), dim=-1, eps=1e-6
        )
        v = v.reshape(-1, self.nheads, self.head_v_dim).float()
        decay, beta = self._gates(x)
        state.mul_(decay[:, 0, :, None, None])
        prediction = torch.einsum("bhk,bhkv->bhv", k, state.float())
        error = beta[:, 0, :, None] * (v - prediction)
        state.add_(torch.einsum("bhk,bhv->bhkv", k, error).to(state.dtype))
        y = torch.einsum("bhk,bhkv->bhv", q, state.float())[:, None]
        return self._finish(y, self.g_proj(x))


class GatedDeltaBlock(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.norm = nn.RMSNorm(config.n_embd, eps=1e-5)
        self.mixer = GatedDeltaNet(config, layer_idx)

    def forward(self, x):
        return x + self.mixer(self.norm(x))


class Model(Transformer):
    def __init__(self, config):
        super().__init__(config)
        self.transformer.ln_f = nn.RMSNorm(config.n_embd, eps=1e-5)

    def build_blocks(self, config):
        return nn.ModuleList(
            [GatedDeltaBlock(config, i) for i in range(config.n_layer)]
        )

    def uses_pos_embedding(self, config):
        return False

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        return -1.0

    def _step_token(self, token, caches):
        x = self.transformer.drop(self.transformer.wte(token))
        for block, cache in zip(self.transformer.h, caches):
            x = x + block.mixer.step(block.norm(x), cache)
        return self.lm_head(self.transformer.ln_f(x))

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        if idx.size(1) == 0:
            raise ValueError("Gated DeltaNet generation requires a nonempty prompt")
        caches = [
            block.mixer.allocate_inference_cache(idx.size(0))
            for block in self.transformer.h
        ]
        logits = None
        for position in range(idx.size(1)):
            logits = self._step_token(idx[:, position : position + 1], caches)
        for step in range(max_new_tokens):
            scores = logits[:, -1] / temperature
            if top_k is not None:
                values, _ = torch.topk(scores, min(top_k, scores.size(-1)))
                scores[scores < values[:, [-1]]] = -float("inf")
            next_token = torch.multinomial(F.softmax(scores, dim=-1), 1)
            idx = torch.cat((idx, next_token), dim=1)
            if step + 1 < max_new_tokens:
                logits = self._step_token(next_token, caches)
        return idx
