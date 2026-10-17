# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
# Adapted from NVlabs/GatedDeltaNet-2 under the NVIDIA Source Code License-NC.
# See gdn2_ops/LICENSE. Local adapter and PyTorch reference added for this repo.
"""GDN2 with official CUDA training kernels and a portable reference recurrence.

Upstream: https://github.com/NVlabs/GatedDeltaNet-2
Revision: de8476de0fecae08fc16f0bf3154900244c61c73
Widths use this repo's convention: expand_k/v multiply n_embd, not head_dim.
"""

import math
import warnings
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from core.config import ModelConfig
from core.model import Transformer, make_norm
from models.gated_deltanet import GatedDeltaNet, Model as GDNModel


@dataclass
class Config(ModelConfig):
    name: str = "gated_deltanet2"
    norm: str = "rms"
    expand_k: float = 1.0
    expand_v: float = 1.0
    d_conv: int = 4
    conv_bias: bool = False
    use_short_conv: bool = True
    allow_neg_eigval: bool = False
    beta_guard: bool = True
    scan_kernel: str = "auto"  # auto | torch | fla (vendored official GDN2)
    a_init_min: float = 1.0
    a_init_max: float = 16.0
    dt_min: float = 0.001
    dt_max: float = 0.1
    dt_init_floor: float = 1e-4

    def __post_init__(self):
        for name in ("n_embd", "n_layer", "n_head", "d_conv"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"model.{name} must be a positive integer")
        for name in ("expand_k", "expand_v"):
            width = self.n_embd * getattr(self, name)
            if width <= 0 or not float(width).is_integer() or int(width) % self.n_head:
                raise ValueError(f"n_embd * {name} must be positive and divisible by n_head")
        if not 0 < self.a_init_min <= self.a_init_max:
            raise ValueError("require 0 < a_init_min <= a_init_max")
        if not 0 < self.dt_min <= self.dt_max or self.dt_init_floor <= 0:
            raise ValueError("require 0 < dt_min <= dt_max and dt_init_floor > 0")
        if self.scan_kernel not in ("auto", "torch", "fla"):
            raise ValueError("model.scan_kernel must be 'auto', 'torch', or 'fla'")


def guard_erase(k, b, log_decay):
    """Official key-weighted beta-contrast projection with straight-through db."""
    with torch.no_grad():
        key2, erase = k.float().square(), b.float()
        n = key2.sum(-1, keepdim=True)
        a = (erase * key2).sum(-1, keepdim=True)
        nu = (erase.square() * key2).sum(-1, keepdim=True)
        r2 = (n * nu - a.square()).clamp_min(0)
        # The reciprocal form avoids inf/inf for strongly decaying channels.
        gmax = log_decay.float().amax(-1, keepdim=True)
        c2 = (-2 * gmax).exp()
        crit = ((c2 - 1) * (1 - (1 - a).square() * (2 * gmax).exp())).clamp_min(0)
        unsafe = (n > 1e-20) & (r2 > crit)
        mu = a / n.clamp_min(1e-20)
        eta = ((1 - 1 / 32) * crit / r2.clamp_min(1e-30)).sqrt().clamp(0, 1)
        candidate = torch.where(unsafe, mu + eta * (erase - mu), erase).to(b.dtype).float()
        aq = (candidate * key2).sum(-1, keepdim=True)
        nuq = (candidate.square() * key2).sum(-1, keepdim=True)
        r2q = (n * nuq - aq.square()).clamp_min(0)
        critq = ((c2 - 1) * (1 - (1 - aq).square() * (2 * gmax).exp())).clamp_min(0)
        tol = 4 * torch.finfo(b.dtype).eps * (n * nuq + aq.square())
        candidate = torch.where(unsafe & (r2q > critq + tol), aq / n.clamp_min(1e-20), candidate)
    return b + (candidate.to(b.dtype) - b).detach()


def gated_delta2_scan(q, k, v, log_decay, b, w, initial_state=None, beta_guard=True):
    """Normalized q/k [B,T,H,K], v/w [B,T,H,V]; return output and [B,H,K,V] state."""
    if q.shape[1] == 0:
        raise ValueError("GDN2 requires a nonempty sequence")
    dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
    with torch.autocast(device_type=q.device.type, enabled=False):
        if beta_guard:
            b = guard_erase(k, b, log_decay)
        q, k, v, log_decay, b, w = (x.to(dtype) for x in (q, k, v, log_decay, b, w))
        state = (q.new_zeros(q.shape[0], q.shape[2], q.shape[3], v.shape[-1])
                 if initial_state is None else initial_state.to(dtype))
        outputs = []
        for t in range(q.shape[1]):
            state = state * log_decay[:, t].exp()[..., None]
            read = torch.einsum("bhk,bhkv->bhv", b[:, t] * k[:, t], state)
            update = w[:, t] * v[:, t] - read
            state = state + k[:, t, :, :, None] * update[..., None, :]
            outputs.append(torch.einsum("bhk,bhkv->bhv", q[:, t], state))
        return torch.stack(outputs, 1) * q.shape[-1] ** -0.5, state


class GatedDeltaNet2(nn.Module):
    _make_conv = GatedDeltaNet._make_conv
    _causal_conv = staticmethod(GatedDeltaNet._causal_conv)
    _conv_step = staticmethod(GatedDeltaNet._conv_step)

    def __init__(self, config, layer_idx=0):
        super().__init__()
        self.layer_idx = layer_idx
        self.nheads = config.n_head
        self.key_dim = int(config.n_embd * config.expand_k)
        self.value_dim = int(config.n_embd * config.expand_v)
        self.head_k_dim = self.key_dim // self.nheads
        self.head_v_dim = self.value_dim // self.nheads
        self.d_conv = config.d_conv
        self.use_short_conv = config.use_short_conv
        self.allow_neg_eigval = config.allow_neg_eigval
        self.beta_guard = config.beta_guard
        self.scan_kernel = config.scan_kernel
        self._chunk_scan = None
        self._backend_checked = False
        self._backend_reported = False
        for name, width in (("q", self.key_dim), ("k", self.key_dim), ("v", self.value_dim)):
            setattr(self, f"{name}_proj", nn.Linear(config.n_embd, width, bias=config.bias))
            if self.use_short_conv:
                setattr(self, f"{name}_conv", self._make_conv(width, config))
        self.f_proj = nn.Sequential(nn.Linear(config.n_embd, self.head_v_dim, bias=False),
                                    nn.Linear(self.head_v_dim, self.key_dim, bias=False))
        self.b_proj = nn.Linear(config.n_embd, self.key_dim, bias=False)
        self.w_proj = nn.Linear(config.n_embd, self.value_dim, bias=False)
        self.g_proj = nn.Sequential(nn.Linear(config.n_embd, self.head_v_dim, bias=False),
                                    nn.Linear(self.head_v_dim, self.value_dim, bias=True))
        self.o_norm = nn.RMSNorm(self.head_v_dim, eps=1e-5)
        self.c_proj = nn.Linear(self.value_dim, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
        self.A_log = nn.Parameter(torch.empty(self.nheads).uniform_(config.a_init_min, config.a_init_max).log())
        dt = torch.empty(self.key_dim).uniform_(math.log(config.dt_min), math.log(config.dt_max)).exp().clamp_min(config.dt_init_floor)
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))

    @staticmethod
    def _normalize(x):
        # Match FLA's l2norm: epsilon is added to the squared norm.
        return (x.float() * torch.rsqrt(x.float().square().sum(-1, keepdim=True) + 1e-6)).to(x.dtype)

    def _gates(self, x):
        shape = (*x.shape[:2], self.nheads, self.head_k_dim)
        log_decay = (-self.A_log.float().exp().repeat_interleave(self.head_k_dim)
                     * F.softplus(self.f_proj(x).float() + self.dt_bias.float())).reshape(shape)
        b = self.b_proj(x).sigmoid().reshape(shape)
        if self.allow_neg_eigval:
            b = b * 2
        w = self.w_proj(x).sigmoid().reshape(*x.shape[:2], self.nheads, self.head_v_dim)
        return log_decay, b, w

    def _finish(self, y, x):
        gate = self.g_proj(x).reshape(*x.shape[:2], self.nheads, self.head_v_dim)
        # The official fused norm computes the gate product before casting.
        y = (self.o_norm(y.float()) * F.silu(gate.float())).to(gate.dtype).flatten(2)
        return self.dropout(self.c_proj(y))

    def _use_fla(self, x):
        if self.scan_kernel == "torch":
            return False
        reason = None
        if x.device.type != "cuda":
            reason = "requires CUDA"
        elif not 2 <= self.head_k_dim <= 256:
            reason = "requires 2 <= key head dimension <= 256"
        elif not self._backend_checked:
            try:
                from models.gdn2_ops.chunk_gdn2 import chunk_gdn2
                self._chunk_scan = chunk_gdn2
            except Exception as error:
                reason = f"requires compatible fla-core[cuda] and Triton: {error}"
            self._backend_checked = True
            if reason and self.scan_kernel == "auto":
                warnings.warn(f"GDN2 {reason}; using the slower PyTorch recurrence", RuntimeWarning)
        if self.scan_kernel == "fla" and (reason or self._chunk_scan is None):
            raise RuntimeError(f"model.scan_kernel='fla' {reason or 'could not import the official GDN2 kernel'}")
        enabled = x.device.type == "cuda" and reason is None and self._chunk_scan is not None
        if not self._backend_reported and self.layer_idx == 0:
            backend = "official fused CUDA training kernel" if enabled else "PyTorch fallback scan"
            print(f"Gated DeltaNet 2: using {backend}")
            self._backend_reported = True
        return enabled

    def forward(self, x):
        projected = []
        for name, head_dim in (("q", self.head_k_dim), ("k", self.head_k_dim), ("v", self.head_v_dim)):
            value = getattr(self, f"{name}_proj")(x)
            value = (self._causal_conv(value, getattr(self, f"{name}_conv"))
                     if self.use_short_conv else F.silu(value))
            projected.append(value.reshape(*x.shape[:2], self.nheads, head_dim))
        q, k, v = projected
        g, b, w = self._gates(x)
        if self._use_fla(x):
            y, _ = self._chunk_scan(q=q, k=k, v=v, g=g, b=b, w=w,
                                    use_qk_l2norm_in_kernel=True, beta_guard=self.beta_guard)
        else:
            y, _ = gated_delta2_scan(self._normalize(q), self._normalize(k), v, g, b, w,
                                     beta_guard=self.beta_guard)
        return self._finish(y, x)

    def allocate_inference_cache(self, batch_size, max_seqlen=None, dtype=None):
        weight = self.c_proj.weight
        conv = tuple(torch.zeros(batch_size, width, self.d_conv, device=weight.device,
                                  dtype=dtype or weight.dtype)
                     for width in (self.key_dim, self.key_dim, self.value_dim))
        state = torch.zeros(batch_size, self.nheads, self.head_k_dim, self.head_v_dim,
                            device=weight.device, dtype=torch.float32)
        return (*conv, state)

    def step(self, x, cache):
        if x.shape[1] != 1:
            raise ValueError("GatedDeltaNet2.step expects exactly one token")
        projected = []
        for name, head_dim, conv_state in zip(("q", "k", "v"),
                                             (self.head_k_dim, self.head_k_dim, self.head_v_dim), cache[:3]):
            value = getattr(self, f"{name}_proj")(x[:, 0])
            value = (self._conv_step(value, conv_state, getattr(self, f"{name}_conv"))
                     if self.use_short_conv else F.silu(value))
            projected.append(value.reshape(x.shape[0], 1, self.nheads, head_dim))
        q, k, v = projected
        y, state = gated_delta2_scan(self._normalize(q), self._normalize(k), v, *self._gates(x),
                                     initial_state=cache[3], beta_guard=self.beta_guard)
        cache[3].copy_(state)
        return self._finish(y, x)


class GatedDelta2Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.norm = make_norm(config)
        self.mixer = GatedDeltaNet2(config, layer_idx)

    def forward(self, x):
        return x + self.mixer(self.norm(x))


class Model(GDNModel):
    def __init__(self, config):
        Transformer.__init__(self, config, use_pos_embedding=False, positional_encoding="nope",
                             positional_markers=(), block_builder=GatedDelta2Block)
        # Match upstream's mixer initialization after the shared model builder.
        for block in self.transformer.h:
            for module in block.mixer.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight, gain=2 ** -2.5)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
