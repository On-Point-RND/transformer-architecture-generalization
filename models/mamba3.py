"""Mamba-3 SISO/MIMO with official CUDA training kernels and a torch reference."""

import math
import warnings
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from core.config import ModelConfig
from core.model import Transformer, make_norm
from models.gated_deltanet import Model as CachedModel


@dataclass
class Config(ModelConfig):
    name: str = "mamba3"
    norm: str = "rms"
    d_state: int = 128
    expand: int = 2
    headdim: int = 64
    ngroups: int = 1
    rope_fraction: float = 0.5
    dt_min: float = 0.001
    dt_max: float = 0.1
    dt_init_floor: float = 1e-4
    a_floor: float = 1e-4
    is_outproj_norm: bool = False
    is_mimo: bool = False
    mimo_rank: int = 4
    chunk_size: int = 0  # 0 = 64 (SISO), 64 / rank (MIMO)
    ssm_kernel: str = "auto"  # auto | torch | fused

    def __post_init__(self):
        for name in ("n_embd", "n_layer", "d_state", "expand", "headdim", "ngroups", "mimo_rank"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"model.{name} must be a positive integer")
        inner = self.n_embd * self.expand
        if inner % self.headdim or (inner // self.headdim) % self.ngroups:
            raise ValueError("expand * n_embd must be divisible by headdim, and SSM heads by ngroups")
        if self.rope_fraction not in (0.5, 1.0) or self.d_state % 4:
            raise ValueError("rope_fraction must be 0.5 or 1.0 and d_state must be divisible by 4")
        if self.d_state < 8:
            raise ValueError("d_state must be at least 8 for the official rotary kernels")
        if self.is_mimo and self.mimo_rank not in (1, 2, 4, 8):
            raise ValueError("MIMO rank must be 1, 2, 4, or 8")
        if type(self.chunk_size) is not int or self.chunk_size < 0:
            raise ValueError("chunk_size must be 0 or a power of two >= 8")
        if self.chunk_size and (self.chunk_size < 8 or self.chunk_size & (self.chunk_size - 1)):
            raise ValueError("chunk_size must be 0 or a power of two >= 8")
        if not 0 < self.dt_min <= self.dt_max or self.dt_init_floor <= 0 or self.a_floor <= 0:
            raise ValueError("require 0 < dt_min <= dt_max, dt_init_floor > 0, and a_floor > 0")
        if self.ssm_kernel not in ("auto", "torch", "fused"):
            raise ValueError("ssm_kernel must be 'auto', 'torch', or 'fused'")


def heavy_tail_activation(x):
    return x.clamp_min(0) + (1 - x.clamp_max(0)).reciprocal()


def rotary(x, angle, is_mimo=False):
    """Upstream SISO uses adjacent pairs; MIMO pairs across the state halves."""
    cos, sin = angle.cos().unsqueeze(1), angle.sin().unsqueeze(1)
    count = angle.shape[-1]
    if is_mimo:
        left, right = x.chunk(2, dim=-1)
    else:
        left, right = x[..., 0::2], x[..., 1::2]
    a, b = left[..., :count], right[..., :count]
    left = torch.cat((a * cos - b * sin, left[..., count:]), -1)
    right = torch.cat((a * sin + b * cos, right[..., count:]), -1)
    return torch.cat((left, right), -1) if is_mimo else torch.stack((left, right), -1).flatten(-2)


def mamba3_scan(q, k, v, adt, dt, trap, angles, q_bias, k_bias, mimo_x,
                initial_state=None, is_mimo=False):
    """Ungated Mamba-3 recurrence; q/k [B,T,R,G,N], v [B,T,H,P].

    Returns [B,T,R,H,P] and (angle, SSM, previous rotated key, previous value).
    The previous key/value term is essential to the trapezoidal discretization.
    """
    if v.shape[1] == 0:
        raise ValueError("Mamba3 requires a nonempty sequence")
    dtype = torch.float64 if v.dtype == torch.float64 else torch.float32
    with torch.autocast(device_type=v.device.type, enabled=False):
        q, k, v, adt, dt, trap, angles, q_bias, k_bias, mimo_x = (
            x.to(dtype) for x in (q, k, v, adt, dt, trap, angles, q_bias, k_bias, mimo_x))
        batch, length, heads, width = v.shape
        rank, state_width = q.shape[2], q.shape[-1]
        q = q.repeat_interleave(heads // q.shape[3], dim=3) + q_bias.permute(1, 0, 2)[None, None]
        k = k.repeat_interleave(heads // k.shape[3], dim=3) + k_bias.permute(1, 0, 2)[None, None]
        if initial_state is None:
            angle = v.new_zeros(batch, heads, angles.shape[-1])
            state = v.new_zeros(batch, heads, width, state_width)
            previous_k = v.new_zeros(batch, rank, heads, state_width)
            previous_v = v.new_zeros(batch, heads, width)
        else:
            angle, state, previous_k, previous_v = (x.to(dtype) for x in initial_state)
        output = []
        for t in range(length):
            angle = angle + math.pi * angles[:, t].tanh() * dt[:, t, :, None]
            if not is_mimo:
                angle = angle.remainder(2 * math.pi)
            qt = rotary(q[:, t], angle, is_mimo)
            kt = rotary(k[:, t], angle, is_mimo)
            vt = v[:, t, None] * mimo_x.permute(1, 0, 2)[None]
            prev_vt = previous_v[:, None] * mimo_x.permute(1, 0, 2)[None]
            alpha, mix = adt[:, t].exp(), trap[:, t].sigmoid()
            beta, gamma = (1 - mix) * dt[:, t] * alpha, mix * dt[:, t]
            prev_kv = torch.einsum("brhn,brhp->bhpn", previous_k, prev_vt)
            kv = torch.einsum("brhn,brhp->bhpn", kt, vt)
            state = (alpha[..., None, None] * state + beta[..., None, None] * prev_kv
                     + gamma[..., None, None] * kv)
            output.append(torch.einsum("bhpn,brhn->brhp", state, qt))
            previous_k, previous_v = kt, v[:, t]
        return torch.stack(output, 1), (angle, state, previous_k, previous_v)


class Mamba3(nn.Module):
    def __init__(self, config, layer_idx=0):
        super().__init__()
        self.layer_idx = layer_idx
        self.d_state, self.headdim, self.ngroups = config.d_state, config.headdim, config.ngroups
        self.d_inner = config.n_embd * config.expand
        self.nheads = self.d_inner // self.headdim
        self.is_mimo = config.is_mimo
        self.rank = config.mimo_rank if config.is_mimo else 1
        self.num_angles = int(config.d_state * config.rope_fraction) // 2
        self.rotary_dim_divisor = int(2 / config.rope_fraction)
        self.chunk_size = config.chunk_size or 64 // self.rank
        self.a_floor = config.a_floor
        self.is_outproj_norm = config.is_outproj_norm
        self.ssm_kernel = config.ssm_kernel
        self._fused_scan = None
        self._import_attempted = False
        self._import_error = None
        self._reported_backend = None
        bc_width = 2 * self.d_state * self.ngroups * self.rank
        self.in_proj = nn.Linear(config.n_embd, 2 * self.d_inner + bc_width + 3 * self.nheads + self.num_angles, bias=False)
        self.B_norm, self.C_norm = nn.RMSNorm(self.d_state, eps=1e-5), nn.RMSNorm(self.d_state, eps=1e-5)
        self.B_bias = nn.Parameter(torch.ones(self.nheads, self.rank, self.d_state))
        self.C_bias = nn.Parameter(torch.ones(self.nheads, self.rank, self.d_state))
        dt = torch.empty(self.nheads).uniform_(math.log(config.dt_min), math.log(config.dt_max)).exp().clamp_min(config.dt_init_floor)
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))
        self.D = nn.Parameter(torch.ones(self.nheads))
        if self.is_mimo:
            self.mimo_x = nn.Parameter(torch.ones(self.nheads, self.rank, self.headdim) / self.rank)
            self.mimo_z = nn.Parameter(torch.ones(self.nheads, self.rank, self.headdim))
            self.mimo_o = nn.Parameter(torch.ones(self.nheads, self.rank, self.headdim) / self.rank)
        else:
            self.register_buffer("mimo_x", torch.ones(self.nheads, 1, self.headdim), persistent=False)
            self.register_buffer("mimo_z", torch.ones(self.nheads, 1, self.headdim), persistent=False)
            self.register_buffer("mimo_o", torch.ones(self.nheads, 1, self.headdim), persistent=False)
        if self.is_outproj_norm:
            self.out_norm_weight = nn.Parameter(torch.ones(self.nheads, self.headdim))
        self.c_proj = nn.Linear(self.d_inner, config.n_embd, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def _project(self, x):
        bc_width = self.d_state * self.ngroups * self.rank
        z, v, b, c, dt, a, trap, angles = self.in_proj(x).split(
            (self.d_inner, self.d_inner, bc_width, bc_width,
             self.nheads, self.nheads, self.nheads, self.num_angles), -1)
        shape = (*x.shape[:2], self.rank, self.ngroups, self.d_state)
        b, c = self.B_norm(b.reshape(shape)), self.C_norm(c.reshape(shape))
        dt = F.softplus(dt.float() + self.dt_bias.float())
        adt = -heavy_tail_activation(a.float()).clamp_min(self.a_floor) * dt
        angles = angles.float().unsqueeze(-2).expand(-1, -1, self.nheads, -1)
        shape_v = (*x.shape[:2], self.nheads, self.headdim)
        return z.reshape(shape_v), c, b, v.reshape(shape_v), adt, dt, trap, angles

    def _use_fused(self, x):
        reason = None
        if self.ssm_kernel == "torch":
            enabled = False
        elif x.device.type != "cuda":
            reason, enabled = "requires CUDA", False
        else:
            if not self._import_attempted:
                try:
                    if self.is_mimo:
                        from mamba_ssm.ops.tilelang.mamba3.mamba3_mimo import mamba3_mimo
                        self._fused_scan = mamba3_mimo
                    else:
                        from mamba_ssm.ops.triton.mamba3.mamba3_siso_combined import mamba3_siso_combined
                        self._fused_scan = mamba3_siso_combined
                except Exception as error:
                    self._import_error = str(error)
                self._import_attempted = True
                if self._fused_scan is None and self.ssm_kernel == "auto":
                    warnings.warn(f"Mamba3 CUDA kernel unavailable; using slower torch recurrence: {self._import_error}", RuntimeWarning)
            enabled = self._fused_scan is not None
            if not enabled:
                reason = f"requires the official Mamba3 source install: {self._import_error}"
        if self.ssm_kernel == "fused" and not enabled:
            raise RuntimeError(f"model.ssm_kernel='fused' {reason}")
        if self.layer_idx == 0 and self._reported_backend != enabled:
            kind = "MIMO TileLang" if self.is_mimo else "SISO Triton"
            print(f"Mamba3: using {kind + ' fused CUDA kernel' if enabled else 'PyTorch fallback scan'}")
            self._reported_backend = enabled
        return enabled

    def _scan(self, c, b, v, adt, dt, trap, angles):
        if not self._use_fused(v):
            y, _ = mamba3_scan(c, b, v, adt, dt, trap, angles, self.C_bias, self.B_bias,
                               self.mimo_x, is_mimo=self.is_mimo)
            return y + self.D.float()[None, None, None, :, None] * v.float()[:, :, None] * self.mimo_x.float().permute(1, 0, 2)[None, None]
        kwargs = dict(Q=c, K=b, V=v, ADT=adt.transpose(1, 2), DT=dt.transpose(1, 2),
                      Trap=trap.transpose(1, 2), Angles=angles, D=self.D, Z=None,
                      chunk_size=self.chunk_size)
        if self.is_mimo:
            return self._fused_scan(**kwargs, Q_bias=self.C_bias, K_bias=self.B_bias,
                                    MIMO_V=self.mimo_x, MIMO_Z=self.mimo_z, MIMO_Out=None,
                                    rotary_dim_divisor=self.rotary_dim_divisor, dtype=v.dtype)
        kwargs.update(Q=c.squeeze(2), K=b.squeeze(2))
        return self._fused_scan(**kwargs, Q_bias=self.C_bias.squeeze(1), K_bias=self.B_bias.squeeze(1)).unsqueeze(2)

    def _finish(self, y, z):
        with torch.autocast(device_type=y.device.type, enabled=False):
            y = y.float()
            if self.is_outproj_norm:
                y = y * torch.rsqrt(y.square().mean(-1, keepdim=True) + 1e-5)
                y = y * self.out_norm_weight.float()[None, None, None]
            gate = z.float()[:, :, None] * self.mimo_z.float().permute(1, 0, 2)[None, None]
            y = (y * F.silu(gate) * self.mimo_o.float().permute(1, 0, 2)[None, None]).sum(2)
        return self.dropout(self.c_proj(y.flatten(2).to(z.dtype)))

    def forward(self, x):
        z, c, b, v, adt, dt, trap, angles = self._project(x)
        return self._finish(self._scan(c, b, v, adt, dt, trap, angles), z)

    def allocate_inference_cache(self, batch_size, max_seqlen=None, dtype=None):
        device = self.in_proj.weight.device
        # Reference decode keeps all recurrent state in float32 on every backend.
        return (torch.zeros(batch_size, self.nheads, self.num_angles, device=device),
                torch.zeros(batch_size, self.nheads, self.headdim, self.d_state, device=device),
                torch.zeros(batch_size, self.rank, self.nheads, self.d_state, device=device),
                torch.zeros(batch_size, self.nheads, self.headdim, device=device))

    def step(self, x, cache):
        if x.shape[1] != 1:
            raise ValueError("Mamba3.step expects exactly one token")
        z, c, b, v, adt, dt, trap, angles = self._project(x)
        y, final = mamba3_scan(c, b, v, adt, dt, trap, angles, self.C_bias, self.B_bias,
                               self.mimo_x, initial_state=cache, is_mimo=self.is_mimo)
        for state, value in zip(cache, final):
            state.copy_(value)
        y = y + self.D.float()[None, None, None, :, None] * v.float()[:, :, None] * self.mimo_x.float().permute(1, 0, 2)[None, None]
        return self._finish(y, z)


class Mamba3Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.norm = make_norm(config)
        self.mixer = Mamba3(config, layer_idx)

    def forward(self, x):
        return x + self.mixer(self.norm(x))


class Model(CachedModel):
    def __init__(self, config):
        Transformer.__init__(self, config, use_pos_embedding=False, positional_encoding="nope",
                             positional_markers=(), block_builder=Mamba3Block)
