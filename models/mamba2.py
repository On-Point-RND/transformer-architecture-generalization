import math
from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn, Tensor
from torch.nn import functional as F

from core.config import ModelConfig
from core.model import Transformer

try:
    from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
except (ImportError, OSError) as error:
    mamba_chunk_scan_combined = None
    _FUSED_IMPORT_ERROR = error
else:
    _FUSED_IMPORT_ERROR = None


@dataclass
class Config(ModelConfig):
    name: str = "mamba2"
    d_state: int = 128
    d_conv: int = 4
    expand: int = 2
    headdim: int = 64
    d_ssm: int = 0  # 0 applies the SSM to the whole inner width
    ngroups: int = 1  # number of shared B/C groups
    chunk_size: int = 256
    d_has_hdim: bool = False  # one D per channel instead of per head
    rmsnorm: bool = True
    norm_before_gate: bool = False
    learnable_init_states: bool = False
    conv_bias: bool = True
    conv_init: Optional[float] = None
    a_init_min: float = 1.0
    a_init_max: float = 16.0
    dt_min: float = 0.001
    dt_max: float = 0.1
    dt_init_floor: float = 1e-4
    dt_limit: tuple[float, float] = (0.0, float("inf"))
    ssm_kernel: str = "auto"  # auto | torch | fused (official CUDA Triton kernel)

    def __post_init__(self):
        for name in (
            "n_embd",
            "n_layer",
            "d_state",
            "d_conv",
            "expand",
            "headdim",
            "ngroups",
            "chunk_size",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"model.{name} must be a positive integer")
        d_inner = self.expand * self.n_embd
        d_ssm = self.d_ssm or d_inner
        if type(self.d_ssm) is not int or not 0 <= self.d_ssm <= d_inner:
            raise ValueError("model.d_ssm must be 0 or an inner width")
        if d_ssm % self.headdim:
            raise ValueError("d_ssm must be divisible by headdim")
        if (d_ssm // self.headdim) % self.ngroups:
            raise ValueError("the number of SSM heads must be divisible by ngroups")
        if not 0 < self.a_init_min <= self.a_init_max:
            raise ValueError("require 0 < a_init_min <= a_init_max")
        if not 0 < self.dt_min <= self.dt_max or self.dt_init_floor <= 0:
            raise ValueError("require 0 < dt_min <= dt_max and dt_init_floor > 0")
        if len(self.dt_limit) != 2 or not 0 <= self.dt_limit[0] <= self.dt_limit[1]:
            raise ValueError("dt_limit must be a nonnegative (minimum, maximum) pair")
        if self.ssm_kernel not in ("auto", "torch", "fused"):
            raise ValueError("model.ssm_kernel must be 'auto', 'torch', or 'fused'")


def segment_sum(x: Tensor):
    """Entry (i, j) is x[j+1] + ... + x[i], or -inf above the diagonal."""
    size = x.size(-1)
    mask = torch.ones(size, size, dtype=torch.bool, device=x.device)
    sums = x[..., :, None].expand(*x.shape, size)
    sums = sums.masked_fill(~mask.tril(-1), 0).cumsum(-2)
    return sums.masked_fill(~mask.tril(), -torch.inf)


def _repeat_groups(x: Tensor, nheads: int):
    return x.repeat_interleave(nheads // x.shape[2], dim=2)


def recurrent_scan(x, dt, a, b, c, initial_state=None, seq_idx=None):
    """Transparent SSD recurrence, also used when packed sequences need resets."""
    batch, length, nheads, headdim = x.shape
    b, c = _repeat_groups(b, nheads), _repeat_groups(c, nheads)
    state = (
        x.new_zeros(batch, nheads, headdim, b.shape[-1])
        if initial_state is None
        else initial_state
    )
    outputs = []
    for t in range(length):
        if seq_idx is not None and t:
            reset = seq_idx[:, t] != seq_idx[:, t - 1]
            fresh = x.new_zeros(state.shape) if initial_state is None else initial_state
            state = torch.where(reset[:, None, None, None], fresh, state)
        decay = torch.exp(dt[:, t] * a)
        update = dt[:, t, :, None, None] * x[:, t, :, :, None] * b[:, t, :, None]
        state = decay[:, :, None, None] * state + update
        outputs.append(torch.einsum("bhpn,bhn->bhp", state, c[:, t]))
    return torch.stack(outputs, 1), state


def ssd_scan(x, dt, a, b, c, chunk_size, initial_state=None, seq_idx=None):
    if x.shape[1] == 0:
        raise ValueError("Mamba2 requires a nonempty sequence")
    dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
    with torch.autocast(device_type=x.device.type, enabled=False):
        x, dt, a, b, c = (v.to(dtype) for v in (x, dt, a, b, c))
        if initial_state is not None:
            initial_state = initial_state.to(dtype)
        if seq_idx is not None:
            return recurrent_scan(x, dt, a, b, c, initial_state, seq_idx)

        batch, length, nheads, headdim = x.shape
        b, c = _repeat_groups(b, nheads), _repeat_groups(c, nheads)
        state = (
            x.new_zeros(batch, nheads, headdim, b.shape[-1])
            if initial_state is None
            else initial_state
        )
        outputs = []
        for start in range(0, length, chunk_size):
            stop = min(start + chunk_size, length)
            u = x[:, start:stop] * dt[:, start:stop, :, None]
            keys, queries = b[:, start:stop], c[:, start:stop]
            log_decay = (dt[:, start:stop] * a).transpose(1, 2)
            decay = segment_sum(log_decay).exp()

            weights = torch.einsum("bihn,bjhn->bhij", queries, keys) * decay
            local = torch.einsum("bhij,bjhp->bihp", weights, u)
            prefix = log_decay.cumsum(-1).exp()
            carried = torch.einsum("bihn,bhpn,bhi->bihp", queries, state, prefix)
            outputs.append(local + carried)

            state = state * prefix[..., -1, None, None] + torch.einsum(
                "bihn,bhi,bihp->bhpn", keys, decay[..., -1, :], u
            )
        return torch.cat(outputs, 1), state


class Mamba2(nn.Module):
    """Causal [batch, time, width] mixer compatible with attention sublayers."""

    def __init__(self, config, layer_idx=0):
        super().__init__()
        self.layer_idx = layer_idx
        self.d_inner = config.expand * config.n_embd
        self.d_ssm = config.d_ssm or self.d_inner
        self.d_mlp = self.d_inner - self.d_ssm
        self.headdim = config.headdim
        self.nheads = self.d_ssm // self.headdim
        self.ngroups = config.ngroups
        self.d_state = config.d_state
        self.d_conv = config.d_conv
        self.chunk_size = config.chunk_size
        self.d_has_hdim = config.d_has_hdim
        self.rmsnorm = config.rmsnorm
        self.norm_before_gate = config.norm_before_gate
        self.dt_limit = config.dt_limit
        self.ssm_kernel = config.ssm_kernel

        self.conv_dim = self.d_ssm + 2 * self.ngroups * self.d_state
        projection_dim = (
            2 * self.d_inner + 2 * self.ngroups * self.d_state + self.nheads
        )
        self.in_proj = nn.Linear(config.n_embd, projection_dim, bias=config.bias)
        self.conv1d = nn.Conv1d(
            self.conv_dim,
            self.conv_dim,
            self.d_conv,
            groups=self.conv_dim,
            padding=self.d_conv - 1,
            bias=config.conv_bias,
        )
        if config.conv_init is not None:
            nn.init.uniform_(self.conv1d.weight, -config.conv_init, config.conv_init)

        dt = (
            torch.empty(self.nheads)
            .uniform_(math.log(config.dt_min), math.log(config.dt_max))
            .exp()
        )
        dt = dt.clamp_min(config.dt_init_floor)
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))
        self.A_log = nn.Parameter(
            torch.empty(self.nheads)
            .uniform_(config.a_init_min, config.a_init_max)
            .log()
        )
        self.D = nn.Parameter(
            torch.ones(self.d_ssm if self.d_has_hdim else self.nheads)
        )
        if config.learnable_init_states:
            self.init_states = nn.Parameter(
                torch.zeros(self.nheads, self.headdim, self.d_state)
            )
        else:
            self.init_states = None
        if self.rmsnorm:
            group_width = self.d_ssm // self.ngroups
            self.norm = (
                nn.RMSNorm(group_width, eps=1e-5)
                if self.ngroups == 1
                else nn.ModuleList(
                    nn.RMSNorm(group_width, eps=1e-5)
                    for _ in range(self.ngroups)
                )
            )
        self.c_proj = nn.Linear(self.d_inner, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def _split_projection(self, projected):
        sizes = (self.d_mlp, self.d_mlp, self.d_ssm, self.conv_dim, self.nheads)
        return projected.split(sizes, dim=-1)

    def _time_steps(self, dt):
        dt = F.softplus(dt.float() + self.dt_bias.float())
        return dt.clamp(min=self.dt_limit[0], max=self.dt_limit[1])

    def _causal_conv(self, xbc, seq_idx=None):
        if seq_idx is None:
            return F.silu(
                self.conv1d(xbc.transpose(1, 2))[..., : xbc.shape[1]].transpose(1, 2)
            )
        output = torch.zeros_like(xbc)
        weights = self.conv1d.weight[:, 0]
        for lag in range(self.d_conv):
            if lag >= xbc.shape[1]:
                break
            same_sequence = seq_idx[:, lag:] == seq_idx[:, : xbc.shape[1] - lag]
            output[:, lag:] += (
                xbc[:, : xbc.shape[1] - lag]
                * weights[:, self.d_conv - 1 - lag]
                * same_sequence[..., None]
            )
        if self.conv1d.bias is not None:
            output = output + self.conv1d.bias
        return F.silu(output)

    def _initial_state(self, batch, dtype, device):
        if self.init_states is None:
            return None
        return self.init_states.to(device=device, dtype=dtype).expand(batch, -1, -1, -1)

    def _use_fused_scan(self, x):
        if self.ssm_kernel == "torch":
            return False
        if x.device.type != "cuda":
            if self.ssm_kernel == "fused":
                raise RuntimeError("model.ssm_kernel='fused' requires a CUDA device")
            return False
        if mamba_chunk_scan_combined is None:
            if self.ssm_kernel == "fused":
                detail = f": {_FUSED_IMPORT_ERROR}" if _FUSED_IMPORT_ERROR else ""
                raise RuntimeError(
                    "model.ssm_kernel='fused' requires the mamba-ssm package" + detail
                )
            return False
        return True

    def _scan(self, x, dt, b, c, initial_state, seq_idx):
        a = -self.A_log.float().exp()
        if self._use_fused_scan(x):
            return mamba_chunk_scan_combined(
                x,
                dt,
                a,
                b,
                c,
                chunk_size=self.chunk_size,
                dt_bias=self.dt_bias,
                dt_softplus=True,
                dt_limit=self.dt_limit,
                initial_states=initial_state,
                seq_idx=seq_idx,
            )
        return ssd_scan(
            x,
            self._time_steps(dt),
            a,
            b,
            c,
            self.chunk_size,
            initial_state,
            seq_idx,
        )[0]

    def _apply_rmsnorm(self, x):
        if self.ngroups == 1:
            return self.norm(x)
        groups = x.reshape(*x.shape[:-1], self.ngroups, -1)
        return torch.stack(
            [norm(groups[..., i, :]) for i, norm in enumerate(self.norm)],
            dim=-2,
        ).flatten(-2)

    def _finish(self, z0, x0, z, x, y):
        skip = (
            self.D.view(self.nheads, self.headdim)
            if self.d_has_hdim
            else self.D[:, None]
        )
        y = y + x.float() * skip.float()[None, None]
        y = y.flatten(2).to(z.dtype)
        if self.rmsnorm:
            if self.norm_before_gate:
                y = self._apply_rmsnorm(y) * F.silu(z)
            else:
                y = self._apply_rmsnorm(y * F.silu(z))
        else:
            y = y * F.silu(z)
        if self.d_mlp:
            y = torch.cat((F.silu(z0) * x0, y), dim=-1)
        return self.dropout(self.c_proj(y))

    def forward(self, x, seq_idx=None):
        batch, length, _ = x.shape
        z0, x0, z, xbc, dt = self._split_projection(self.in_proj(x))
        xbc = self._causal_conv(xbc, seq_idx)
        x, b, c = xbc.split(
            (self.d_ssm, self.ngroups * self.d_state, self.ngroups * self.d_state),
            dim=-1,
        )
        x = x.reshape(batch, length, self.nheads, self.headdim)
        b = b.reshape(batch, length, self.ngroups, self.d_state)
        c = c.reshape(batch, length, self.ngroups, self.d_state)
        y = self._scan(
            x, dt, b, c, self._initial_state(batch, x.dtype, x.device), seq_idx
        )
        return self._finish(z0, x0, z, x, y)

    def allocate_inference_cache(self, batch_size, max_seqlen=None, dtype=None):
        """Return `(convolution_state, SSM_state)` for cached decoding."""
        device = self.c_proj.weight.device
        conv_dtype = dtype or self.conv1d.weight.dtype
        state_dtype = dtype or self.in_proj.weight.dtype
        conv = torch.zeros(
            batch_size, self.conv_dim, self.d_conv, device=device, dtype=conv_dtype
        )
        initial = self._initial_state(batch_size, state_dtype, device)
        ssm = (
            torch.zeros(
                batch_size,
                self.nheads,
                self.headdim,
                self.d_state,
                device=device,
                dtype=state_dtype,
            )
            if initial is None
            else initial.clone()
        )
        return conv, ssm

    def step(self, x, conv_state, ssm_state):
        """Decode one token and update both cache tensors in place."""
        if x.shape[1] != 1:
            raise ValueError("Mamba2.step expects exactly one token")
        z0, x0, z, xbc, dt = self._split_projection(self.in_proj(x[:, 0]))
        conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))
        conv_state[:, :, -1] = xbc
        xbc = (conv_state * self.conv1d.weight[:, 0]).sum(-1)
        if self.conv1d.bias is not None:
            xbc = xbc + self.conv1d.bias
        xbc = F.silu(xbc)
        x, b, c = xbc.split(
            (self.d_ssm, self.ngroups * self.d_state, self.ngroups * self.d_state),
            dim=-1,
        )
        x = x.reshape(x.shape[0], self.nheads, self.headdim)
        b = b.reshape(b.shape[0], self.ngroups, self.d_state)
        c = c.reshape(c.shape[0], self.ngroups, self.d_state)
        b = b.repeat_interleave(self.nheads // self.ngroups, dim=1)
        c = c.repeat_interleave(self.nheads // self.ngroups, dim=1)
        dt = self._time_steps(dt)
        decay = torch.exp(dt * -self.A_log.float().exp())
        update = dt[:, :, None, None] * x.float()[..., None] * b.float()[:, :, None]
        ssm_state.copy_(decay[:, :, None, None] * ssm_state + update)
        y = torch.einsum("bhpn,bhn->bhp", ssm_state.float(), c.float())[:, None]
        return self._finish(z0[:, None], x0[:, None], z[:, None], x[:, None], y)


class MambaBlock(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.norm = nn.RMSNorm(config.n_embd, eps=1e-5)
        self.mixer = Mamba2(config, layer_idx)

    def forward(self, x):
        return x + self.mixer(self.norm(x))


class Model(Transformer):
    def __init__(self, config):
        super().__init__(config)
        self.transformer.ln_f = nn.RMSNorm(config.n_embd, eps=1e-5)

    def build_blocks(self, config):
        return nn.ModuleList([MambaBlock(config, i) for i in range(config.n_layer)])

    def uses_pos_embedding(self, config):
        return False

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        return -1.0  # the shared estimate assumes attention

    def _step_token(self, token, caches):
        x = self.transformer.drop(self.transformer.wte(token))
        for block, (conv_state, ssm_state) in zip(self.transformer.h, caches):
            x = x + block.mixer.step(block.norm(x), conv_state, ssm_state)
        return self.lm_head(self.transformer.ln_f(x))

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """Complete tokens using each mixer layer's recurrent inference cache."""
        if idx.size(1) == 0:
            raise ValueError("Mamba2 generation requires a nonempty prompt")
        caches = [block.mixer.allocate_inference_cache(idx.size(0))
                  for block in self.transformer.h]
        logits = None
        for position in range(idx.size(1)):
            logits = self._step_token(idx[:, position : position + 1], caches)
        for step in range(max_new_tokens):
            scores = logits[:, -1, :] / temperature
            if top_k is not None:
                values, _ = torch.topk(scores, min(top_k, scores.size(-1)))
                scores[scores < values[:, [-1]]] = -float("inf")
            next_token = torch.multinomial(F.softmax(scores, dim=-1), 1)
            idx = torch.cat((idx, next_token), dim=1)
            if step + 1 < max_new_tokens:
                logits = self._step_token(next_token, caches)
        return idx
