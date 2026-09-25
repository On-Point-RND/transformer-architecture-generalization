import torch
from torch import nn
from torch.nn import functional as F


class RMSNorm(nn.Module):
    def __init__(self, width: int, eps: float = 1e-5, groups: int = 1):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps
        self.groups = groups

    def forward(self, x: torch.Tensor):
        dtype = x.dtype
        x = x.float() if dtype != torch.float64 else x
        shape = (*x.shape[:-1], self.groups, x.shape[-1] // self.groups)
        grouped = x.reshape(shape)
        grouped = grouped * torch.rsqrt(
            grouped.square().mean(-1, keepdim=True) + self.eps
        )
        return (grouped.flatten(-2) * self.weight).to(dtype)


class GatedRMSNorm(RMSNorm):
    def __init__(self, width: int, groups: int = 1, norm_before_gate: bool = False):
        super().__init__(width, groups=groups)
        self.norm_before_gate = norm_before_gate

    def forward(self, x, gate):
        if self.norm_before_gate:
            return super().forward(x) * F.silu(gate)
        return super().forward(x * F.silu(gate))
