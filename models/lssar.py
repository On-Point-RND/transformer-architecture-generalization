"""Length Scaled Softplus Attention with Re-weighting (LSSAR) from arXiv:2501.13428."""

from dataclasses import dataclass

import torch

from core.config import ModelConfig
from models.lssa import Attention as LssaAttention, lssa_attention
from models.lssa import Model as LssaModel


def lssar_attention(q, k, v, p=15, eps=1e-8):
    """LSSA followed by Shift-ReLU^p sharpening.

    Args:
        q, k, v: [B, H, L, D]
        p: sharpening power (paper default 15)
        eps: stabilizer for L1 normalisation

    Returns:
        output: [B, H, L, D]
        attention: [B, H, L, L] sharpened causal L1-normalised weights
    """
    _, attention = lssa_attention(q, k, v, eps=eps)
    length = q.size(-2)
    n = torch.arange(1, length + 1, device=q.device, dtype=attention.dtype).view(1, 1, length, 1)
    offset = torch.ones(1, 1, length, 1, device=q.device, dtype=attention.dtype)
    offset[:, :, :3, :] = 0.0
    attention = torch.relu(attention * n - offset).pow(p)
    attention = attention / (attention.sum(dim=-1, keepdim=True) + eps)
    return attention @ v, attention


@dataclass
class Config(ModelConfig):
    name: str = "lssar"
    rope_theta: float = 10000.0
    p: int = 15


class Attention(LssaAttention):
    def __init__(self, config, layer_idx=0):
        super().__init__(config, layer_idx)
        self.p = config.p

    def _mix_values(self, q, k, v, t, device):
        _, attention = lssar_attention(q, k, v, p=self.p)
        return self.attn_dropout(attention) @ v


class Model(LssaModel):
    def build_attention(self, config, layer_idx):
        return Attention(config, layer_idx)
