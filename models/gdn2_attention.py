"""Stacks of GDN2 and causal attention blocks, ordered by a repeated pattern."""

from dataclasses import dataclass
import math

from torch import nn

from core.model import Block, CausalAttention, MLP, Transformer
from models.gated_deltanet2 import Config as GDN2Config
from models.gated_deltanet2 import GatedDelta2Block
from models.positional import rope_cache, rotate_half


@dataclass
class Config(GDN2Config):
    name: str = "gdn2_attention"
    layer_pattern: tuple[str, ...] = ("gdn2", "attention")
    use_pos_embedding: bool = True  # learned absolute embeddings at stack input
    attention_pos_encoding: str = "nope"  # nope | rope; applies only to attention
    rope_theta: float = 10000.0

    def __post_init__(self):
        super().__post_init__()
        if (not isinstance(self.layer_pattern, (tuple, list))
                or not self.layer_pattern
                or any(kind not in ("gdn2", "attention") for kind in self.layer_pattern)):
            raise ValueError("model.layer_pattern must be a nonempty list of gdn2/attention")
        if type(self.use_pos_embedding) is not bool:
            raise ValueError("model.use_pos_embedding must be a boolean")
        if self.attention_pos_encoding not in ("nope", "rope"):
            raise ValueError("model.attention_pos_encoding must be nope/rope")
        if not math.isfinite(self.rope_theta) or self.rope_theta <= 0:
            raise ValueError("model.rope_theta must be finite and positive")
        if self.attention_pos_encoding == "rope":
            if self.use_pos_embedding:
                raise ValueError("disable model.use_pos_embedding when using attention RoPE")
            if "attention" in self.layer_pattern and (self.n_embd // self.n_head) % 2:
                raise ValueError("attention RoPE requires an even head dimension")


class RotaryAttention(CausalAttention):
    """Standard causal attention with RoPE on Q/K, reusing the PE study helpers."""

    def __init__(self, config, layer_idx=0):
        super().__init__(config, layer_idx)
        self.rope_theta = config.rope_theta

    def transform_qk(self, q, k):
        cos, sin = rope_cache(q.size(-2), self.head_dim, self.rope_theta, q.device, q.dtype)
        return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin


def build_block(config, layer_idx):
    kind = config.layer_pattern[layer_idx % len(config.layer_pattern)]
    if kind == "gdn2":
        return GatedDelta2Block(config, layer_idx)
    attention = RotaryAttention if config.attention_pos_encoding == "rope" else CausalAttention
    return Block(config, attention(config, layer_idx), MLP(config, layer_idx))


class Model(Transformer):
    def __init__(self, config):
        has_attention = any(config.layer_pattern[i % len(config.layer_pattern)] == "attention"
                            for i in range(config.n_layer))
        positional_encoding = "wpe" if config.use_pos_embedding else (
            "rope" if has_attention and config.attention_pos_encoding == "rope" else "nope"
        )
        super().__init__(
            config,
            use_pos_embedding=config.use_pos_embedding,
            positional_encoding=positional_encoding,
            positional_markers=("transformer.wpe",) if config.use_pos_embedding else (),
            block_builder=build_block,
        )
        # Keep the upstream GDN2 mixer initialization used by the pure model.
        # Attention and MLP retain the shared Transformer's initialization.
        for block in self.transformer.h:
            if isinstance(block, GatedDelta2Block):
                for module in block.mixer.modules():
                    if isinstance(module, nn.Linear):
                        nn.init.xavier_uniform_(module.weight, gain=2 ** -2.5)
                        if module.bias is not None:
                            nn.init.zeros_(module.bias)

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        # The shared estimate assumes every block is a Transformer block.
        return -1.0
