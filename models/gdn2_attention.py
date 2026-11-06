"""Stacks of GDN2 and causal attention blocks, ordered by a repeated pattern."""

from dataclasses import dataclass

from torch import nn

from core.model import Block, CausalAttention, MLP, Transformer
from models.gated_deltanet2 import Config as GDN2Config
from models.gated_deltanet2 import GatedDelta2Block


@dataclass
class Config(GDN2Config):
    name: str = "gdn2_attention"
    layer_pattern: tuple[str, ...] = ("gdn2", "attention")
    use_pos_embedding: bool = True

    def __post_init__(self):
        super().__post_init__()
        if (not isinstance(self.layer_pattern, (tuple, list))
                or not self.layer_pattern
                or any(kind not in ("gdn2", "attention") for kind in self.layer_pattern)):
            raise ValueError("model.layer_pattern must be a nonempty list of gdn2/attention")
        if type(self.use_pos_embedding) is not bool:
            raise ValueError("model.use_pos_embedding must be a boolean")


def build_block(config, layer_idx):
    kind = config.layer_pattern[layer_idx % len(config.layer_pattern)]
    if kind == "gdn2":
        return GatedDelta2Block(config, layer_idx)
    return Block(config, CausalAttention(config, layer_idx), MLP(config, layer_idx))


class Model(Transformer):
    def __init__(self, config):
        super().__init__(
            config,
            use_pos_embedding=config.use_pos_embedding,
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
