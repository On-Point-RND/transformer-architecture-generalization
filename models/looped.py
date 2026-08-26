from dataclasses import dataclass

import torch.nn as nn

from core.config import ModelConfig
from core.model import Block, Transformer


@dataclass
class Config(ModelConfig):
    name: str = "looped"
    n_layer: int = 1  # one distinct block exists; n_loops is what varies
    n_loops: int = 4


class Model(Transformer):
    residual_init_scaling = False

    def build_blocks(self, config):
        # One Block object listed n_loops times, so the weights are shared:
        # named_children dedupes, so _init_weights runs once, and
        # named_parameters dedupes, so the count stays that of a single block.
        block = Block(config, self.build_attention(config, 0), self.build_mlp(config, 0))
        return nn.ModuleList([block] * config.n_loops)
