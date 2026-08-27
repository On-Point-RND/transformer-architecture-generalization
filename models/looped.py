from dataclasses import dataclass

import torch.nn as nn

from core.config import ModelConfig
from core.model import Block, Transformer


@dataclass
class Config(ModelConfig):
    name: str = "looped"
    n_layer: int = 1  
    n_loops: int = 4


class Model(Transformer):
    residual_init_scaling = False

    def build_blocks(self, config):
        block = Block(config, self.build_attention(config, 0), self.build_mlp(config, 0))
        return nn.ModuleList([block] * config.n_loops)
