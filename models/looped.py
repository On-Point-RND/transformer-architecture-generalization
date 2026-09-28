from dataclasses import dataclass

from core.config import ModelConfig
from core.model import Transformer


@dataclass
class Config(ModelConfig):
    name: str = "looped"
    n_layer: int = 1  
    n_loops: int = 4


def build_model(config):
    return Transformer(
        config,
        n_block_applications=config.n_loops,
        share_block_weights=True,
        residual_init_scaling=False,
    )
