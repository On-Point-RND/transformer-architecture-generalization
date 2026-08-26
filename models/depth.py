from dataclasses import dataclass

from core.config import ModelConfig
from core.model import Transformer


@dataclass
class Config(ModelConfig):
    name: str = "depth"
    n_head: int = 0  # 0 = derive from n_embd, so a width sweep stays one axis


def default_n_heads(n_embd):
    """The divisor of n_embd whose head dimension is closest to 32."""
    return min((h for h in range(1, n_embd + 1) if n_embd % h == 0),
               key=lambda h: abs(n_embd // h - 32))


class Model(Transformer):
    residual_init_scaling = False

    def __init__(self, config):
        if not config.n_head:
            # resolved on the config itself, so the checkpoint records the
            # head count the run actually used
            config.n_head = default_n_heads(config.n_embd)
        super().__init__(config)
