"""Explicit model registry: config schema plus one concrete builder per model."""

from core.config import ModelConfig
from core.model import Transformer
from models.attention_design import Config as AttentionConfig
from models.attention_design import build_model as build_attention
from models.ffn_gating import Config as FFNConfig
from models.ffn_gating import build_model as build_ffn
from models.looped import Config as LoopedConfig
from models.looped import build_model as build_looped
from models.lssa import LSSAConfig, LSSARConfig, build_lssa, build_lssar
from models.positional import Config as PositionalConfig
from models.positional import build_model as build_positional
from models.relumax import Config as RelumaxConfig
from models.relumax import build_model as build_relumax
from models.taylor import Config as TaylorConfig
from models.taylor import build_model as build_taylor


def build_vanilla(config):
    return Transformer(config)


MODELS = {
    "vanilla": (ModelConfig, build_vanilla),
    "positional": (PositionalConfig, build_positional),
    "attention_design": (AttentionConfig, build_attention),
    "ffn_gating": (FFNConfig, build_ffn),
    "looped": (LoopedConfig, build_looped),
    "taylor": (TaylorConfig, build_taylor),
    "relumax": (RelumaxConfig, build_relumax),
    "lssa": (LSSAConfig, build_lssa),
    "lssar": (LSSARConfig, build_lssar),
}


def get_model(name):
    try:
        return MODELS[name]
    except KeyError:
        raise ValueError(
            f"unknown model {name!r}; choose one of: {', '.join(MODELS)}"
        ) from None
