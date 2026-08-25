from dataclasses import dataclass

from core.config import ModelConfig
from core.model import CausalAttention, Transformer


@dataclass
class Config(ModelConfig):
    """Architecture-specific fields go here; YAML validation follows the class."""
    name: str = "vanilla"


class Attention(CausalAttention):
    # False keeps the fused SDPA kernel, which never materialises the T x T
    # score matrix. Set True when a hook below needs those scores.
    needs_scores = False

    def transform_qk(self, q, k):
        """Rotary-style position transforms; q, k are [B, n_head, T, head_dim]."""
        return q, k

    def add_bias(self, scores, q, k):
        """Additive attention bias — ALiBi, relative bias, CoPE, CAPE.

        Only called when needs_scores is True. Returns the scores so that an
        architecture composing several terms controls the order of additions.
        """
        return scores


class Model(Transformer):
    def build_attention(self, config, layer_idx):
        """The attention sublayer: any nn.Module mapping [B, T, C] -> [B, T, C]."""
        return Attention(config, layer_idx)

    def uses_pos_embedding(self, config):
        """Is there a learned wpe table? False for NoPE- or RoPE-style models."""
        return True

    def param_report(self):
        """Positional parameter counts and mechanism name for the run log."""
        return super().param_report()
