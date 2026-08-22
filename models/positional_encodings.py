"""Composable positional mechanisms for ``models/positional.py``.

The four orthogonal slots are deliberately explicit:

* hidden-state embedding: ``wpe``;
* Q/K transform: ``rope`` or ``fope``;
* additive attention bias: ``alibi`` or ``relative_bias``;
* context-dependent score: ``cope`` or ``cape``.

All mechanisms are consumed by the single composable positional model.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


_ALIASES = {
    "none": "nope", "nope": "nope", "wpe": "wpe", "rope": "rope",
    "alibi": "alibi", "rpe": "relative_bias", "relative": "relative_bias",
    "relative_bias": "relative_bias", "t5_rpe": "relative_bias",
    "cope": "cope", "cape": "cape", "fope": "fope",
}
_SLOTS = {
    "nope": None, "wpe": "embedding", "rope": "qk", "fope": "qk",
    "alibi": "bias", "relative_bias": "bias", "cope": "context",
    "cape": "context",
}


@dataclass(frozen=True)
class PositionalSpec:
    embedding: str = "nope"
    qk: str = "none"
    bias: str = "none"
    context: str = "none"

    @property
    def canonical(self):
        parts = []
        if self.embedding != "nope":
            parts.append(self.embedding)
        if self.qk != "none":
            parts.append(self.qk)
        if self.bias != "none":
            parts.append(self.bias)
        if self.context != "none":
            parts.append(self.context)
        return "+".join(parts) or "nope"


def parse_positional_spec(value):
    """Parse e.g. ``rope+relative_bias`` and reject ambiguous same-slot mixes."""
    raw = [p.strip().lower() for p in value.replace(",", "+").split("+") if p.strip()]
    raw = raw or ["nope"]
    selected = {"embedding": "nope", "qk": "none", "bias": "none", "context": "none"}
    for name in raw:
        if name not in _ALIASES:
            raise ValueError(f"unknown positional mechanism {name!r}; known: {sorted(_ALIASES)}")
        name = _ALIASES[name]
        slot = _SLOTS[name]
        if slot is None:
            continue
        if selected[slot] not in ("none", "nope") and selected[slot] != name:
            raise ValueError(f"cannot combine {selected[slot]!r} and {name!r}: both occupy {slot!r}")
        selected[slot] = name
    return PositionalSpec(**selected)


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def rope_cache(seq_len, head_dim, theta, device, dtype):
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    freqs = torch.outer(torch.arange(seq_len, device=device).float(), inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def _alibi_slopes(n_head):
    """Slopes from the reference ALiBi implementation, including non-powers of 2."""
    def power_of_two(n):
        start = 2 ** (-(2 ** -(math.log2(n) - 3)))
        return [start * start ** i for i in range(n)]
    if math.log2(n_head).is_integer():
        return power_of_two(n_head)
    p = 2 ** math.floor(math.log2(n_head))
    return power_of_two(p) + _alibi_slopes(2 * p)[0::2][:n_head - p]


def t5_relative_position_bucket(relative_position, num_buckets=32, max_distance=128):
    """T5's unidirectional log buckets for non-negative backward distances."""
    distance = relative_position.clamp_min(0)
    max_exact = num_buckets // 2
    is_small = distance < max_exact
    large = max_exact + (
        torch.log(distance.float().clamp_min(1) / max_exact)
        / math.log(max_distance / max_exact)
        * (num_buckets - max_exact)
    ).long()
    large = large.clamp(max=num_buckets - 1)
    return torch.where(is_small, distance, large)


class RelativeBias(nn.Module):
    def __init__(self, n_head, num_buckets, max_distance):
        super().__init__()
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.embedding = nn.Embedding(num_buckets, n_head)

    def forward(self, length, device, dtype):
        pos = torch.arange(length, device=device)
        distance = pos[:, None] - pos[None, :]
        buckets = t5_relative_position_bucket(distance, self.num_buckets, self.max_distance)
        return self.embedding(buckets).permute(2, 0, 1).unsqueeze(0).to(dtype)


class CoPE(nn.Module):
    """Contextual Position Encoding, equations (3)--(9) of Golovneva et al."""
    def __init__(self, head_dim, max_position):
        super().__init__()
        self.max_position = max_position
        self.position_embeddings = nn.Embedding(max_position + 1, head_dim)

    def forward(self, q, semantic_scores):
        # Gates above the causal diagonal cannot contribute to p_ij.
        t = semantic_scores.size(-1)
        causal = torch.ones(t, t, device=q.device, dtype=torch.bool).tril()
        gates = torch.sigmoid(semantic_scores) * causal
        positions = gates.flip(-1).cumsum(-1).flip(-1).clamp(max=self.max_position)
        z = torch.einsum("bhtd,pd->bhtp", q, self.position_embeddings.weight)
        lo = positions.floor().long()
        hi = positions.ceil().long()
        alpha = positions - lo
        return torch.gather(z, -1, lo) * (1 - alpha) + torch.gather(z, -1, hi) * alpha


class CAPE(nn.Module):
    """Context-Adaptive PE: per-head two-layer LeakyReLU f(attention, base_bias)."""
    def __init__(self, n_head, hidden_dim):
        super().__init__()
        self.mlps = nn.ModuleList([
            nn.Sequential(nn.Linear(2, hidden_dim), nn.LeakyReLU(), nn.Linear(hidden_dim, 1))
            for _ in range(n_head)
        ])

    def forward(self, semantic_scores, base_bias):
        if base_bias.size(0) == 1 and semantic_scores.size(0) != 1:
            base_bias = base_bias.expand(semantic_scores.size(0), -1, -1, -1)
        outputs = []
        for head, mlp in enumerate(self.mlps):
            features = torch.stack((semantic_scores[:, head], base_bias[:, head]), dim=-1)
            outputs.append(mlp(features).squeeze(-1))
        return torch.stack(outputs, dim=1)


class FoPE(nn.Module):
    """Frozen Fourier-series rotary maps following Hua et al., Appendix B."""
    def __init__(self, n_head, head_dim, theta, train_length, init_gain):
        super().__init__()
        all_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        inv_freq = all_freq[all_freq >= 2 * math.pi / train_length]
        input_dim = len(inv_freq)
        output_dim = min(input_dim, head_dim // 4)
        if output_dim == 0:
            raise ValueError("FoPE has no trained frequency; increase fope_train_length or head_dim")
        sin_coef = torch.empty(n_head, input_dim, output_dim)
        cos_coef = torch.empty_like(sin_coef)
        nn.init.xavier_normal_(sin_coef, gain=init_gain)
        nn.init.xavier_normal_(cos_coef, gain=init_gain)
        eye = torch.eye(input_dim, output_dim).unsqueeze(0)
        self.register_buffer("inv_freq", inv_freq, persistent=True)
        self.register_buffer("sin_coef", sin_coef + eye, persistent=True)
        self.register_buffer("cos_coef", cos_coef + eye, persistent=True)
        self.head_dim = head_dim

    @staticmethod
    def _normalise(coef):
        denom = coef.sum(dim=-2, keepdim=True)
        return coef / denom.where(denom.abs() > 1e-6, torch.full_like(denom, 1e-6))

    def apply(self, x):
        t = x.size(-2)
        freqs = torch.outer(torch.arange(t, device=x.device).float(), self.inv_freq.to(x.device))
        pos_sin = freqs.sin().to(x.dtype).view(1, 1, t, -1).expand(x.size(0), x.size(1), -1, -1)
        pos_cos = freqs.cos().to(x.dtype).view(1, 1, t, -1).expand_as(pos_sin)
        fourier_sin = torch.einsum("bhtD,hDd->bhtd", pos_sin, self._normalise(self.sin_coef).to(x.dtype))
        fourier_cos = torch.einsum("bhtD,hDd->bhtd", pos_cos, self._normalise(self.cos_coef).to(x.dtype))
        pad = self.head_dim // 2 - fourier_sin.size(-1)
        fourier_sin = F.pad(fourier_sin, (0, pad), value=1.0)
        fourier_cos = F.pad(fourier_cos, (0, pad), value=1.0)
        fourier_sin = torch.cat((fourier_sin, fourier_sin), dim=-1)
        fourier_cos = torch.cat((fourier_cos, fourier_cos), dim=-1)
        return x * fourier_cos - rotate_half(x) * fourier_sin


class PositionalAttention(nn.Module):
    """Owns only PE-specific state and composes the four positional slots."""
    def __init__(self, config):
        super().__init__()
        self.spec = parse_positional_spec(config.pos_encoding)
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.rope_theta = config.rope_theta
        if self.spec.bias == "alibi" or (self.spec.context == "cape" and self.spec.bias == "none"):
            self.register_buffer("alibi_slopes", torch.tensor(_alibi_slopes(config.n_head)), persistent=True)
        if self.spec.bias == "relative_bias":
            self.relative_bias = RelativeBias(config.n_head, config.rpe_num_buckets, config.rpe_max_distance)
        if self.spec.context == "cope":
            self.cope = CoPE(self.head_dim, config.cope_max_position)
        if self.spec.context == "cape":
            self.cape = CAPE(config.n_head, config.cape_hidden_dim)
        if self.spec.qk == "fope":
            self.fope = FoPE(config.n_head, self.head_dim, config.rope_theta,
                             config.fope_train_length, config.fope_init_gain)

    @property
    def is_legacy_fast_path(self):
        return self.spec.bias == "none" and self.spec.context == "none"

    def transform_qk(self, q, k):
        if self.spec.qk == "rope":
            cos, sin = rope_cache(q.size(-2), self.head_dim, self.rope_theta, q.device, q.dtype)
            return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin
        if self.spec.qk == "fope":
            return self.fope.apply(q), self.fope.apply(k)
        return q, k

    def _alibi(self, t, device, dtype):
        pos = torch.arange(t, device=device)
        distance = (pos[:, None] - pos[None, :]).clamp_min(0).to(dtype)
        return -self.alibi_slopes.to(device=device, dtype=dtype).view(1, -1, 1, 1) * distance

    def static_bias(self, t, device, dtype):
        if self.spec.bias == "alibi":
            return self._alibi(t, device, dtype)
        if self.spec.bias == "relative_bias":
            return self.relative_bias(t, device, dtype)
        return None

    def contextual_bias(self, q, semantic_scores, static_bias):
        if self.spec.context == "cope":
            return self.cope(q, semantic_scores)
        if self.spec.context == "cape":
            base = static_bias if static_bias is not None else self._alibi(
                semantic_scores.size(-1), semantic_scores.device, semantic_scores.dtype)
            return self.cape(semantic_scores, base)
        return None
