import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

from core.config import ModelConfig
from core.model import CausalAttention, Transformer


@dataclass
class Config(ModelConfig):
    name: str = "positional"
    pos_encoding: str = "nope"  
    rope_theta: float = 10000.0
    rpe_num_buckets: int = 32
    rpe_max_distance: int = 128
    cope_max_position: int = 64
    cape_hidden_dim: int = 32
    fope_train_length: int = 128
    fope_init_gain: float = 0.3

    def __post_init__(self):
        parse_positional_spec(self.pos_encoding)  # reject a bad spec at config load


SLOTS = {
    "nope": None,
    "wpe": "embedding", "sinusoidal": "embedding",
    "abs_shift": "embedding", "pose": "embedding",
    "rope": "qk", "fope": "qk",
    "alibi": "bias", "relative_bias": "bias",
    "cope": "context", "cape": "context",
}


@dataclass(frozen=True)
class PositionalSpec:
    embedding: str | None = None
    qk: str | None = None
    bias: str | None = None
    context: str | None = None

    @property
    def canonical(self):
        filled = (self.embedding, self.qk, self.bias, self.context)
        return "+".join(name for name in filled if name) or "nope"


def parse_positional_spec(value):
    raw = [p.strip().lower() for p in value.replace(",", "+").split("+") if p.strip()]
    selected = {slot: None for slot in ("embedding", "qk", "bias", "context")}
    for name in raw or ["nope"]:
        if name not in SLOTS:
            raise ValueError(f"unknown positional mechanism {name!r}; known: {sorted(SLOTS)}")
        slot = SLOTS[name]
        if slot is None:  
            continue
        if selected[slot] not in (None, name):
            raise ValueError(f"cannot combine {selected[slot]!r} and {name!r}: "
                             f"both occupy {slot!r}")
        selected[slot] = name
    return PositionalSpec(**selected)


def sinusoidal_table(block_size, n_embd):
    position = torch.arange(block_size).float().unsqueeze(1)
    angles = position * torch.exp(
        torch.arange(0, n_embd, 2).float() * (-math.log(10000.0) / n_embd))
    table = torch.zeros(block_size, n_embd)
    table[:, 0::2] = angles.sin()
    table[:, 1::2] = angles.cos()[:, : n_embd // 2]
    return table


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def rope_cache(seq_len, head_dim, theta, device, dtype):
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    freqs = torch.outer(torch.arange(seq_len, device=device).float(), inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def _alibi_slopes(n_head):
    def power_of_two(n):
        start = 2 ** (-(2 ** -(math.log2(n) - 3)))
        return [start * start ** i for i in range(n)]
    if math.log2(n_head).is_integer():
        return power_of_two(n_head)
    p = 2 ** math.floor(math.log2(n_head))
    return power_of_two(p) + _alibi_slopes(2 * p)[0::2][:n_head - p]


def t5_relative_position_bucket(relative_position, num_buckets=32, max_distance=128):
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
    def __init__(self, head_dim, max_position):
        super().__init__()
        self.max_position = max_position
        self.position_embeddings = nn.Embedding(max_position + 1, head_dim)

    def forward(self, q, semantic_scores):
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

    def rotate(self, x):    
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
    def __init__(self, config):
        super().__init__()
        self.spec = parse_positional_spec(config.pos_encoding)
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.rope_theta = config.rope_theta
        if self.spec.bias == "alibi" or (self.spec.context == "cape" and self.spec.bias is None):
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
        return self.spec.bias is None and self.spec.context is None

    def transform_qk(self, q, k):
        if self.spec.qk == "rope":
            cos, sin = rope_cache(q.size(-2), self.head_dim, self.rope_theta, q.device, q.dtype)
            return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin
        if self.spec.qk == "fope":
            return self.fope.rotate(q), self.fope.rotate(k)
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


class PositionalSelfAttention(CausalAttention):
    def __init__(self, config, layer_idx=0):
        super().__init__(config, layer_idx)
        if self.head_dim % 2 and parse_positional_spec(config.pos_encoding).qk in ("rope", "fope"):
            raise ValueError(f"rotary PE needs an even head_dim, got {self.head_dim}")
        self.positioning = PositionalAttention(config)
        self.needs_scores = not self.positioning.is_legacy_fast_path

    def transform_qk(self, q, k):
        return self.positioning.transform_qk(q, k)

    def add_bias(self, scores, q, k):
        static = self.positioning.static_bias(scores.size(-1), scores.device, scores.dtype)
        context = self.positioning.contextual_bias(q, scores, static)
        if static is not None and self.positioning.spec.context != "cape":
            scores = scores + static
        if context is not None:
            scores = scores + context
        return scores


def _is_positional(name):
    return ".positioning." in name or name.startswith("transformer.wpe")


class Model(Transformer):
    def __init__(self, config):
        super().__init__(config)
        self.positional_spec = parse_positional_spec(config.pos_encoding)
        if self.positional_spec.embedding == "sinusoidal":
            with torch.no_grad():
                self.transformer.wpe.weight.copy_(
                    sinusoidal_table(config.block_size, config.n_embd))
            self.transformer.wpe.weight.requires_grad_(False)

    def build_attention(self, config, layer_idx):
        return PositionalSelfAttention(config, layer_idx)

    def uses_pos_embedding(self, config):
        return parse_positional_spec(config.pos_encoding).embedding in (
            "wpe", "sinusoidal", "abs_shift", "pose"
        )

    def _check_sample_shape(self, batch_size, length):
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError(f"batch_size must be a positive integer, got {batch_size!r}")
        if not isinstance(length, int) or not 1 <= length <= self.config.block_size:
            raise ValueError(
                f"length must be in [1, {self.config.block_size}], got {length!r}"
            )

    def _sample_pose_for_lengths(self, lengths, width, device):
        """PoSE indices for variable valid lengths, padded to ``width`` with 0."""
        base = torch.arange(width, device=device).unsqueeze(0).expand(len(lengths), width)
        lengths = lengths.to(device=device, dtype=torch.long).view(-1, 1)
        budget = self.config.block_size - lengths
        active = (lengths >= 2) & (budget > 0)

        # One split and two ordered skips per row. Multiplication by the
        # row-specific integer range is the vectorised equivalent of randint.
        split = 1 + (torch.rand(len(lengths), 1, device=device)
                     * (lengths - 1).clamp_min(1)).long()
        first = (torch.rand(len(lengths), 1, device=device) * (budget + 1)).long()
        second = first + (
            torch.rand(len(lengths), 1, device=device) * (budget - first + 1)
        ).long()
        first = torch.where(active, first, torch.zeros_like(first))
        second = torch.where(active, second, torch.zeros_like(second))
        shifted = base + torch.where(base < split, first, second)
        return torch.where(base < lengths, shifted, torch.zeros_like(shifted))

    def sample_pose_positions(self, batch_size, length, device):
        """Sample old-style two-chunk PoSE indices with shape ``[B, T]``."""
        self._check_sample_shape(batch_size, length)
        if length < 2 or length == self.config.block_size:
            return torch.arange(length, device=device).expand(batch_size, length)
        lengths = torch.full((batch_size,), length, dtype=torch.long, device=device)
        return self._sample_pose_for_lengths(lengths, length, device)

    def _effective_lengths(self, idx, targets):
        if targets is None:
            return torch.full(
                (idx.size(0),), idx.size(1), dtype=torch.long, device=idx.device
            )
        if targets.shape != idx.shape:
            raise ValueError(
                f"targets must have shape {tuple(idx.shape)}, got {tuple(targets.shape)}"
            )
        supervised = targets.ne(-1)
        one_based = torch.arange(1, idx.size(1) + 1, device=idx.device)
        return (supervised * one_based).amax(dim=1)

    def _sample_shift_for_lengths(self, lengths, width, device):
        base = torch.arange(width, device=device).unsqueeze(0).expand(len(lengths), width)
        lengths = lengths.to(device=device, dtype=torch.long).view(-1, 1)
        budget = self.config.block_size - lengths
        offsets = (torch.rand(len(lengths), 1, device=device) * (budget + 1)).long()
        shifted = base + offsets
        return torch.where(base < lengths, shifted, torch.zeros_like(shifted))

    def forward(self, idx, targets=None, positions=None):
        embedding = self.positional_spec.embedding
        if positions is None and self.training and embedding in ("abs_shift", "pose"):
            lengths = self._effective_lengths(idx, targets)
            if embedding == "abs_shift":
                positions = self._sample_shift_for_lengths(
                    lengths, idx.size(1), idx.device
                )
            else:
                positions = self._sample_pose_for_lengths(
                    lengths, idx.size(1), idx.device
                )
        return super().forward(idx, targets, positions=positions)

    def param_report(self):
        counted = [(p.requires_grad, p.numel())
                   for name, p in self.named_parameters() if _is_positional(name)]
        trainable = sum(n for grad, n in counted if grad)
        frozen = sum(n for grad, n in counted if not grad)
        frozen += sum(b.numel() for name, b in self.named_buffers()
                      if ".positioning." in name)
        return {
            "positional_encoding": self.positional_spec.canonical,
            "positional_parameters": {"trainable": trainable, "frozen": frozen,
                                      "total": trainable + frozen},
        }
