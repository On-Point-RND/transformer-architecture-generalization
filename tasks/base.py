"""Task interface: a live stream of prompt->answer examples.

Training and validation data are generated on the fly — there are no on-disk
bins and no meta.pkl. A task implements ``_sample_one`` plus ``vocab_size``;
the base class provides dedup'd train/val streams, the answer-masked collate
and the rng state needed for exact resume.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from numbers import Integral
from typing import Any

import numpy as np


def validate_int_spec(spec, name: str, minimum: int = 0):
    """Validate and normalize an integer or a half-open ``[lo, hi)`` range."""
    if isinstance(spec, Integral) and not isinstance(spec, bool):
        value = int(spec)
        if value < minimum:
            raise ValueError(f"{name} must be >= {minimum}, got {value}")
        return value
    if not isinstance(spec, (tuple, list)) or len(spec) != 2:
        raise TypeError(f"{name} must be an integer or a [lo, hi) pair, got {spec!r}")
    lo, hi = spec
    if (not isinstance(lo, Integral) or isinstance(lo, bool)
            or not isinstance(hi, Integral) or isinstance(hi, bool)):
        raise TypeError(f"{name} bounds must be integers, got {spec!r}")
    lo, hi = int(lo), int(hi)
    if lo < minimum or hi <= lo:
        raise ValueError(
            f"{name} must satisfy {minimum} <= lo < hi for [lo, hi), got {spec!r}"
        )
    return lo, hi


def sample_int(rng, spec) -> int:
    """Return an int, or sample from [lo, hi) when given a range.

    One convention for every task, the one range() uses: the upper bound is
    excluded, so ``n_pairs: [2, 25]`` yields 2..24.
    """
    if isinstance(spec, Integral):
        return spec
    return int(rng.integers(spec[0], spec[1]))


def max_int(spec) -> int:
    """Return the largest value ``sample_int`` can produce."""
    return int(spec) if isinstance(spec, Integral) else spec[1] - 1


@dataclass
class DatasetItem:
    prompt: np.ndarray
    answer: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)


class Task(ABC):
    PAD_ID: int = 0

    def __init__(self, seed: int | None):
        self.rng = np.random.default_rng(seed)
        self._val_prompts: set[bytes] | None = None
        self._train_pool: list[DatasetItem] | None = None

    @property
    @abstractmethod
    def vocab_size(self) -> int: ...

    @abstractmethod
    def _sample_one(self) -> DatasetItem: ...

    def metrics(self, predicted, targets) -> dict[str, float]:
        """Named scores for one batch of predictions — override to add your own.

        ``predicted`` and ``targets`` are [batch, block_size] int arrays and
        ``targets`` is -1 outside the answer span. Every value must be a mean
        over the batch: the training loop averages each key across eval batches
        and logs it as ``<split>_<key>``.

        ``acc`` is answer exact-match: a row counts only if every answer
        position is right. ``token_acc`` is the fraction of answer tokens that
        are right, so 9 of 10 digits is 0.9 rather than 0. For single-token
        answers the two coincide.
        """
        answer = targets != -1
        correct = ((predicted == targets) | ~answer).all(axis=1)
        return {"acc": float(correct.astype(np.float32).mean()),
                "token_acc": float((predicted == targets)[answer].astype(np.float32).mean())}

    def collate(self, items: list[DatasetItem], block_size: int):
        """Pack prompt->answer items into answer-masked (x, y) arrays.

        For each item, ``seq = concat(prompt, answer)``. The input ``x`` is
        ``seq[:-1]`` right-padded to ``block_size`` with ``PAD_ID``. The target
        ``y`` is ``-1`` everywhere (ignored by ``F.cross_entropy(ignore_index=-1)``)
        except the final ``len(answer)`` positions of ``seq[1:]``, which hold the
        answer tokens. Right-padding is safe for a causal model: the answer span
        precedes the padding, so attention never looks forward into it.

        Returns two ``(len(items), block_size)`` int64 numpy arrays. torch is
        intentionally kept out of this package; the training loop tensorizes.
        """
        x = np.full((len(items), block_size), self.PAD_ID, dtype=np.int64)
        y = np.full((len(items), block_size), -1, dtype=np.int64)
        for b, item in enumerate(items):
            seq = np.concatenate([item.prompt, item.answer]).astype(np.int64)
            self._check_fits(seq, block_size)
            end = len(seq) - 1
            x[b, :end] = seq[:-1]
            y[b, end - len(item.answer) : end] = item.answer
        return x, y

    @staticmethod
    def _check_fits(seq, block_size):
        if len(seq) <= block_size + 1:
            return
        raise ValueError(
            f"packed example length {len(seq)} exceeds block_size+1="
            f"{block_size + 1}; increase block_size or shrink the task "
            f"(e.g. fewer pairs) so that len(prompt)+len(answer) <= block_size+1"
        )

    @staticmethod
    def _prompt_key(item: DatasetItem) -> bytes:
        return item.prompt.tobytes()

    MAX_REJECTS = 10_000

    def _sample_unique(self, taken, what: str) -> DatasetItem:
        """An example whose prompt is not already in ``taken``."""
        for _ in range(self.MAX_REJECTS):
            item = self._sample_one()
            if self._prompt_key(item) not in taken:
                return item
        raise ValueError(
            f"{type(self).__name__} drew {self.MAX_REJECTS} duplicate prompts in a "
            f"row while building {what}: the task has fewer distinct prompts than "
            f"were asked of it. Request fewer examples (task.n_val / task.n_train) "
            f"or widen the task, e.g. more items or a larger vocabulary."
        )

    def generate_val(self, n: int) -> list[DatasetItem]:
        val, seen = [], set()
        while len(val) < n:
            item = self._sample_unique(seen, f"the {n}-example validation set")
            seen.add(self._prompt_key(item))
            val.append(item)
        self._val_prompts = seen
        return val

    def sample_train(self, n: int = 1) -> list[DatasetItem]:
        if self._val_prompts is None:
            raise RuntimeError("Call `generate_val(...)' before sampling train")
        if self._train_pool is not None:
            indices = self.rng.integers(len(self._train_pool), size=n)
            return [self._train_pool[i] for i in indices]
        return [self._sample_unique(self._val_prompts, "a training batch")
                for _ in range(n)]

    def build_train_pool(self, n: int) -> None:
        """Train from a fixed set of n examples instead of an endless stream.

        The size of that set is an experimental variable, not an implementation
        detail: the same model memorises 100k examples and generalises from 1M.
        Batches are then drawn from the pool uniformly with replacement. The
        pool is rebuilt from the seed on resume, like the val set, so nothing
        extra goes into state_dict.
        """
        self._train_pool = self.sample_train(n)

    def state_dict(self) -> dict[str, Any]:
        return {"rng": self.rng.bit_generator.state}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.rng.bit_generator.state = state["rng"]
