"""Task interface: a live stream of prompt->answer examples.

Training and validation data are generated on the fly — there are no on-disk
bins and no meta.pkl. A task implements ``_sample_one`` plus ``vocab_size``;
the base class provides dedup'd train/val streams, the answer-masked collate
and the rng state needed for exact resume.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np


@dataclass
class DatasetItem:
    prompt: np.ndarray
    answer: np.ndarray
    # Optional per-example axes for positional OOD analysis.
    metadata: Dict[str, Any] = field(default_factory=dict)


class Task(ABC):
    PAD_ID: int = 0  # token used to right-pad inputs; subclasses may override

    # Subclasses set ``self.rng = np.random.default_rng(seed)`` and expose a
    # ``vocab_size`` property (the number of distinct token ids they emit).

    @abstractmethod
    def _sample_one(self) -> DatasetItem: ...

    def metrics(self, predicted, targets) -> Dict[str, float]:
        """Named scores for one batch of predictions — override to add your own.

        ``predicted`` and ``targets`` are [batch, block_size] int arrays and
        ``targets`` is -1 outside the answer span. Every value must be a mean
        over the batch: the training loop averages each key across eval batches
        and logs it as ``<split>_<key>``.

        The default is answer exact-match: a row counts only if every answer
        position is right. For single-token answers that is token accuracy.
        """
        answer = targets != -1
        correct = ((predicted == targets) | ~answer).all(axis=1)
        return {"acc": float(correct.astype(np.float32).mean())}

    def collate(self, items: List[DatasetItem], block_size: int):
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
            end = len(seq) - 1  # length of seq[:-1] / seq[1:]
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

    def sample(self, n: int = 1) -> List[DatasetItem]:
        return [self._sample_one() for _ in range(n)]

    @staticmethod
    def _hash(item: DatasetItem) -> bytes:
        return item.prompt.tobytes()

    def generate_val(self, n: int) -> List[DatasetItem]:
        val, seen = [], set()
        while len(val) < n:
            item = self._sample_one()
            h = self._hash(item)
            if h in seen:
                continue
            seen.add(h)
            val.append(item)
        self.val_hashes = seen
        return val

    def sample_train(self, n: int = 1) -> List[DatasetItem]:
        if not hasattr(self, "val_hashes"):
            raise RuntimeError("Call `generate_val(...)' before sampling train")
        out = []
        while len(out) < n:
            item = self._sample_one()
            if self._hash(item) not in self.val_hashes:
                out.append(item)
        return out

    # --- resume ---------------------------------------------------------
    # The held-out val set is regenerated from the same seed on resume, so only
    # the generator position has to be restored — after generate_val, not before.

    def state_dict(self) -> Dict[str, Any]:
        return {"rng": self.rng.bit_generator.state}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.rng.bit_generator.state = state["rng"]
