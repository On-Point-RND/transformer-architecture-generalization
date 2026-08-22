from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np


@dataclass
class DatasetItem:
    prompt: np.ndarray
    answer: np.ndarray
    # Optional per-example axes for positional OOD analysis. Existing
    # generators construct DatasetItem(prompt, answer) unchanged.
    metadata: Dict[str, Any] = field(default_factory=dict)


class Generator(ABC):
    seed: int = 42
    PAD_ID: int = 0  # token used to right-pad inputs; subclasses may override

    # Subclasses must also expose a ``vocab_size`` property (the number of
    # distinct token ids they emit). ``train.py`` reads it to size the model,
    # in place of the old ``meta['vocab_size']``.

    @abstractmethod
    def _sample_one(self) -> DatasetItem: ...

    def collate(self, items: List[DatasetItem], block_size: int):
        """Pack prompt->answer items into answer-masked (x, y) arrays.

        For each item, ``seq = concat(prompt, answer)``. The input ``x`` is
        ``seq[:-1]`` right-padded to ``block_size`` with ``PAD_ID``. The target
        ``y`` is ``-1`` everywhere (ignored by ``F.cross_entropy(ignore_index=-1)``)
        except the final ``len(answer)`` positions of ``seq[1:]``, which hold the
        answer tokens. Right-padding is safe for a causal model: the answer span
        precedes the padding, so attention never looks forward into it.

        Returns two ``(len(items), block_size)`` int64 numpy arrays. torch is
        intentionally kept out of this package; ``train.py`` tensorizes.
        """
        n = len(items)
        x = np.full((n, block_size), self.PAD_ID, dtype=np.int64)
        y = np.full((n, block_size), -1, dtype=np.int64)
        for b, item in enumerate(items):
            seq = np.concatenate([item.prompt, item.answer]).astype(np.int64)
            if len(seq) > block_size + 1:
                raise ValueError(
                    f"packed example length {len(seq)} exceeds block_size+1="
                    f"{block_size + 1}; increase block_size or shrink the task "
                    f"(e.g. fewer pairs) so that len(prompt)+len(answer) <= "
                    f"block_size+1"
                )
            la = len(item.answer)
            end = len(seq) - 1  # length of seq[:-1] / seq[1:]
            x[b, :end] = seq[:-1]
            y[b, end - la : end] = item.answer
        return x, y

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
