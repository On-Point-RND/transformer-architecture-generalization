"""Common task interface for the synthetic-sequence experiments.

Every task (addition, permutation, sorting, ...) produces prompt->answer
examples encoded as ``(input_ids, target_ids)`` numpy arrays, where target_ids
is ``ignore_index`` everywhere except on the answer tokens. A task exposes just
enough for the shared training loop, sweep runner and layer probes to be
task-agnostic:

  - ``vocab_size``          : token count (to size the model)
  - ``seq_len(length)``     : full sequence length for a length parameter
                             (N digits / K permutations / n items)
  - ``make_splits(...)``    : disjoint (train, test) lists of (input_ids, target)
  - ``answer_span(length)`` : (start, length) of the answer inside the sequence
  - ``difficulty(ids, ...)``: per-example difficulty L(x) (carry-chain length /
                             #non-identity permutations / #inversions)
  - ``sample_balanced(...)``: examples balanced across the difficulty bins
  - ``variants``            : sub-conditions, e.g. ("SEQ","INDEP") / ("S5","C5")
                             or None
  - ``ignore_index``        : loss-mask value (default -100)

The concrete tasks live in the sibling modules (addition.py, permutation.py,
sorting.py, ...) and are registered in ``datasets/__init__.py``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional, Sequence, Tuple

import numpy as np

Example = Tuple[np.ndarray, np.ndarray]  # (input_ids, target_ids)


class Task(ABC):
    name: str = "task"
    variants: Optional[Tuple[str, ...]] = None  # sub-conditions, or None
    difficulty_label: str = "L(x)"
    difficulty_col: str = "difficulty"   # CSV column for the money-plot metric
    by_suffix: str = "by_difficulty"     # results/<sweep>_<by_suffix>.csv
    ignore_index: int = -100

    @property
    @abstractmethod
    def vocab_size(self) -> int: ...

    @abstractmethod
    def seq_len(self, length: int) -> int: ...

    @abstractmethod
    def make_splits(self, length: int, n_train: int, n_test: int, seed: int,
                    variant: Optional[str] = None) -> Tuple[List[Example], List[Example]]: ...

    @abstractmethod
    def answer_span(self, length: int) -> Tuple[int, int]:
        """(start_index, answer_length) of the answer tokens in the sequence."""

    @abstractmethod
    def difficulty(self, input_ids: Sequence[int], length: int) -> int:
        """Sequential-dependency length L(x) for one example."""

    @abstractmethod
    def sample_balanced(self, length: int, per_bin: int, seed: int,
                        variant: Optional[str] = None) -> List[Example]:
        """Examples balanced across difficulty bins (for the money-plot / probes)."""

    def default_variant(self) -> Optional[str]:
        return self.variants[0] if self.variants else None
