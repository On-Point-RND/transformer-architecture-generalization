"""Selective counting: how many marks are hidden among the fillers?

    RESET . R . . R . . . R . QUERY   ->   3

The marks sit at random positions, so the count is an aggregate over the whole
sequence: no single position answers it, and the model has to reach back as
far as the leftmost mark. The answer is one class token, CLASS_LOW + count.
Most of the 160 token ids go unused; they are kept so the stream matches the
runs made before this task had a file of its own.
"""

import numpy as np

from .base import DatasetItem, Task, max_int, sample_int, validate_int_spec


class SelectiveCountTask(Task):
    PAD_ID = 0
    QUERY, RESET, RELEVANT = 2, 6, 7
    FILL_LOW, FILL_HIGH = 16, 63
    CLASS_LOW = 128

    def __init__(
        self,
        length_range=(16, 33),
        relevant_count_range=(1, 9),
        seed=42,
    ):
        """
        :param length_range: sequence length, [lo, hi).
        :param relevant_count_range: how many marks, [lo, hi); capped at length - 3.
        """
        super().__init__(seed)
        self.length_range = validate_int_spec(length_range, "length_range", 4)
        self.relevant_count_range = validate_int_spec(
            relevant_count_range, "relevant_count_range", 1
        )
        max_count = min(max_int(self.relevant_count_range), max_int(self.length_range) - 3)
        if self.CLASS_LOW + max_count >= self.vocab_size:
            raise ValueError(f"relevant_count_range exceeds the class token block "
                             f"(max {self.vocab_size - self.CLASS_LOW - 1})")

    @property
    def vocab_size(self):
        return 160

    def _sample_one(self) -> DatasetItem:
        length = sample_int(self.rng, self.length_range)
        count = min(sample_int(self.rng, self.relevant_count_range), length - 3)
        prompt = self.rng.integers(self.FILL_LOW, self.FILL_HIGH + 1, size=length, dtype=np.int64)
        prompt[0], prompt[-1] = self.RESET, self.QUERY
        positions = self.rng.choice(np.arange(1, length - 1), size=count, replace=False)
        prompt[positions] = self.RELEVANT
        first = int(positions.min())
        metadata = {
            "sequence_length": length,
            "absolute_target_position": first,
            "normalized_target_position": first / max(length - 1, 1),
            "relative_distance": length - 1 - first,
            "relevant_count": count,
            "number_of_distractors": length - count - 2,
        }
        return DatasetItem(prompt, np.array([self.CLASS_LOW + count], dtype=np.int64), metadata)
