"""
Task: Array Sorting (SORT). Sort a sequence of random integers (0-99) in ascending order.
Parameters:
  - Train/Val/Test: array lengths 8 and 16
  - Test-OOD: array lengths 32 and 64
Example:
  "SORT 79 32 68 90 77 18 39 12 =" -> "12 18 32 39 68 77 79 90"
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, Tuple

from base import Split, Task


@dataclass
class SortingConfig:
    length_choices: Dict[Split, Tuple[int, ...]] = field(default_factory=lambda: {
        Split.TRAIN: (8, 16),
        Split.VAL: (8, 16),
        Split.TEST: (8, 16),
        Split.TEST_OOD: (32, 64),
    })
    value_range: Tuple[int, int] = (0, 99)


class SortingTask(Task):
    name = "sorting"

    def __init__(self, config: SortingConfig | None = None):
        self.config = config or SortingConfig()

    def sample_difficulty(self, split: Split, rng: random.Random) -> Dict[str, Any]:
        length = rng.choice(self.config.length_choices[split])
        return {"length": length}

    def generate_one(self, params: Dict[str, Any], rng: random.Random) -> Tuple[str, str]:
        n = params["length"]
        lo, hi = self.config.value_range
        values = [rng.randint(lo, hi) for _ in range(n)]
        prompt = "SORT " + " ".join(map(str, values)) + " ="
        target = " ".join(map(str, sorted(values)))
        return prompt, target


if __name__ == "__main__":
    from base import run_task_cli
    run_task_cli(SortingTask())