"""
Task: Indexing (INDEX). Return the element at a specific integer index from a sequence.
Parameters:
  - Train/Val/Test: array lengths 16 and 64
  - Test-OOD: array lengths 256 and 1024
Example:
  "ARR s u r t v c a d u g t s d m c l IDX 3 =" -> "t"
"""

from __future__ import annotations

import random
import string
from dataclasses import dataclass, field
from typing import Any, Dict, Tuple

from base import Split, Task


@dataclass
class IndexConfig:
    length_choices: Dict[Split, Tuple[int, ...]] = field(default_factory=lambda: {
        Split.TRAIN: (16, 64),
        Split.VAL: (16, 64),
        Split.TEST: (16, 64),
        Split.TEST_OOD: (256, 1024),
    })
    alphabet: str = string.ascii_lowercase


class IndexTask(Task):
    name = "indexing"

    def __init__(self, config: IndexConfig | None = None):
        self.config = config or IndexConfig()

    def sample_difficulty(self, split: Split, rng: random.Random) -> Dict[str, Any]:
        length = rng.choice(self.config.length_choices[split])
        return {"length": length}

    def generate_one(self, params: Dict[str, Any], rng: random.Random) -> Tuple[str, str]:
        n = params["length"]
        arr = [rng.choice(self.config.alphabet) for _ in range(n)]
        idx = rng.randrange(n)
        prompt = "ARR " + " ".join(arr) + f" IDX {idx} ="
        target = arr[idx]
        return prompt, target


if __name__ == "__main__":
    from base import run_task_cli
    run_task_cli(IndexTask())