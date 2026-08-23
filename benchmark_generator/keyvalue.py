"""
Task: Key-Value Retrieval (KV). Retrieve the target value for a queried key from a padded key-value dict.
Parameters:
  - Train/Val/Test: 16 and 64 pairs
  - Test-OOD: 256 and 1024 pairs
Example:
  "k0037:v735 k0056:v126 k0011:v490 QUERY k0011 =" -> "v490"
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, Tuple

from base import Split, Task


@dataclass
class KVConfig:
    n_choices: Dict[Split, Tuple[int, ...]] = field(
        default_factory=lambda: {
            Split.TRAIN: (4, 10),
            Split.VAL: (4, 10),
            Split.TEST: (4, 10),
            Split.TEST_OOD: (20, 40),
        }
    )
    value_range: Tuple[int, int] = (0, 999)
    # key pool multiplier relative to n, must stay unique at largest n (1024)
    key_pool_multiplier: int = 8

    key_width: int = 4
    value_width: int = 3


class KVTask(Task):
    name = "kv"

    def __init__(self, config: KVConfig | None = None):
        self.config = config or KVConfig()

    def sample_difficulty(self, split: Split, rng: random.Random) -> Dict[str, Any]:
        n = rng.choice(self.config.n_choices[split])
        return {"n": n}

    def generate_one(self, params: Dict[str, Any], rng: random.Random) -> Tuple[str, str]:
        n = params["n"]
        pool_size = max(n * self.config.key_pool_multiplier, n + 1)
        keys = rng.sample(range(pool_size), n)
        lo, hi = self.config.value_range
        values = [rng.randint(lo, hi) for _ in range(n)]

        query_idx = rng.randrange(n)
        query_key, answer = keys[query_idx], values[query_idx]

        w_k = self.config.key_width
        w_v = self.config.value_width
        body = " ".join(f"k{k:0{w_k}d}:v{v:0{w_v}d}" for k, v in zip(keys, values))
        prompt = f"{body} QUERY k{query_key:0{w_k}d} ="
        target = f"v{answer:0{w_v}d}"
        return prompt, target


if __name__ == "__main__":
    from base import run_task_cli
    run_task_cli(KVTask())