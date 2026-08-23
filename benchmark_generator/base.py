"""Core abstractions shared by every synthetic benchmark task.

Design follows the paper's setup: each task has four regimes -
train / val / test (same difficulty distribution, different samples)
and test_ood (difficulty shifted along a task-specific axis).
Data is generated on the fly - there is no fixed corpus for `train`.
"""
from __future__ import annotations

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict


class Split(str, Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"
    TEST_OOD = "test_ood"


@dataclass
class Example:
    prompt: str
    target: str
    task: str
    split: str
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prompt": self.prompt,
            "target": self.target,
            "task": self.task,
            "split": self.split,
            "meta": self.meta,
        }


class Task(ABC):
    """Abstract base class every benchmark task implements.

    Subclasses only need to implement `sample_difficulty` (pick the
    difficulty parameters for a given split, e.g. number of digits,
    sequence length, nesting depth ...) and `generate_one` (turn those
    parameters into a concrete (prompt, target) pair).
    """

    name: str = "task"

    @abstractmethod
    def sample_difficulty(self, split: Split, rng: random.Random) -> Dict[str, Any]:
        ...

    @abstractmethod
    def generate_one(self, params: Dict[str, Any], rng: random.Random) -> tuple[str, str]:
        ...

    def generate(self, split: Split, rng: random.Random) -> Example:
        params = self.sample_difficulty(split, rng)
        prompt, target = self.generate_one(params, rng)
        return Example(prompt=prompt, target=target, task=self.name, split=split.value, meta=params)

    # Convenience -----------------------------------------------------
    def generate_many(self, split: Split, n: int, seed: int) -> list[Example]:
        """Deterministic, reproducible batch - use this for val/test/test_ood."""
        rng = random.Random(seed)
        return [self.generate(split, rng) for _ in range(n)]

    def infinite_stream(self, split: Split, seed: int):
        """Unbounded generator - use this for train (never memorised)."""
        rng = random.Random(seed)
        while True:
            yield self.generate(split, rng)

def run_task_cli(task: Task):
    import argparse
    import json
    
    p = argparse.ArgumentParser(description=f"Generator for {task.name}")
    p.add_argument("--split", choices=[s.value for s in Split], default="train")
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default=None, help="Save to JSONL file")
    args = p.parse_args()

    examples = task.generate_many(Split(args.split), args.n, args.seed)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            for ex in examples:
                f.write(json.dumps(ex.to_dict(), ensure_ascii=False) + "\n")
        print(f"Saved {len(examples)} examples -> {args.out}")
    else:
        for ex in examples:
            print(f"{ex.prompt} -> {ex.target}  (meta={ex.meta})")
