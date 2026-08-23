"""Long Addition (ADD).

Paper spec (Sec 2.1 / 3.3, Table 2):
  - digit-by-digit addition of two integers, "reversed-digit format"
    is used in the abstract because it is the format that admits a
    short RASP-L programme (Zhou et al.) and therefore length-generalises.
  - train/val/test: 5- and 7-digit numbers
  - test_ood: 10- and 20-digit numbers
  - example (display / non-reversed form): "58392 74918 =" -> "133310"
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, Tuple

from base import Split, Task


@dataclass
class AdditionConfig:
    digit_choices: Dict[Split, Tuple[int, ...]] = field(default_factory=lambda: {
        Split.TRAIN: (5, 7),
        Split.VAL: (5, 7),
        Split.TEST: (5, 7),
        Split.TEST_OOD: (10, 20),
    })
    reverse_digits: bool = True


class AdditionTask(Task):
    name = "addition"

    def __init__(self, config: AdditionConfig | None = None):
        self.config = config or AdditionConfig()

    def sample_difficulty(self, split: Split, rng: random.Random) -> Dict[str, Any]:
        digits = rng.choice(self.config.digit_choices[split])
        return {"digits": digits}

    def _rand_n_digit(self, digits: int, rng: random.Random) -> int:
        if digits <= 1:
            return rng.randint(0, 9)
        lo, hi = 10 ** (digits - 1), 10 ** digits - 1
        return rng.randint(lo, hi)

    def generate_one(self, params: Dict[str, Any], rng: random.Random) -> Tuple[str, str]:
        d = params["digits"]
        a = self._rand_n_digit(d, rng)
        b = self._rand_n_digit(d, rng)
        s = a + b

        if self.config.reverse_digits:
            a_str, b_str, t_str = str(a)[::-1], str(b)[::-1], str(s)[::-1]
        else:
            a_str, b_str, t_str = str(a), str(b), str(s)

        prompt = f"ADD {a_str} {b_str} ="
        return prompt, t_str
    
if __name__ == "__main__":
    from base import run_task_cli
    run_task_cli(AdditionTask())