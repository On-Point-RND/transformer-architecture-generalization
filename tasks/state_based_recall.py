"""State-based recall (Olmo Hybrid): track a pointer, then read memory at it.

    b_0 .. b_{n-1}  INIT a 36 b 23 c 12 d 2 e 56  OPS a c  b e ..  QUERY a   ->   b_{final a}

A bit array is followed by five pointer variables holding distinct addresses
into it, then a sequence of swaps between the variables. The answer is the bit
at the address variable a holds after all the swaps. Neither half is enough:
tracking gives the address but not the bit, and retrieval cannot start before
the address is known. It is state tracking composed with content retrieval, as
when code updates variables and then indexes memory with them.

Reading the bit at a's starting address is right whenever a ends where it
began; metadata ``moved`` marks the examples where that shortcut fails.
"""

from typing import Tuple

import numpy as np

from .base import DatasetItem, Task, max_int, sample_int, validate_int_spec

N_POINTERS = 5


class StateBasedRecallTask(Task):
    PAD_ID = 0
    INIT_ID, OPS_ID, QUERY_ID = 1, 2, 3
    BIT_LOW = 4
    VAR_LOW = 6
    ADDR_LOW = VAR_LOW + N_POINTERS

    def __init__(
        self,
        n_bits: int | Tuple[int, int] = (16, 33),
        n_swaps: int | Tuple[int, int] = (4, 17),
        max_bits: int | None = None,
        seed: int | None = 42,
    ):
        """
        :param n_bits: length of the bit array, an int or [lo, hi).
        :param n_swaps: swaps between pointer variables.
        :param max_bits: size of the address vocabulary; None means the largest
            n_bits. Set it above that to evaluate on longer arrays later.
        """
        super().__init__(seed)
        self.n_bits = validate_int_spec(n_bits, "n_bits", N_POINTERS)
        self.n_swaps = validate_int_spec(n_swaps, "n_swaps", 0)
        self.max_bits = max_int(self.n_bits) if max_bits is None else validate_int_spec(
            max_bits, "max_bits", N_POINTERS
        )
        if max_int(self.n_bits) > self.max_bits:
            raise ValueError(f"max_bits={self.max_bits} cannot address arrays of "
                             f"length {max_int(self.n_bits)}")

    @property
    def vocab_size(self) -> int:
        return self.ADDR_LOW + self.max_bits

    def _sample_one(self) -> DatasetItem:
        n = sample_int(self.rng, self.n_bits)
        m = sample_int(self.rng, self.n_swaps)
        bits = self.rng.integers(0, 2, size=n)
        initial = self.rng.choice(n, size=N_POINTERS, replace=False).tolist()
        swaps = [self.rng.choice(N_POINTERS, size=2, replace=False).tolist() for _ in range(m)]

        pointers = list(initial)
        for x, y in swaps:
            pointers[x], pointers[y] = pointers[y], pointers[x]
        address = pointers[0]

        prompt = [self.BIT_LOW + int(b) for b in bits] + [self.INIT_ID]
        prompt += [token for i, p in enumerate(initial)
                   for token in (self.VAR_LOW + i, self.ADDR_LOW + p)]
        prompt.append(self.OPS_ID)
        prompt += [self.VAR_LOW + v for pair in swaps for v in pair]
        prompt += [self.QUERY_ID, self.VAR_LOW]

        metadata = {
            "n_bits": n,
            "n_swaps": m,
            "sequence_length": len(prompt),
            "address": address,
            "moved": address != initial[0],
        }
        return DatasetItem(np.array(prompt, dtype=np.int64),
                           np.array([self.BIT_LOW + int(bits[address])], dtype=np.int64),
                           metadata)
