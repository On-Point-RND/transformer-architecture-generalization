from typing import Tuple

import numpy as np

from .base import DatasetItem, Generator


class AdditionGenerator(Generator):
    PAD_ID = 0
    PLUS_ID = 1
    EQ_ID = 2
    N_SPECIAL = 3

    BASE = 10

    def __init__(
        self,
        n_digits: int | Tuple[int, int],
        reverse: bool = True,
        seed: int | None = 42,
    ):
        """
        Generator for the decimal addition task.

        Prompt layout:  a_0 .. a_{d-1}  PLUS  b_0 .. b_{d-1}  EQ
        Answer layout:  s_0 .. s_d              (fixed width d+1, zero-padded)

        Both operands have exactly ``d`` digit tokens (leading zeros allowed), so
        prompt length is deterministic given ``d``, and the answer always has
        ``d + 1`` tokens, so the number of supervised positions never leaks the
        carry-out.

        :param n_digits: Digits per operand. Fixed ``n_digits`` if an integer, or
            sampled from [n_digits[0], n_digits[1]) if a tuple.
        :type n_digits: int | Tuple[int, int]
        :param reverse: If True, emit operands and answer least-significant-digit
            first. This aligns answer position i with operand positions <= i under
            the carry recurrence, and is the main lever on length generalization.
        :type reverse: bool
        :param seed: Randomization seed, None for non-reproducible environment.
        :type seed: int | None
        """
        self.n_digits = n_digits
        self.reverse = reverse

        self.d_token_ids = np.arange(self.BASE) + self.N_SPECIAL

        self.rng = np.random.default_rng(seed)

    @property
    def vocab_size(self) -> int:
        return self.N_SPECIAL + self.BASE

    @property
    def max_digits(self) -> int:
        if isinstance(self.n_digits, int):
            return self.n_digits
        return self.n_digits[1] - 1

    @property
    def min_block_size(self) -> int:
        """Smallest ``block_size`` that ``collate`` will accept.

        len(prompt) + len(answer) = (2d + 2) + (d + 1) = 3d + 3 <= block_size + 1
        """
        return 3 * self.max_digits + 2

    def _add_digits(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """MSB-first digit arrays of length d -> MSB-first sum of length d+1.

        Done digit-wise rather than via int conversion so that large ``d`` (OOD
        evaluation at 16, 20, ... digits) cannot overflow.
        """
        d = len(a)
        out = np.empty(d + 1, dtype=np.int64)
        carry = 0
        for i in range(d - 1, -1, -1):
            s = int(a[i]) + int(b[i]) + carry
            out[i + 1] = s % self.BASE
            carry = s // self.BASE
        out[0] = carry
        return out

    def _sample_one(self) -> DatasetItem:
        if isinstance(self.n_digits, int):
            d = self.n_digits
        else:
            d = int(self.rng.integers(self.n_digits[0], self.n_digits[1]))

        a = self.rng.integers(0, self.BASE, size=d)
        b = self.rng.integers(0, self.BASE, size=d)
        s = self._add_digits(a, b)

        if self.reverse:
            a, b, s = a[::-1], b[::-1], s[::-1]

        prompt = np.concatenate(
            [
                self.d_token_ids[a],
                np.array([self.PLUS_ID]),
                self.d_token_ids[b],
                np.array([self.EQ_ID]),
            ]
        ).astype(np.int64)
        answer = self.d_token_ids[s].astype(np.int64)

        return DatasetItem(prompt=prompt, answer=answer)
