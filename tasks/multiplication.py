from typing import Tuple

import numpy as np

from .base import DatasetItem, Task


class MultiplicationTask(Task):
    PAD_ID = 0
    MUL_ID = 1
    EQ_ID = 2
    N_SPECIAL = 3

    BASE = 10

    def __init__(
        self,
        n_digits: int | Tuple[int, int],
        multiplier_digits: int = 3,
        reverse: bool = False,
        seed: int | None = 42,
    ):
        """
        Generator for decimal multiplication.

        Prompt layout:
            a_0 .. a_{d-1}  MUL  b_0 .. b_{m-1}  EQ

        Answer layout:
            p_0 .. p_{d+m-1}

        The first operand has exactly ``d`` digits, where ``d`` is either fixed
        by ``n_digits`` or sampled from [n_digits[0], n_digits[1]).

        The second operand has exactly ``multiplier_digits`` digits.

        The answer always has exactly ``d + multiplier_digits`` digit tokens,
        with leading zeros when necessary. This prevents answer length from
        leaking information about the product.

        Examples
        --------
        ID training:
            MultiplicationTask(n_digits=5, multiplier_digits=3)

        OOD evaluation:
            MultiplicationTask(n_digits=35, multiplier_digits=3)

        Variable-length evaluation:
            MultiplicationTask(n_digits=(5, 36), multiplier_digits=3)

        Parameters
        ----------
        n_digits:
            Digits in the first operand. Fixed if an integer, or sampled from
            [lo, hi) if a tuple.

        multiplier_digits:
            Number of digits in the second operand.

        reverse:
            If True, emit operands and answer least-significant-digit first.

        seed:
            Randomization seed. None gives non-reproducible sampling.
        """
        self.n_digits = n_digits
        self.multiplier_digits = multiplier_digits
        self.reverse = reverse

        if isinstance(n_digits, int):
            if n_digits < 1:
                raise ValueError("n_digits must be >= 1")
        else:
            lo, hi = n_digits
            if lo < 1 or hi <= lo:
                raise ValueError(
                    "n_digits tuple must satisfy 1 <= lo < hi"
                )

        if multiplier_digits < 1:
            raise ValueError("multiplier_digits must be >= 1")

        self.n_special = self.N_SPECIAL
        self.d_token_ids = np.arange(self.BASE) + self.n_special

        self.rng = np.random.default_rng(seed)

    @property
    def vocab_size(self) -> int:
        return self.n_special + self.BASE

    @property
    def max_digits(self) -> int:
        if isinstance(self.n_digits, int):
            return self.n_digits
        return self.n_digits[1] - 1

    @property
    def min_block_size(self) -> int:
        """
        Smallest block_size accepted by ``Task.collate``.

        For first-operand length d and multiplier length m:

            prompt:
                d digits + MUL + m digits + EQ
                = d + m + 2

            answer:
                d + m digits

            total:
                2d + 2m + 2

        ``collate`` requires

            len(prompt) + len(answer) <= block_size + 1

        so

            block_size >= 2d + 2m + 1.
        """
        return (
            2 * self.max_digits
            + 2 * self.multiplier_digits
            + 1
        )

    def metrics(self, predicted, targets):
        """
        Exact-match accuracy plus per-digit answer accuracy.
        """
        scores = super().metrics(predicted, targets)

        answer = targets != -1

        scores["digit_acc"] = float(
            (predicted == targets)[answer]
            .astype(np.float32)
            .mean()
        )

        return scores

    def _sample_n_digits(self) -> int:
        """
        Sample the length of the first operand.
        """
        if isinstance(self.n_digits, int):
            return self.n_digits

        return int(
            self.rng.integers(
                self.n_digits[0],
                self.n_digits[1],
            )
        )

    def _sample_digits(self, n: int) -> np.ndarray:
        """
        Sample an exactly n-digit positive decimal integer as an MSB-first
        digit array.

        The most-significant digit is sampled from 1..9, so leading zeros
        are not allowed.
        """
        digits = self.rng.integers(
            0,
            self.BASE,
            size=n,
            dtype=np.int64,
        )

        digits[0] = self.rng.integers(
            1,
            self.BASE,
        )

        return digits

    def _multiply_digits(
        self,
        a: np.ndarray,
        b: np.ndarray,
    ) -> np.ndarray:
        """
        Multiply two MSB-first decimal digit arrays.

        Returns an MSB-first array of fixed length

            len(a) + len(b).

        The implementation uses grade-school multiplication rather than
        converting the operands to NumPy integers. This avoids overflow for
        long OOD examples such as 35-digit operands.

        Leading zeros in the output are retained so answer length depends
        only on operand lengths.

        Example
        -------
        123 * 45 = 5535

        Since 3-digit x 2-digit multiplication has maximum width 5:

            [1, 2, 3] * [4, 5]
                -> [0, 5, 5, 3, 5]
        """
        da = len(a)
        db = len(b)

        out = np.zeros(
            da + db,
            dtype=np.int64,
        )

        # Standard grade-school multiplication.
        for i in range(da - 1, -1, -1):
            for j in range(db - 1, -1, -1):
                out[i + j + 1] += int(a[i]) * int(b[j])

        # Propagate carries from right to left.
        for k in range(len(out) - 1, 0, -1):
            carry = int(out[k]) // self.BASE

            out[k] %= self.BASE
            out[k - 1] += carry

        # The maximum product of a d-digit and m-digit number fits in
        # d + m digits, so the first position must now also be a digit.
        if out[0] >= self.BASE:
            raise RuntimeError(
                "Internal multiplication error: carry overflow"
            )

        return out

    def _sample_one(self) -> DatasetItem:
        # Length controlling the generalization experiment.
        d = self._sample_n_digits()

        # Exactly d digits.
        a = self._sample_digits(d)

        # Exactly multiplier_digits digits.
        b = self._sample_digits(self.multiplier_digits)

        # Fixed-width product of length d + multiplier_digits.
        product = self._multiply_digits(a, b)

        if self.reverse:
            a = a[::-1]
            b = b[::-1]
            product = product[::-1]

        prompt = np.concatenate(
            [
                self.d_token_ids[a],
                np.array([self.MUL_ID]),
                self.d_token_ids[b],
                np.array([self.EQ_ID]),
            ]
        ).astype(np.int64)

        answer = self.d_token_ids[product].astype(np.int64)

        return DatasetItem(
            prompt=prompt,
            answer=answer,
            metadata={
                "n_digits": d,
                "multiplier_digits": self.multiplier_digits,
                "length_group": "LONG" if d > 5 else "SHORT",
                "variant": "MULT",
            },
        )
