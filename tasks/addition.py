from typing import Tuple

import numpy as np

from .base import DatasetItem, Task


class AdditionTask(Task):
    PAD_ID = 0
    PLUS_ID = 1
    EQ_ID = 2
    N_SPECIAL = 3

    BASE = 10

    def __init__(
        self,
        n_digits: int | Tuple[int, int],
        reverse: bool = True,
        carry: bool = True,
        bos_eos: bool = False,
        seed: int | None = 42,
    ):
        """
        Generator for the decimal addition task.

        Prompt layout:  a_0 .. a_{d-1}  PLUS  b_0 .. b_{d-1}  EQ
        Answer layout:  s_0 .. s_d              (fixed width d+1, zero-padded)

        With ``bos_eos`` the sequence gains the BOS/EOS wrapper used by the
        depth study, ``BOS a PLUS b EQ s EOS``, and EOS becomes part of the
        answer, so the model has to predict where the answer ends.

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
        :param carry: True is ordinary addition, where a carry propagates and
            makes the task sequential. False sums each digit independently
            (``c_i = (a_i + b_i) % 10``), which is the parallel control: same
            length, same vocabulary, no dependency between positions.
        :type carry: bool
        :param bos_eos: Wrap the sequence in BOS/EOS and supervise the EOS.
        :type bos_eos: bool
        :param seed: Randomization seed, None for non-reproducible environment.
        :type seed: int | None
        """
        self.n_digits = n_digits
        self.reverse = reverse
        self.carry = carry
        self.bos_eos = bos_eos

        # token ids follow the layout in use, so the plain task keeps the
        # vocabulary of 13 it was published with and the wrapped one gets 15
        if bos_eos:
            self.BOS_ID, self.EOS_ID, self.PLUS_ID, self.EQ_ID = 1, 2, 3, 4
            self.n_special = 5
        else:
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
        """Smallest ``block_size`` that ``collate`` will accept.

        len(prompt) + len(answer) = (2d + 2) + (d + 1) = 3d + 3 <= block_size + 1,
        plus the two wrapper tokens when bos_eos is on.
        """
        return 3 * self.max_digits + 2 + 2 * self.bos_eos

    def metrics(self, predicted, targets):
        """Exact match plus per-digit accuracy: 9 of 10 digits right is not 0."""
        scores = super().metrics(predicted, targets)
        answer = targets != -1
        scores["digit_acc"] = float((predicted == targets)[answer].astype(np.float32).mean())
        return scores

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

    def _carry_chain(self, a: np.ndarray, b: np.ndarray) -> int:
        """Longest run of consecutive positions that each hand a carry onwards.

        This is the difficulty L(x): how far a carry actually has to travel, not
        how many carries there are. Digits arrive most-significant first, so the
        scan runs backwards.
        """
        carry, run, longest = 0, 0, 0
        for i in range(len(a) - 1, -1, -1):
            carry = int(int(a[i]) + int(b[i]) + carry >= self.BASE)
            run = run + 1 if carry else 0
            longest = max(longest, run)
        return longest

    def _sample_one(self) -> DatasetItem:
        if isinstance(self.n_digits, int):
            d = self.n_digits
        else:
            d = int(self.rng.integers(self.n_digits[0], self.n_digits[1]))

        a = self.rng.integers(0, self.BASE, size=d)
        b = self.rng.integers(0, self.BASE, size=d)
        chain = self._carry_chain(a, b)
        if self.carry:
            s = self._add_digits(a, b)
        else:
            # per-digit sums; the leading digit is always 0, so the answer keeps
            # the width d+1 and the two conditions look identical from outside
            s = np.concatenate([[0], (a + b) % self.BASE]).astype(np.int64)

        if self.reverse:
            a, b, s = a[::-1], b[::-1], s[::-1]

        head = [np.array([self.BOS_ID])] if self.bos_eos else []
        prompt = np.concatenate(
            head
            + [
                self.d_token_ids[a],
                np.array([self.PLUS_ID]),
                self.d_token_ids[b],
                np.array([self.EQ_ID]),
            ]
        ).astype(np.int64)
        answer = self.d_token_ids[s]
        if self.bos_eos:
            answer = np.concatenate([answer, np.array([self.EOS_ID])])

        return DatasetItem(
            prompt=prompt,
            answer=answer.astype(np.int64),
            metadata={"n_digits": d, "chain_length": chain,
                      "variant": "SEQ" if self.carry else "INDEP"},
        )
