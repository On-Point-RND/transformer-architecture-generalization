from typing import Tuple

import numpy as np

from .base import DatasetItem, Generator


class DyckGenerator(Generator):
    PAD_ID = 0
    COMPLETE_MARKER_ID = 1
    N_SPECIAL = 2

    def __init__(
        self,
        n_types: int = 2,
        prefix_len: int | Tuple[int, int] = 16,
        max_depth: int = 8,
        p_open: float = 0.5,
        seed: int | None = 42,
    ):
        """
        Generator for Dyck-k prefix completion.

        Prompt layout:  <prefix of a Dyck-k word>  COMPLETE
        Answer layout:  the closers of every still-open bracket, innermost first

        The answer is the unique suffix that balances the prefix, so its length
        equals the stack depth at the truncation point and varies across samples.
        Prefixes are resampled until the depth is at least 1 (an empty answer
        would leave the example with no supervised positions).

        :param n_types: Number of bracket types (the k in Dyck-k).
        :type n_types: int
        :param prefix_len: Length of the prompt prefix in bracket tokens.
            `prefix_len' if the parameter is an integer, and a value in
            [prefix_len[0], prefix_len[1]) if it is a tuple.
        :type prefix_len: int | Tuple[int, int]
        :param max_depth: Cap on nesting depth; also caps the answer length. This
            is the knob that separates depth-OOD from length-OOD, which are
            otherwise confounded when only prefix_len is swept.
        :type max_depth: int
        :param p_open: Probability of opening rather than closing when both moves
            are legal. Values above 0.5 bias toward deeper stacks and longer
            answers, below 0.5 toward shallower ones.
        :type p_open: float
        :param seed: Randomization seed, None for non-reproducible environment.
        :type seed: int | None
        """
        if max_depth < 1:
            raise ValueError("max_depth must be at least 1")

        self.n_types = n_types
        self.prefix_len = prefix_len
        self.max_depth = max_depth
        self.p_open = p_open

        self.open_token_ids = np.arange(n_types) + self.N_SPECIAL
        self.close_token_ids = np.arange(n_types) + self.N_SPECIAL + n_types

        self.rng = np.random.default_rng(seed)

        if self.max_prefix_len < 1:
            raise ValueError("prefix_len must be at least 1")

    @property
    def vocab_size(self) -> int:
        return self.N_SPECIAL + 2 * self.n_types

    @property
    def max_prefix_len(self) -> int:
        if isinstance(self.prefix_len, int):
            return self.prefix_len
        return self.prefix_len[1] - 1

    @property
    def min_block_size(self) -> int:
        """len(prompt) + len(answer) <= (L + 1) + max_depth <= block_size + 1"""
        return self.max_prefix_len + self.max_depth

    def _sample_one(self) -> DatasetItem:
        if isinstance(self.prefix_len, int):
            length = self.prefix_len
        else:
            length = int(self.rng.integers(self.prefix_len[0], self.prefix_len[1]))

        while True:
            prefix = np.empty(length, dtype=np.int64)
            stack: list[int] = []

            for t in range(length):
                can_open = len(stack) < self.max_depth
                can_close = len(stack) > 0

                if can_open and can_close:
                    do_open = self.rng.random() < self.p_open
                else:
                    do_open = can_open

                if do_open:
                    j = int(self.rng.integers(self.n_types))
                    stack.append(j)
                    prefix[t] = self.open_token_ids[j]
                else:
                    j = stack.pop()
                    prefix[t] = self.close_token_ids[j]

            if stack:
                break

        prompt = np.concatenate([prefix, np.array([self.COMPLETE_MARKER_ID])]).astype(
            np.int64
        )
        answer = self.close_token_ids[np.array(stack[::-1])].astype(np.int64)

        return DatasetItem(prompt=prompt, answer=answer)
