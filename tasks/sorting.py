from typing import Tuple

import numpy as np

from .base import DatasetItem, Task


class SortingTask(Task):
    PAD_ID = 0
    SORT_MARKER_ID = 1
    N_SPECIAL = 2

    def __init__(
        self,
        v_card: int,
        n_items: int | Tuple[int, int],
        duplicates: bool = True,
        descending: bool = False,
        sample_v_card: int | None = None,
        seed: int | None = 42,
    ):
        """
        Generator for the sorting task.

        Prompt layout:  v_0 .. v_{n-1}  SORT
        Answer layout:  sorted(v_0 .. v_{n-1})

        Token ids are assigned in increasing order of value, so sorting token ids
        is equivalent to sorting values and no decoding step is needed.

        :param v_card: Total number of value tokens allocated in the model
            vocabulary. This fixes ``vocab_size = N_SPECIAL + v_card``.
        :type v_card: int
        :param n_items: Amount of items to sort. `n_items' if the parameter is an
            integer, and a value in [n_items[0], n_items[1]) if it is a tuple.
        :type n_items: int | Tuple[int, int]
        :param duplicates: Whether the same value may appear more than once. False
            makes every item distinct (requires n_items <= sample_v_card) and
            removes tie-breaking from the task.
        :type duplicates: bool
        :param descending: Sort order of the answer.
        :type descending: bool
        :param sample_v_card: Number of the lowest value tokens that may actually
            be sampled. ``None`` (default) means all ``v_card`` values. Keeping
            ``v_card=30`` while training with ``sample_v_card=20`` allocates a
            30-value model vocabulary but exposes only values 0..19 in training;
            evaluation can then set ``sample_v_card=30`` without an embedding
            index overflow.
        :type sample_v_card: int | None
        :param seed: Randomization seed, None for non-reproducible environment.
        :type seed: int | None
        """
        self.v_card = v_card
        self.n_items = n_items
        self.duplicates = duplicates
        self.descending = descending
        self.sample_v_card = v_card if sample_v_card is None else sample_v_card

        if not 1 <= self.sample_v_card <= self.v_card:
            raise ValueError(
                f"sample_v_card must be in [1, v_card], got "
                f"sample_v_card={self.sample_v_card}, v_card={self.v_card}"
            )

        self.v_token_ids = np.arange(self.sample_v_card) + self.N_SPECIAL

        self.rng = np.random.default_rng(seed)

        if not duplicates and self.max_items > self.sample_v_card:
            raise ValueError(
                f"duplicates=False requires n_items <= sample_v_card, got max "
                f"n_items={self.max_items} and sample_v_card={self.sample_v_card}"
            )

    @property
    def vocab_size(self) -> int:
        return self.N_SPECIAL + self.v_card

    @property
    def max_items(self) -> int:
        if isinstance(self.n_items, int):
            return self.n_items
        return self.n_items[1] - 1

    @property
    def min_block_size(self) -> int:
        """len(prompt) + len(answer) = (n + 1) + n = 2n + 1 <= block_size + 1"""
        return 2 * self.max_items

    def _sample_one(self) -> DatasetItem:
        if isinstance(self.n_items, int):
            n = self.n_items
        else:
            n = int(self.rng.integers(self.n_items[0], self.n_items[1]))

        values = self.rng.choice(self.v_token_ids, size=n, replace=self.duplicates)

        answer = np.sort(values)
        if self.descending:
            answer = answer[::-1]

        prompt = np.concatenate([values, np.array([self.SORT_MARKER_ID])]).astype(
            np.int64
        )

        return DatasetItem(prompt=prompt, answer=answer.astype(np.int64))
