from typing import Tuple

import numpy as np

from .base import DatasetItem, Generator


class IndexingGenerator(Generator):
    PAD_ID = 0
    QUERY_MARKER_ID = 1
    N_SPECIAL = 2

    def __init__(
        self,
        v_card: int,
        n_items: int | Tuple[int, int],
        max_items: int | None = None,
        seed: int | None = 42,
    ):
        """
        Generator for the indexing task: retrieve the value at a queried position.

        Prompt layout:  v_0 .. v_{n-1}  QUERY  idx
        Answer layout:  v_idx

        The index is emitted as its own token from a dedicated block of the
        vocabulary, so positional access is a content lookup over index symbols
        rather than something the model can only read off the positional encoding.

        :param v_card: Amount of possible values.
        :type v_card: int
        :param n_items: Length of the list. `n_items' if the parameter is an
            integer, and a value in [n_items[0], n_items[1]) if it is a tuple.
        :type n_items: int | Tuple[int, int]
        :param max_items: Size of the index token block, i.e. the largest list
            length this vocabulary can address. Defaults to the largest length
            `n_items' can produce. Set it explicitly and identically across a
            length-OOD sweep so that vocab_size stays fixed and one trained model
            can be evaluated at every length.
        :type max_items: int | None
        :param seed: Randomization seed, None for non-reproducible environment.
        :type seed: int | None
        """
        self.v_card = v_card
        self.n_items = n_items

        n_max = n_items if isinstance(n_items, int) else n_items[1] - 1
        self.max_items = n_max if max_items is None else max_items
        if self.max_items < n_max:
            raise ValueError(
                f"max_items={self.max_items} cannot address lists of length {n_max}"
            )

        self.v_token_ids = np.arange(v_card) + self.N_SPECIAL
        self.i_token_ids = np.arange(self.max_items) + self.N_SPECIAL + v_card

        self.rng = np.random.default_rng(seed)

    @property
    def vocab_size(self) -> int:
        return self.N_SPECIAL + self.v_card + self.max_items

    @property
    def min_block_size(self) -> int:
        """len(prompt) + len(answer) = (n + 2) + 1 <= block_size + 1"""
        n_max = self.n_items if isinstance(self.n_items, int) else self.n_items[1] - 1
        return n_max + 2

    def _sample_one(self) -> DatasetItem:
        if isinstance(self.n_items, int):
            n = self.n_items
        else:
            n = int(self.rng.integers(self.n_items[0], self.n_items[1]))

        values = self.rng.choice(self.v_token_ids, size=n, replace=True)

        qi = int(self.rng.integers(n))

        prompt = np.concatenate(
            [
                values,
                np.array([self.QUERY_MARKER_ID]),
                np.array([self.i_token_ids[qi]]),
            ]
        ).astype(np.int64)
        answer = np.array([values[qi]], dtype=np.int64)

        return DatasetItem(prompt=prompt, answer=answer)
