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

        return DatasetItem(
            prompt=prompt,
            answer=answer.astype(np.int64),
            metadata={"n_items": n, "inversions": _inversions(values)},
        )


def _inversions(values) -> int:
    """Pairs that are out of order — the difficulty L(x) of a sort.

    Counted on token ids, which is the same as counting on values: ids are
    assigned in increasing order of value.
    """
    return int(sum(values[i] > values[j]
                   for i in range(len(values)) for j in range(i + 1, len(values))))



class StringSortingTask(Task):
    PAD_ID = 0
    SORT_ID = 1
    SEP_ID = 2
    EOS_ID = 3
    N_SPECIAL = 4

    def __init__(
        self,
        n_items: int | Tuple[int, int],
        item_len: int | Tuple[int, int] = (2, 5),
        alphabet: str = "0123456789abcdefghijklmnopqrstuvwxyz",
        duplicates: bool = True,
        descending: bool = False,
        seed: int | None = 42,
    ):
        self.n_items = n_items
        self.item_len = item_len
        self.duplicates = duplicates
        self.descending = descending
        self.rng = np.random.default_rng(seed)

        self.char_to_id = {
            char: idx + self.N_SPECIAL
            for idx, char in enumerate(alphabet)
        }
        self.id_to_char = {
            idx: char
            for char, idx in self.char_to_id.items()
        }

    @property
    def vocab_size(self) -> int:
        return self.N_SPECIAL + len(self.char_to_id)

    def _sample_n_items(self) -> int:
        if isinstance(self.n_items, int):
            return self.n_items
        return int(self.rng.integers(*self.n_items))

    def _sample_item_len(self) -> int:
        if isinstance(self.item_len, int):
            return self.item_len
        return int(self.rng.integers(*self.item_len))

    def _sample_string(self) -> str:
        length = self._sample_item_len()
        chars = self.rng.choice(list(self.char_to_id), size=length)
        return "".join(chars)

    def _encode_items(self, items: list[str]) -> np.ndarray:
        encoded = []

        for i, item in enumerate(items):
            encoded.extend(self.char_to_id[ch] for ch in item)

            if i < len(items) - 1:
                encoded.append(self.SEP_ID)

        return np.asarray(encoded, dtype=np.int64)

    def _sample_one(self) -> DatasetItem:
        n = self._sample_n_items()

        items = []
        seen = set()

        while len(items) < n:
            item = self._sample_string()

            if self.duplicates or item not in seen:
                items.append(item)
                seen.add(item)

        sorted_items = sorted(items, reverse=self.descending)

        prompt = np.concatenate([
            self._encode_items(items),
            np.asarray([self.SORT_ID], dtype=np.int64),
        ])

        answer = np.concatenate([
            self._encode_items(sorted_items),
            np.asarray([self.EOS_ID], dtype=np.int64),
        ])

        return DatasetItem(
            prompt=prompt,
            answer=answer,
            metadata={
                "n_items": n,
                "items": items,
                "sorted_items": sorted_items,
            },
        )

    def metrics(self, predicted, targets) -> dict:
        """Return batch-level metrics: exact-match `acc` and token-level `token_acc`.

        `predicted` and `targets` are numpy arrays shaped [batch, block_size].
        Positions where `targets == -1` are ignored for `token_acc`.
        """
        base = super().metrics(predicted, targets)
        mask = targets != -1
        total = mask.sum()
        if total:
            correct = ((predicted == targets) & mask).sum()
            token_acc = float(correct) / float(total)
        else:
            token_acc = 0.0
        base.update({"token_acc": float(token_acc)})
        return base
