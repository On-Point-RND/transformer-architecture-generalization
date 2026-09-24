"""String sorting: put a list of short strings in lexicographic order.

    m i 9 b | 2 0 | t n | i l y q  SORT   ->   2 0 | i l y q | m i 9 b | t n  EOS

Each string is spelled one character per token, items are separated by SEP and
the answer ends in EOS. The order is Python's string order, so a string's rank
is decided by comparing several positions left to right, and both prompt and
answer lengths vary with n_items and item_len.
"""

from typing import Tuple

import numpy as np

from .base import DatasetItem, Task, max_int, sample_int, validate_int_spec


class StringSortingTask(Task):
    PAD_ID = 0
    SORT_ID = 1
    SEP_ID = 2
    EOS_ID = 3
    N_SPECIAL = 4

    def __init__(
        self,
        n_items: int | Tuple[int, int] = (4, 9),
        item_len: int | Tuple[int, int] = (2, 5),
        alphabet: str = "0123456789abcdefghijklmnopqrstuvwxyz",
        duplicates: bool = True,
        descending: bool = False,
        seed: int | None = 42,
    ):
        super().__init__(seed)
        self.n_items = validate_int_spec(n_items, "n_items", 1)
        self.item_len = validate_int_spec(item_len, "item_len", 1)
        self.duplicates = duplicates
        self.descending = descending

        if not alphabet or len(set(alphabet)) != len(alphabet):
            raise ValueError("alphabet must be non-empty and contain unique characters")

        self.char_to_id = {
            char: idx + self.N_SPECIAL
            for idx, char in enumerate(alphabet)
        }
        self.id_to_char = {
            idx: char
            for char, idx in self.char_to_id.items()
        }
        if not duplicates:
            lo = self.item_len if isinstance(self.item_len, int) else self.item_len[0]
            lengths = range(lo, max_int(self.item_len) + 1)
            capacity = sum(len(alphabet) ** length for length in lengths)
            if max_int(self.n_items) > capacity:
                raise ValueError(
                    f"duplicates=False needs {max_int(self.n_items)} distinct strings, "
                    f"but item_len/alphabet provide only {capacity}"
                )

    @property
    def vocab_size(self) -> int:
        return self.N_SPECIAL + len(self.char_to_id)

    def _sample_string(self) -> str:
        length = sample_int(self.rng, self.item_len)
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
        n = sample_int(self.rng, self.n_items)

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
