"""Nested key-value retrieval: follow a key path and return its value.

    depth=1:  { a 7, b 4 }                 ? b       -> 4
    depth=2:  { a 7, b { c 9, d 2 } }      ? b c     -> 9

Every dictionary has ``n_pairs`` entries and exactly one entry continues the
queried path until ``depth`` is reached. Other entries are distractors. Keys are
unique within a dictionary, so every query has one unambiguous answer.
"""

import numpy as np

from .base import DatasetItem, Task, max_int, sample_int, validate_int_spec


class KVRetrievalTask(Task):
    PAD_ID = 0
    QUERY_ID = 1
    DICT_START_ID = 2
    DICT_END_ID = 3
    ENTRY_START_ID = 4
    ENTRY_END_ID = 5
    N_SPECIAL = 6

    def __init__(
        self,
        k_card: int = 64,
        v_card: int = 64,
        n_pairs: int | tuple[int, int] = (4, 9),
        depth: int = 1,
        seed: int | None = 42,
    ):
        super().__init__(seed)
        self.k_card = validate_int_spec(k_card, "k_card", 1)
        self.v_card = validate_int_spec(v_card, "v_card", 1)
        self.n_pairs = validate_int_spec(n_pairs, "n_pairs", 1)
        self.depth = validate_int_spec(depth, "depth", 1)
        if max_int(self.n_pairs) > self.k_card:
            raise ValueError("n_pairs cannot exceed k_card: keys must be unique in each dict")

        self.key_ids = np.arange(self.k_card, dtype=np.int64) + self.N_SPECIAL
        self.value_ids = (
            np.arange(self.v_card, dtype=np.int64) + self.N_SPECIAL + self.k_card
        )

    @property
    def vocab_size(self) -> int:
        return self.N_SPECIAL + self.k_card + self.v_card

    def _build_dictionary(self, level: int):
        n_pairs = sample_int(self.rng, self.n_pairs)
        keys = self.rng.choice(
            self.key_ids, size=n_pairs, replace=False
        )
        target_slot = int(self.rng.integers(n_pairs))
        tokens = [self.DICT_START_ID]
        target_position = None

        for slot, key_token in enumerate(keys):
            entry_position = len(tokens)
            tokens += [self.ENTRY_START_ID, int(key_token)]
            if slot == target_slot and level < self.depth:
                child, child_path, answer, child_position, sizes = self._build_dictionary(level + 1)
                tokens += child
                path = [int(key_token), *child_path]
                target_position = entry_position + 2 + child_position
                level_sizes = [n_pairs, *sizes]
            else:
                value = int(self.rng.choice(self.value_ids))
                tokens.append(value)
                if slot == target_slot:
                    path = [int(key_token)]
                    answer = value
                    target_position = entry_position
                    level_sizes = [n_pairs]
            tokens.append(self.ENTRY_END_ID)

        tokens.append(self.DICT_END_ID)
        return tokens, path, answer, target_position, level_sizes

    def _sample_one(self) -> DatasetItem:
        dictionary, path, value, target_position, level_sizes = self._build_dictionary(1)
        prompt = np.asarray(dictionary + [self.QUERY_ID, *path], dtype=np.int64)
        return DatasetItem(
            prompt=prompt,
            answer=np.asarray([value], dtype=np.int64),
            metadata={
                "depth": self.depth,
                "n_pairs_by_level": level_sizes,
                "sequence_length": len(prompt),
                "absolute_target_position": target_position,
                "relative_distance": len(prompt) - 1 - target_position,
            },
        )
