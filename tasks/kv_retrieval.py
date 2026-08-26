from typing import Tuple

import numpy as np

from .base import DatasetItem, Task


class KVRetrievalTask(Task):
    PAD_ID = 0
    QUERY_MARKER_ID = 1
    SOD_ID = 2  # start of dict
    EOD_ID = 3  # end of dict
    BOE_ID = 4  # beginning of entry
    EOE_ID = 5  # end of entry
    N_SPECIAL = 6

    def __init__(
        self,
        k_card: int,
        v_card: int,
        n_pairs: int | Tuple[int, int],
        duplicate_keys: bool = False,
        seed: int | None = 42,
    ):
        """
        Generator for KV retrieval task.

        :param k_card: Amount of possible keys.
        :type k_card: int
        :param v_card: Amount of possible values.
        :type v_card: int
        :param n_pairs: Amount of pairs in each prompt. `n_pairs' if the parameter is an
        integer, and a value in [n_pairs[0], n_pairs[1]) if it is a tuple.
        :type n_pairs: int | Tuple[int, int]
        :param duplicate_keys: Whether to allow duplicate keys in the prompt.
        :type duplicate_keys: bool
        :param seed: Randomization seed, None for non-reproducible environment.
        :type seed: int | None

        NOTE(frozen): with duplicate_keys=True the queried key may occur more
        than once and the answer is taken from the *drawn* occurrence, not the
        last one, so ~25% of prompts (507/2000 measured) have an ambiguous
        target. Kept as-is: the published runs were trained on this stream, and
        fixing it changes the data. tests/check_data_equivalence.py, if it is
        still around, will say exactly what moved.
        """
        self.k_card = k_card
        self.v_card = v_card
        self.n_pairs = n_pairs

        self.k_token_ids = np.arange(k_card) + self.N_SPECIAL
        self.v_token_ids = np.arange(v_card) + self.N_SPECIAL + k_card

        self.rng = np.random.default_rng(seed)

        self.duplicate_keys = duplicate_keys

    @property
    def vocab_size(self) -> int:
        return self.N_SPECIAL + self.k_card + self.v_card

    def _sample_one(self) -> DatasetItem:
        if isinstance(self.n_pairs, int):
            n_pairs = self.n_pairs
        else:
            n_pairs = int(self.rng.integers(self.n_pairs[0], self.n_pairs[1]))
        keys = self.rng.choice(
            self.k_token_ids, size=n_pairs, replace=self.duplicate_keys
        )
        values = self.rng.choice(self.v_token_ids, size=n_pairs, replace=True)

        # Each entry: BOE key value EOE  -> 4 tokens per pair.
        # Dict: SOD <entries> EOD.
        # Tail: query separator + query.
        seq = np.empty(4 * n_pairs + 2 + 2, dtype=np.int64)

        seq[0] = self.SOD_ID
        entries = seq[1 : 1 + 4 * n_pairs]
        entries[0::4] = self.BOE_ID
        entries[1::4] = keys
        entries[2::4] = values
        entries[3::4] = self.EOE_ID
        seq[1 + 4 * n_pairs] = self.EOD_ID

        seq[-2] = self.QUERY_MARKER_ID

        qi = int(self.rng.integers(n_pairs))
        seq[-1] = keys[qi]
        answer = np.array([values[qi]], dtype=np.int64)

        # axes for position-resolved analysis; metadata never touches the rng,
        # so recording it leaves the generated stream unchanged
        metadata = {
            "sequence_length": len(seq),
            "n_pairs": n_pairs,
            "target_entry": qi,
            "absolute_target_position": 1 + 4 * qi,  # the entry's BOE token
            "normalized_target_position": qi / max(n_pairs - 1, 1),
        }
        return DatasetItem(prompt=seq, answer=answer, metadata=metadata)
