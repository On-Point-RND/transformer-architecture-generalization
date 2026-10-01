"""Multi-query associative recall (MQAR) from Zoology (Arora et al., 2023).

    x:  k1 v1 k2 v2 ... kK vK   r  r  k2  r  r  r  r  k1  r ...
    y:   -  -  -  -  ...  -  -   -  -  v2  -  -  -  -  v1  - ...

The first 2K tokens are K key-value pairs: distinct keys from the lower half
of the vocabulary (token 0 excluded) and distinct values from the upper half.
Every key then appears once more, at an even offset whose distance follows a
power law (``power_a``); the label at that position is the key's value, all
other positions are unsupervised. The rest are uniformly random tokens from
the whole vocabulary. This is zoology/data/multiquery_ar.py with its defaults
(``num_passes=1``, ``random_non_queries=True``): the same token ids, labels
and lengths, drawn from our own random stream.

``token_acc`` is Zoology's accuracy, the share of values recalled (with a
fixed K the per-example and pooled means coincide); ``acc`` needs every query
of an example right. Needs ``block_size >= input_seq_len``.
"""

import numpy as np

from .base import DatasetItem, Task


class KVRetrievalTask(Task):
    PAD_ID = 0  

    def __init__(
        self,
        vocab_size: int = 8192,
        input_seq_len: int = 64,
        num_kv_pairs: int = 8,
        power_a: float = 0.01,
        seed: int | None = 42,
    ):
        super().__init__(seed)
        if input_seq_len % 2:
            raise ValueError(f"input_seq_len must be even, got {input_seq_len}")
        if vocab_size <= input_seq_len:
            raise ValueError("vocab_size must exceed input_seq_len")
        if 4 * num_kv_pairs > input_seq_len:
            raise ValueError("input_seq_len must be at least 4 * num_kv_pairs "
                             "(K pairs plus room for K distinct query slots)")
        self.n_vocab = vocab_size
        self.input_seq_len = input_seq_len
        self.num_kv_pairs = num_kv_pairs
        self.key_choices = np.arange(1, vocab_size // 2)
        self.value_choices = np.arange(vocab_size // 2, vocab_size)

        space = (input_seq_len - 2 * num_kv_pairs) // 2
        weights = power_a * np.arange(1, space + 1) ** (power_a - 1)
        self.gap_probs = weights / weights.sum()

    @property
    def vocab_size(self) -> int:
        return self.n_vocab

    def _sample_one(self) -> DatasetItem:
        n_pairs, context = self.num_kv_pairs, 2 * self.num_kv_pairs
        keys = self.rng.choice(self.key_choices, size=n_pairs, replace=False)
        values = self.rng.choice(self.value_choices, size=n_pairs, replace=False)
        gaps = self.rng.choice(len(self.gap_probs), size=n_pairs, replace=False, p=self.gap_probs)

        prompt = self.rng.integers(0, self.n_vocab, size=self.input_seq_len)  # the random non-queries
        prompt[0:context:2] = keys
        prompt[1:context:2] = values
        query_positions = context + 2 * gaps
        prompt[query_positions] = keys
        labels = np.full(self.input_seq_len, -1, dtype=np.int64)
        labels[query_positions] = values

        return DatasetItem(
            prompt,
            values[np.argsort(query_positions)],  # the answers in query order; collate uses labels
            metadata={"num_kv_pairs": n_pairs, "input_seq_len": self.input_seq_len},
            labels=labels,
        )
