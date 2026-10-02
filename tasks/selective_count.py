"""Original Query Count task from Yehudai et al. (2024).

    3 7 1 3 5 0 3 7   ->   2

The input is a length-``n`` sequence sampled uniformly with replacement from
``m`` symbols. Its last symbol is the query; the single target is that symbol's
count in the whole sequence, itself included. Following the original
experiments, ``n = count_ratio * m`` by default (``count_ratio=10``), which
keeps the expected count approximately constant while ``m`` changes.

The answer is encoded as ``CLASS_LOW + count``. ``acc`` and ``token_acc`` are
identical because there is one target. ``nmae`` additionally follows the later
paper revision: ``|prediction - count| / (n/m + 1)``.
"""

import numpy as np

from .base import DatasetItem, Task, validate_int_spec


class SelectiveCountTask(Task):
    PAD_ID = 0
    SYMBOL_LOW = 1

    def __init__(self, n_symbols: int = 32, count_ratio: int = 10,
                 seq_len: int | None = None, max_count: int | None = None,
                 seed: int | None = 42):
        """
        :param n_symbols: vocabulary size m of the input.
        :param count_ratio: default ratio n/m used when seq_len is omitted.
        :param seq_len: optional explicit input length n, useful for OOD tests.
        :param max_count: largest count token; None means seq_len. Set it above
            seq_len to evaluate on longer sequences later.
        """
        super().__init__(seed)
        self.n_symbols = validate_int_spec(n_symbols, "n_symbols", 2)
        self.count_ratio = validate_int_spec(count_ratio, "count_ratio", 1)
        if not all(isinstance(v, int) for v in (self.n_symbols, self.count_ratio)):
            raise TypeError("n_symbols and count_ratio must be single integers")
        self.seq_len = (
            self.count_ratio * self.n_symbols
            if seq_len is None
            else validate_int_spec(seq_len, "seq_len", 1)
        )
        self.max_count = (
            self.seq_len
            if max_count is None
            else validate_int_spec(max_count, "max_count", 1)
        )
        if not all(isinstance(v, int) for v in (self.seq_len, self.max_count)):
            raise TypeError("seq_len and max_count must be single integers")
        if self.max_count < self.seq_len:
            raise ValueError(f"max_count={self.max_count} cannot hold a count of up to "
                             f"seq_len={self.seq_len}")
        self.CLASS_LOW = self.SYMBOL_LOW + self.n_symbols

    @property
    def vocab_size(self) -> int:
        return self.CLASS_LOW + self.max_count + 1

    def _sample_one(self) -> DatasetItem:
        symbols = self.rng.integers(
            0, self.n_symbols, size=self.seq_len, dtype=np.int64
        )
        query = int(symbols[-1])
        count = int(np.count_nonzero(symbols == query))
        return DatasetItem(
            self.SYMBOL_LOW + symbols,
            np.array([self.CLASS_LOW + count], dtype=np.int64),
            metadata={
                "query_symbol": query,
                "final_count": count,
                "seq_len": self.seq_len,
                "n_symbols": self.n_symbols,
            },
        )

    def metrics(self, predicted, targets) -> dict[str, float]:
        scores = super().metrics(predicted, targets)
        answer = targets != -1
        true = targets[answer] - self.CLASS_LOW
        guess = np.clip(predicted[answer] - self.CLASS_LOW, 0, self.max_count)
        expected = self.seq_len / self.n_symbols + 1
        scores["nmae"] = float(np.abs(guess - true).mean() / expected)
        return scores
