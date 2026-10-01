"""Query Count (Yehudai et al., 2024): count the final token in the body.

    3 7 1 3 5 0 2 3 6 4 1 7  3   ->   3      (n = 12, m = 8)

The body contains ``seq_len`` uniform samples from ``n_symbols``. A random
query is forced into the body and appended last. The target is its count in
the body, encoded as ``CLASS_LOW + count``. The paper's formula 3 also counts
the appended query, so its target is one larger.

The task is hard because any symbol can be queried and every position matters
(Section 4). Uniform attention forms a histogram when ``d >= m``; otherwise
selective attention produces roughly ``1/count``, whose inversion needs MLP
width proportional to ``n``. Expect a transition near ``d = m`` and poor
generalization beyond the training length.

Their metric is ``nmae = |prediction - count| / (n/m + 1)``. For large ``m``,
the median-count baseline is already strong (``acc=0.825``, ``nmae=0.161`` at
``m=512, n=100``), so compare against it rather than the non-integer ``n/m``
baseline. This task replaces the earlier fixed-marker version, which one
attention head solved at any length.
"""

import numpy as np

from .base import DatasetItem, Task, validate_int_spec


class SelectiveCountTask(Task):
    PAD_ID = 0
    SYMBOL_LOW = 1  

    def __init__(self, n_symbols: int = 32, seq_len: int = 50,
                 max_count: int | None = None, seed: int | None = 42):
        """
        :param n_symbols: vocabulary size m of the body.
        :param seq_len: body length n; one int, since nmae is normalized by n/m.
        :param max_count: largest count token; None means seq_len. Set it above
            seq_len to evaluate on longer sequences later.
        """
        super().__init__(seed)
        self.n_symbols = validate_int_spec(n_symbols, "n_symbols", 2)
        self.seq_len = validate_int_spec(seq_len, "seq_len", 1)
        self.max_count = self.seq_len if max_count is None else validate_int_spec(max_count, "max_count", 1)
        if not all(isinstance(v, int) for v in (self.n_symbols, self.seq_len, self.max_count)):
            raise TypeError("n_symbols, seq_len and max_count must be single integers")
        if self.max_count < self.seq_len:
            raise ValueError(f"max_count={self.max_count} cannot hold a count of up to "
                             f"seq_len={self.seq_len}")
        self.CLASS_LOW = self.SYMBOL_LOW + self.n_symbols  

    @property
    def vocab_size(self) -> int:
        return self.CLASS_LOW + self.max_count + 1

    def _sample_one(self) -> DatasetItem:
        body = self.rng.integers(0, self.n_symbols, size=self.seq_len)
        query = int(self.rng.integers(self.n_symbols))
        body[self.rng.integers(self.seq_len)] = query 
        count = int((body == query).sum())
        prompt = self.SYMBOL_LOW + np.append(body, query).astype(np.int64)
        return DatasetItem(prompt, np.array([self.CLASS_LOW + count], dtype=np.int64),
                           metadata={"query": query, "count": count,
                                     "seq_len": self.seq_len, "n_symbols": self.n_symbols})

    def metrics(self, predicted, targets) -> dict[str, float]:
        scores = super().metrics(predicted, targets)
        answer = targets != -1
        true = targets[answer] - self.CLASS_LOW
        guess = np.clip(predicted[answer] - self.CLASS_LOW, 0, self.max_count)
        expected = self.seq_len / self.n_symbols + 1
        scores["nmae"] = float(np.abs(guess - true).mean() / expected)
        return scores
