"""C5/S5 state tracking by sequential permutation composition.

    BOS  p1 p2 ... pk  QUERY x   ->   (pk o ... o p2 o p1)(x)

S5 samples all 120 permutations of five elements. C5 samples the five powers
of one 5-cycle. The token layout and answer space are identical; only the
allowed operations differ. S5 is non-commutative, while C5 is the matched
abelian control whose composition reduces to addition modulo five.
"""

from itertools import permutations

import numpy as np

from .base import DatasetItem, Task, sample_int, validate_int_spec

N_ELEMENTS = 5
ALL_PERMUTATIONS = np.asarray(list(permutations(range(N_ELEMENTS))), dtype=np.int64)
PERMUTATION_INDEX = {tuple(p): i for i, p in enumerate(ALL_PERMUTATIONS)}
CYCLIC_PERMUTATIONS = np.asarray(
    [PERMUTATION_INDEX[tuple((np.arange(N_ELEMENTS) + shift) % N_ELEMENTS)]
     for shift in range(N_ELEMENTS)],
    dtype=np.int64,
)


class PermutationTask(Task):
    PAD_ID = 0
    BOS_ID = 1
    QUERY_ID = 2
    PERM_LOW = 3
    POINT_LOW = PERM_LOW + len(ALL_PERMUTATIONS)

    def __init__(
        self,
        group: str,
        n_permutations: int | tuple[int, int] = (4, 33),
        seed: int | None = 42,
    ):
        super().__init__(seed)
        group = group.lower()
        if group not in ("c5", "s5"):
            raise ValueError(f"group must be 'c5' or 's5', got {group!r}")
        self.group = group
        self.n_permutations = validate_int_spec(n_permutations, "n_permutations", 1)
        self.allowed = (
            CYCLIC_PERMUTATIONS
            if group == "c5"
            else np.arange(len(ALL_PERMUTATIONS), dtype=np.int64)
        )

    @property
    def vocab_size(self) -> int:
        return self.POINT_LOW + N_ELEMENTS

    def _sample_one(self) -> DatasetItem:
        length = sample_int(self.rng, self.n_permutations)
        factors = self.rng.choice(
            self.allowed, size=length, replace=True
        )
        query = int(self.rng.integers(N_ELEMENTS))
        result = query
        for factor in factors:
            result = int(ALL_PERMUTATIONS[int(factor), result])

        prompt = np.asarray(
            [self.BOS_ID]
            + [self.PERM_LOW + int(factor) for factor in factors]
            + [self.QUERY_ID, self.POINT_LOW + query],
            dtype=np.int64,
        )
        return DatasetItem(
            prompt,
            np.asarray([self.POINT_LOW + result], dtype=np.int64),
            metadata={
                "group": self.group,
                "n_permutations": length,
                "non_identity": int(np.count_nonzero(factors)),
                "query": query,
                "result": result,
            },
        )
