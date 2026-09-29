"""S5/C5 word problem: after every move, name the composed permutation so far.

    x:  p1  p2  p3  ...  pK          moves, one token each
    y:  s1  s2  s3  ...  sK          s_i = p_1 o p_2 o ... o p_i, the same token ids

Every position carries a label, so one example contains the tasks of length
1..K at once; s_1 is a copy of p_1. This is the token-tagging format of
Merrill et al. 2024 (jopetty/word-problem, generate_data.py) and Li et al.
2025 (belindal/state-tracking): the model reads only the moves and must keep
the running product itself. Composition follows both of them: (a o b)[j] =
a[b[j]] and each new factor is multiplied on the right, s_i = s_{i-1} o p_i,
so labels are comparable with their data element for element. The base metrics map onto theirs -- ``token_acc``
is their token accuracy, ``acc`` their sequence accuracy.

S5 draws from all 120 permutations of five elements; chance is 1/120. C5 draws
from the five powers of one 5-cycle, so composition reduces to addition mod 5;
chance is 1/5. Both share the vocabulary and the sequence length.
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
    PERM_LOW = 1  # PERM_LOW + i is permutation i of ALL_PERMUTATIONS

    def __init__(
        self,
        group: str,
        n_permutations: int | tuple[int, int] = (4, 33),
        seed: int | None = 42,
    ):
        """
        :param group: 'c5' or 's5'.
        :param n_permutations: moves per example, an int or [lo, hi); needs
            block_size >= the largest value.
        """
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
        return self.PERM_LOW + len(ALL_PERMUTATIONS)

    def _sample_one(self) -> DatasetItem:
        length = sample_int(self.rng, self.n_permutations)
        factors = self.rng.choice(self.allowed, size=length, replace=True)

        state = tuple(range(N_ELEMENTS))  # the identity: nothing composed yet
        products = np.empty(length, dtype=np.int64)
        for i, factor in enumerate(factors):
            move = ALL_PERMUTATIONS[int(factor)]
            state = tuple(state[int(m)] for m in move)  # s_{i-1} o p_i
            products[i] = PERMUTATION_INDEX[state]

        prompt = self.PERM_LOW + factors
        labels = self.PERM_LOW + products
        return DatasetItem(
            prompt,
            labels[-1:],  # DatasetItem needs an answer; collate uses labels instead
            metadata={"group": self.group, "n_permutations": length},
            labels=labels,
        )
