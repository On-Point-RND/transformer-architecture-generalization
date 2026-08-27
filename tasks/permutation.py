from itertools import permutations
from typing import Tuple

import numpy as np

from .base import DatasetItem, Task

NELEM = 5  # permutations act on {0, 1, 2, 3, 4}
ALL_PERMS = sorted(permutations(range(NELEM)))  # 120, in a fixed order
IDENTITY = tuple(range(NELEM))
CYCLE = tuple((i + 1) % NELEM for i in range(NELEM))  # the 5-cycle (0 1 2 3 4)


def compose(state, p):
    """Apply `state` first, then `p`: result[i] = p[state[i]]."""
    return tuple(p[state[i]] for i in range(NELEM))


def compose_all(perms):
    state = IDENTITY
    for p in perms:
        state = compose(state, p)
    return state


def cyclic_group():
    """<(0 1 2 3 4)> — 5 elements including the identity."""
    out, current = [], IDENTITY
    for _ in range(NELEM):
        out.append(current)
        current = compose(current, CYCLE)
    return out


GROUPS = {"S5": ALL_PERMS, "C5": cyclic_group()}


class PermutationTask(Task):
    """Compose K permutations of {0..4} and write down where each element lands.

    Prompt layout:  BOS  p_1 .. p_K  EQ
    Answer layout:  d_0 .. d_4  EOS      (the composed permutation, then EOS)

    Two variants share the vocabulary, the sequence length and the answer format
    and differ only in which group the factors come from:

    S5 — the full symmetric group. It is non-solvable, so its word problem is
         NC^1-complete (Barrington) and cannot be collapsed to constant depth.
         This is the sequential task the depth study is built around.
    C5 — the cyclic subgroup. Cyclic implies abelian, so composing reduces to
         adding exponents mod 5, which a prefix scan does in parallel. The
         control: same surface form, no sequential requirement.

    Difficulty L(x) is the number of non-identity factors — identity elements
    are no-ops, so they cost no sequential step.
    """

    PAD_ID = 0
    BOS_ID = 1
    EOS_ID = 2
    EQ_ID = 3
    PERM_OFFSET = 4  # permutation tokens occupy PERM_OFFSET .. PERM_OFFSET+119
    DIGIT_OFFSET = PERM_OFFSET + len(ALL_PERMS)  # answer digits 0..4

    def __init__(
        self,
        n_perms: int | Tuple[int, int],
        variant: str = "S5",
        seed: int | None = 42,
    ):
        """
        :param n_perms: Factors per example (K). Fixed if an integer, sampled
            from [n_perms[0], n_perms[1]) if a tuple.
        :param variant: 'S5' (sequential) or 'C5' (parallel control).
        :param seed: Randomization seed, None for a non-reproducible environment.
        """
        if variant not in GROUPS:
            raise ValueError(f"variant must be one of {sorted(GROUPS)}, got {variant!r}")
        self.n_perms = n_perms
        self.variant = variant
        self.group = GROUPS[variant]
        self.non_identity = [p for p in self.group if p != IDENTITY]
        self.perm_ids = {p: i for i, p in enumerate(ALL_PERMS)}
        self.rng = np.random.default_rng(seed)

    @property
    def vocab_size(self) -> int:
        return self.DIGIT_OFFSET + NELEM  # 129

    @property
    def max_perms(self) -> int:
        if isinstance(self.n_perms, int):
            return self.n_perms
        return self.n_perms[1] - 1

    @property
    def min_block_size(self) -> int:
        """len(prompt) + len(answer) = (K + 2) + 6 = K + 8 <= block_size + 1"""
        return self.max_perms + 7

    def _with_chain_length(self, k: int, chain: int):
        """K factors of which exactly `chain` are non-identity, at random spots."""
        perms = [IDENTITY] * k
        for i in self.rng.choice(k, size=chain, replace=False) if chain else ():
            perms[int(i)] = self.non_identity[int(self.rng.integers(len(self.non_identity)))]
        return perms

    def _sample_one(self) -> DatasetItem:
        if isinstance(self.n_perms, int):
            k = self.n_perms
        else:
            k = int(self.rng.integers(self.n_perms[0], self.n_perms[1]))

        chain = int(self.rng.integers(0, k + 1))
        perms = self._with_chain_length(k, chain)
        result = compose_all(perms)

        prompt = np.array(
            [self.BOS_ID] + [self.PERM_OFFSET + self.perm_ids[p] for p in perms]
            + [self.EQ_ID], dtype=np.int64
        )
        answer = np.array(
            [self.DIGIT_OFFSET + v for v in result] + [self.EOS_ID], dtype=np.int64
        )
        return DatasetItem(
            prompt=prompt,
            answer=answer,
            metadata={"variant": self.variant, "n_perms": k, "chain_length": chain},
        )
