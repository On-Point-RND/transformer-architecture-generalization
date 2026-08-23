"""Concrete Task wrappers around the per-task generation modules.

These adapt the (proven) module-level functions in addition.py / permutation.py /
sorting.py to the common ``Task`` interface, so the training loop, sweep runner
and probes can stay task-agnostic. The generation logic itself is untouched.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from datasets import addition as _add
from datasets import permutation as _perm
from datasets import sorting as _sort
from datasets.base import Example, Task


class AdditionTask(Task):
    name = "addition"
    variants = ("SEQ", "INDEP")           # integer sum vs per-digit sum (no carry)
    difficulty_label = "carry-chain length L(x)"
    difficulty_col = "chain_length"
    by_suffix = "by_chain"
    ignore_index = _add.IGNORE_INDEX

    @property
    def vocab_size(self) -> int:
        return _add.VOCAB_SIZE

    def seq_len(self, N: int) -> int:
        return _add.seq_len_for(N)

    def make_splits(self, N, n_train, n_test, seed, variant="SEQ"):
        return _add.make_splits(variant, N, n_train, n_test, seed)

    def answer_span(self, N: int) -> Tuple[int, int]:
        return (2 * N + 3, N + 1)          # [BOS] a_r + b_r = c_r(N+1) [EOS]

    def difficulty(self, input_ids, N: int) -> int:
        a, b = _add._operands_from_ids(input_ids, N)
        return _add.carry_chain_length(a, b)

    def sample_balanced(self, N, per_bin, seed, variant="SEQ"):
        return _add.sample_balanced_by_chain(N, per_bin, seed, task=variant)


class PermutationTask(Task):
    name = "permutation"
    variants = ("S5", "C5")               # non-solvable (sequential) vs abelian control
    difficulty_label = "chain length L(x) = #non-identity"
    difficulty_col = "chain_length"
    by_suffix = "by_chain"
    ignore_index = _perm.IGNORE_INDEX

    @property
    def vocab_size(self) -> int:
        return _perm.VOCAB_SIZE

    def seq_len(self, K: int) -> int:
        return _perm.seq_len_for(K)

    def make_splits(self, K, n_train, n_test, seed, variant="S5"):
        return _perm.make_splits(variant, K, n_train, n_test, seed)

    def answer_span(self, K: int) -> Tuple[int, int]:
        return (K + 2, _perm.NELEM)        # [BOS] p1..pK [EQ] d0..d4 [EOS]

    def difficulty(self, input_ids, K: int) -> int:
        return _perm.chain_length(_perm.perms_from_ids(input_ids, K))

    def sample_balanced(self, K, per_bin, seed, variant="S5"):
        return _perm.sample_balanced_by_chain(K, per_bin, seed, task=variant)


class SortingTask(Task):
    name = "sorting"
    variants = None
    difficulty_label = "#inversions"
    difficulty_col = "inversions"
    by_suffix = "by_inv"
    ignore_index = _sort.IGNORE_INDEX

    @property
    def vocab_size(self) -> int:
        return _sort.VOCAB_SIZE

    def seq_len(self, n: int) -> int:
        return _sort.seq_len_for(n)

    def make_splits(self, n, n_train, n_test, seed, variant=None):
        return _sort.make_splits(n, n_train, n_test, seed)

    def answer_span(self, n: int) -> Tuple[int, int]:
        return (n + 1, n)                  # v_0..v_{n-1} SORT s_0..s_{n-1}

    def difficulty(self, input_ids, n: int) -> int:
        return _sort.n_inversions(_sort.prompt_values(input_ids, n))

    def sample_balanced(self, n, per_bin, seed, variant=None):
        return _sort.sample_balanced_by_inversions(n, per_bin, seed)
