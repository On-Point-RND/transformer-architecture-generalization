"""Sorting task (values), adapted from the reference SortingGenerator into the
(input_ids, target_ids) format used by our GPT pipeline.

Sort n values. Token ids are assigned in increasing order of value, so sorting
token ids == sorting values (no decoding needed).

Sequence layout (fixed n, no BOS/EOS needed):
    v_0 v_1 ... v_{n-1}  SORT  s_0 s_1 ... s_{n-1}
    |------ prompt (n+1) ----|  |---- answer (n) ----|
Loss is on the answer tokens only.

This is a PARALLEL task (rank-counting is constant depth), so the expectation
is that it behaves like addition: shallow models solve it, no depth-vs-difficulty
scaling, and the layer-truncation probe shows no "staircase".
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Set, Tuple

import numpy as np

# ---------------------------------------------------------------- vocabulary
PAD_ID = 0
SORT_ID = 1
N_SPECIAL = 2
V_CARD = 52                      # number of distinct values (0..V_CARD-1)
VOCAB_SIZE = N_SPECIAL + V_CARD  # value v -> token id (v + N_SPECIAL)

IGNORE_INDEX = -100


def value_tok(v: int) -> int:
    return v + N_SPECIAL


def tok_value(t: int) -> int:
    return int(t) - N_SPECIAL


def seq_len_for(n: int) -> int:
    return 2 * n + 1


# ---------------------------------------------------------------- difficulty
def n_inversions(values: Sequence[int]) -> int:
    """Number of out-of-order pairs (i<j, values[i] > values[j])."""
    n, c = len(values), 0
    for i in range(n):
        for j in range(i + 1, n):
            if values[i] > values[j]:
                c += 1
    return c


# ---------------------------------------------------------------- generation
def build_example(values: Sequence[int], descending: bool = False):
    """(input_ids, target_ids) for one sort. `values` are 0-based values."""
    n = len(values)
    srt = sorted(values, reverse=descending)
    toks = [value_tok(v) for v in values] + [SORT_ID] + [value_tok(v) for v in srt]
    input_ids = np.array(toks, dtype=np.int64)
    target_ids = np.full_like(input_ids, IGNORE_INDEX)
    for pos in range(n, 2 * n):          # positions predicting s_0..s_{n-1}
        target_ids[pos] = input_ids[pos + 1]
    return input_ids, target_ids


def make_example(n: int, rng: np.random.Generator, v_card: int = V_CARD,
                 duplicates: bool = True, descending: bool = False):
    values = list(rng.choice(v_card, size=n, replace=duplicates))
    return build_example(values, descending=descending)


def example_key(input_ids: Sequence[int], n: int) -> str:
    return ",".join(str(int(t)) for t in input_ids[:n])   # the prompt values


def sample_dataset(n: int, n_examples: int, seed: int, v_card: int = V_CARD,
                   duplicates: bool = True, descending: bool = False,
                   exclude: Optional[Set[str]] = None):
    rng = np.random.default_rng(seed)
    seen: Set[str] = set() if exclude is None else set(exclude)
    out = []
    attempts, max_attempts = 0, max(n_examples * 20, 2000)
    while len(out) < n_examples and attempts < max_attempts:
        attempts += 1
        inp, tgt = make_example(n, rng, v_card, duplicates, descending)
        k = example_key(inp, n)
        if k in seen:
            continue
        seen.add(k)
        out.append((inp, tgt))
    return out


def make_splits(n: int, n_train=100_000, n_test=5_000, seed=0, v_card: int = V_CARD,
                duplicates: bool = True, descending: bool = False):
    train = sample_dataset(n, n_train, seed, v_card, duplicates, descending)
    keys = {example_key(i, n) for i, _ in train}
    test = sample_dataset(n, n_test, seed + 1, v_card, duplicates, descending, exclude=keys)
    return train, test


def sample_balanced_by_inversions(n: int, per_bin: int, seed: int,
                                  v_card: int = V_CARD, descending: bool = False):
    """Examples with the number of inversions balanced across bins {0..max}.
    Long-inversion inputs are constructed by shuffling a sorted list toward the
    target inversion count (rejection is cheap for small n)."""
    rng = np.random.default_rng(seed)
    max_inv = n * (n - 1) // 2
    out = []
    for target in range(max_inv + 1):
        made, attempts = 0, 0
        seen: Set[str] = set()
        while made < per_bin and attempts < per_bin * 400 + 2000:
            attempts += 1
            values = list(rng.choice(v_card, size=n, replace=True))
            if n_inversions(values) != target:
                continue
            inp, tgt = build_example(values, descending=descending)
            k = example_key(inp, n)
            if k in seen:
                continue
            seen.add(k)
            out.append((inp, tgt))
            made += 1
    return out


def inversions_of(examples, n: int) -> List[int]:
    return [n_inversions([tok_value(e[0][i]) for i in range(n)]) for e in examples]


# ---------------------------------------------------------------- decode
def prompt_values(input_ids: Sequence[int], n: int) -> List[int]:
    return [tok_value(input_ids[i]) for i in range(n)]


def answer_values(input_ids: Sequence[int], n: int) -> List[int]:
    return [tok_value(input_ids[n + 1 + i]) for i in range(n)]


# ---------------------------------------------------------------- checks
def _acceptance_checks() -> None:
    print("=" * 66)
    print("data_sort.py ACCEPTANCE CHECKS")
    print("=" * 66)
    print(f"vocab={VOCAB_SIZE}  v_card={V_CARD}  (PAD=0, SORT=1, values->2..)")

    for n in (8, 16):
        rng = np.random.default_rng(0)
        print(f"\n--- 2 examples, n={n} ---")
        for _ in range(2):
            inp, tgt = make_example(n, rng)
            vals = prompt_values(inp, n)
            ans = answer_values(inp, n)
            print(f"  input : {vals}")
            print(f"  sorted: {ans}   inversions(input)={n_inversions(vals)}")
            assert ans == sorted(vals), "answer not sorted!"
            assert (tgt != IGNORE_INDEX).sum() == n, "loss mask should cover n answer tokens"
        print(f"  seq_len={len(inp)} (=2n+1={seq_len_for(n)})  supervised tokens={n}  OK")

    print("\n--- balanced-by-inversions sampler (n=8) ---")
    bal = sample_balanced_by_inversions(8, per_bin=30, seed=1)
    invs = inversions_of(bal, 8)
    max_inv = 8 * 7 // 2
    counts = {k: invs.count(k) for k in range(max_inv + 1)}
    filled = sum(1 for v in counts.values() if v >= 30)
    print(f"  inversion bins 0..{max_inv}: {filled}/{max_inv + 1} filled to 30 "
          f"(rare extreme bins may be short)")

    print("\nAll data_sort checks executed.")


if __name__ == "__main__":
    _acceptance_checks()
