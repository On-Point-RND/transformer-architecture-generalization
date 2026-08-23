"""Data generation for the depth-vs-sequentiality addition subproject.

Encoding (following Lee et al. 2023): digit-level, operands reversed
(least-significant digit first). Leading zeros are allowed so both operands
have exactly N digits.

Two task conditions:
  - SEQ:   c = a + b            (integer addition, carries propagate)
  - INDEP: c_i = (a_i + b_i) % 10   (per-digit sum, NO carry)

To hold vocabulary, sequence length and surface form fixed across the two
conditions, the answer c is ALWAYS written with N+1 digits:
  - SEQ:   the true sum, zero-padded to N+1 digits.
  - INDEP: the per-digit sums (N digits) with an extra leading 0 (its top
           digit is always 0 because there is no carry).

Full token sequence (indices, LSB-first digits):
    [BOS] a_r[0..N-1] '+' b_r[0..N-1] '=' c_r[0..N] [EOS]
length = 3N + 5 for both tasks and every example at a given N.

Loss (target_ids) is defined on the ANSWER tokens only: every non-answer
target position is filled with IGNORE_INDEX (-100).
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Set, Tuple

import numpy as np

# --------------------------------------------------------------------------
# Vocabulary
# --------------------------------------------------------------------------
PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
PLUS_ID = 3
EQ_ID = 4
DIGIT_OFFSET = 5  # digit d -> DIGIT_OFFSET + d

VOCAB_SIZE = DIGIT_OFFSET + 10  # 15

IGNORE_INDEX = -100  # torch's default cross-entropy ignore index

ID_TO_STR = {
    PAD_ID: "PAD",
    BOS_ID: "BOS",
    EOS_ID: "EOS",
    PLUS_ID: "+",
    EQ_ID: "=",
}
for _d in range(10):
    ID_TO_STR[DIGIT_OFFSET + _d] = str(_d)

TASKS = ("SEQ", "INDEP")


def seq_len_for(N: int) -> int:
    """Total token-sequence length for a given operand digit count N."""
    return 3 * N + 5


def max_seq_len_for(Ns: Sequence[int]) -> int:
    return max(seq_len_for(N) for N in Ns)


# --------------------------------------------------------------------------
# Digit / number helpers  (all digit lists are LSB-first)
# --------------------------------------------------------------------------
def digits_to_int(digits: Sequence[int]) -> int:
    """LSB-first digit list -> integer."""
    val = 0
    for i, d in enumerate(digits):
        val += int(d) * (10 ** i)
    return val


def int_to_digits(value: int, length: int) -> List[int]:
    """integer -> LSB-first digit list, zero-padded/truncated to `length`."""
    digs = [0] * length
    v = value
    for i in range(length):
        digs[i] = v % 10
        v //= 10
    return digs


def _digits_to_readable(digits: Sequence[int]) -> str:
    """LSB-first digit list -> normal MSB-first string (e.g. [3,2,1] -> '123')."""
    return "".join(str(int(d)) for d in reversed(digits))


# --------------------------------------------------------------------------
# Carry-chain length
# --------------------------------------------------------------------------
def carry_chain_length(a_digits: Sequence[int], b_digits: Sequence[int]) -> int:
    """Longest run of consecutive positions over which a carry is generated and
    propagated (i.e. the maximum distance a carry travels).

    Digits are LSB-first. At each position i the sum is
        s_i = a_i + b_i + carry_in_i,   carry_out_i = 1 if s_i >= 10 else 0.
    A carry_out of 1 means position i hands a carry to position i+1. The chain
    length is the longest maximal run of consecutive positions with
    carry_out == 1. Returns 0 when no carry is ever produced.
    """
    n = max(len(a_digits), len(b_digits))
    carry = 0
    cur_run = 0
    best = 0
    for i in range(n):
        a = int(a_digits[i]) if i < len(a_digits) else 0
        b = int(b_digits[i]) if i < len(b_digits) else 0
        s = a + b + carry
        carry_out = 1 if s >= 10 else 0
        if carry_out:
            cur_run += 1
            if cur_run > best:
                best = cur_run
        else:
            cur_run = 0
        carry = carry_out
    return best


# --------------------------------------------------------------------------
# Example construction
# --------------------------------------------------------------------------
def _digit_token(d: int) -> int:
    return DIGIT_OFFSET + int(d)


def _build_ids(
    a_digits: Sequence[int],
    b_digits: Sequence[int],
    c_digits: Sequence[int],
) -> Tuple[np.ndarray, np.ndarray]:
    """Assemble (input_ids, target_ids) from LSB-first digit lists.

    a_digits, b_digits have length N; c_digits has length N+1.
    """
    N = len(a_digits)
    assert len(b_digits) == N
    assert len(c_digits) == N + 1

    tokens: List[int] = [BOS_ID]
    tokens += [_digit_token(d) for d in a_digits]
    tokens.append(PLUS_ID)
    tokens += [_digit_token(d) for d in b_digits]
    tokens.append(EQ_ID)
    answer_start = len(tokens)  # index of first answer token (c_r[0])
    tokens += [_digit_token(d) for d in c_digits]
    tokens.append(EOS_ID)

    input_ids = np.array(tokens, dtype=np.int64)

    # Labels: next-token prediction, loss on answer tokens (c_r + EOS) only.
    target_ids = np.full_like(input_ids, IGNORE_INDEX)
    # Positions answer_start-1 .. len-2 predict input_ids[answer_start .. len-1].
    for pos in range(answer_start - 1, len(tokens) - 1):
        target_ids[pos] = input_ids[pos + 1]

    return input_ids, target_ids


def build_example(
    task: str,
    a_digits: Sequence[int],
    b_digits: Sequence[int],
) -> Tuple[np.ndarray, np.ndarray]:
    """Build (input_ids, target_ids) for given LSB-first operands and task.

    The answer c always has N+1 digits (SEQ: true sum; INDEP: per-digit sums
    with a leading 0). Note L(x) = carry_chain_length(a, b) is a property of the
    operands only, so INDEP examples can be stratified by L(x) too (as a
    control: L(x) is irrelevant to the INDEP answer).
    """
    N = len(a_digits)
    a = [int(d) for d in a_digits]
    b = [int(d) for d in b_digits]
    if task == "SEQ":
        c = int_to_digits(digits_to_int(a) + digits_to_int(b), N + 1)
    elif task == "INDEP":
        c = [(a[i] + b[i]) % 10 for i in range(N)] + [0]  # top digit always 0
    else:
        raise ValueError(f"unknown task {task!r}")
    return _build_ids(a, b, c)


def make_example_seq(N: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """One SEQ (integer addition) example. Returns (input_ids, target_ids)."""
    a = list(rng.integers(0, 10, size=N))
    b = list(rng.integers(0, 10, size=N))
    return build_example("SEQ", a, b)


def make_example_indep(N: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """One INDEP (per-digit sum mod 10, no carry) example."""
    a = list(rng.integers(0, 10, size=N))
    b = list(rng.integers(0, 10, size=N))
    return build_example("INDEP", a, b)


def _operands_from_ids(input_ids: Sequence[int], N: int) -> Tuple[List[int], List[int]]:
    """Recover LSB-first a, b digit lists from an input_ids sequence."""
    a = [int(input_ids[1 + i]) - DIGIT_OFFSET for i in range(N)]
    b = [int(input_ids[N + 2 + i]) - DIGIT_OFFSET for i in range(N)]
    return a, b


def example_key(input_ids: Sequence[int]) -> str:
    """String key for de-duplication (operands only; the answer is a function
    of the operands + task, but tasks are sampled separately)."""
    return ",".join(str(int(t)) for t in input_ids)


# --------------------------------------------------------------------------
# Decoding
# --------------------------------------------------------------------------
def decode_ids(ids: Sequence[int]) -> str:
    """Token id sequence -> raw token string (reversed-digit surface form)."""
    return " ".join(ID_TO_STR.get(int(t), f"<{int(t)}>") for t in ids)


def example_to_readable(input_ids: Sequence[int], N: int, task: str) -> str:
    """Human-readable 'a + b = c' (normal MSB-first orientation)."""
    a, b = _operands_from_ids(input_ids, N)
    # answer digits: indices 2N+3 .. 3N+3 (N+1 digits), LSB-first
    c_start = 2 * N + 3
    c = [int(input_ids[c_start + i]) - DIGIT_OFFSET for i in range(N + 1)]
    a_s = _digits_to_readable(a)
    b_s = _digits_to_readable(b)
    c_s = _digits_to_readable(c)
    op = "+" if task == "SEQ" else "(+)"  # (+) denotes per-digit mod-10 sum
    return f"{a_s} {op} {b_s} = {c_s}"


# --------------------------------------------------------------------------
# Dataset samplers
# --------------------------------------------------------------------------
def sample_dataset(
    task: str,
    N: int,
    n_examples: int,
    seed: int,
    exclude: Optional[Set[str]] = None,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Uniformly sample n_examples distinct examples for (task, N).

    De-duplicates within the returned set and against `exclude` (a set of
    example_key strings, e.g. the training split's keys).
    """
    if task not in TASKS:
        raise ValueError(f"unknown task {task!r}")
    rng = np.random.default_rng(seed)
    make = make_example_seq if task == "SEQ" else make_example_indep

    seen: Set[str] = set() if exclude is None else set(exclude)
    out: List[Tuple[np.ndarray, np.ndarray]] = []

    # Cap attempts to avoid infinite loops when the space is small (tiny N).
    max_distinct = 10 ** (2 * N)  # number of (a, b) operand pairs
    target = min(n_examples, max_distinct)
    attempts = 0
    max_attempts = max(target * 50, 1000)
    while len(out) < target and attempts < max_attempts:
        attempts += 1
        inp, tgt = make(N, rng)
        k = example_key(inp)
        if k in seen:
            continue
        seen.add(k)
        out.append((inp, tgt))
    return out


def make_splits(
    task: str,
    N: int,
    n_train: int = 100_000,
    n_test: int = 5_000,
    seed: int = 0,
) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], List[Tuple[np.ndarray, np.ndarray]]]:
    """Disjoint train / id-test splits for (task, N), de-duplicated by key."""
    train = sample_dataset(task, N, n_train, seed=seed)
    train_keys = {example_key(inp) for inp, _ in train}
    test = sample_dataset(task, N, n_test, seed=seed + 1, exclude=train_keys)
    return train, test


def sample_balanced_by_chain(
    N: int,
    per_bin: int,
    seed: int,
    task: str = "SEQ",
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Examples with carry-chain length L(x) balanced across {0..N}.

    L(x) is a property of the operands, so this works for either task: SEQ
    answers depend on L(x); INDEP answers do not (control condition).

    Long chains are exponentially rare under uniform sampling, so examples are
    CONSTRUCTED directly to have an exact target chain length L:
      - a run of length L is placed starting at position 0 (LSB):
          * position 0 GENERATES a carry (a0 + b0 >= 10),
          * positions 1..L-1 PROPAGATE  (a_i + b_i == 9),
      - all remaining positions L..N-1 are "safe" (a_i + b_i <= 8) so the carry
        is absorbed and no other (longer) run can form.
    Free digit choices are randomized for variety and de-duplicated per bin.
    """
    rng = np.random.default_rng(seed)

    # Precompute the digit-pair menus (as LSB-first single-position choices).
    generate_pairs = [(a, b) for a in range(10) for b in range(10) if a + b >= 10]
    propagate_pairs = [(a, b) for a in range(10) for b in range(10) if a + b == 9]
    safe_pairs = [(a, b) for a in range(10) for b in range(10) if a + b <= 8]

    out: List[Tuple[np.ndarray, np.ndarray]] = []
    for L in range(0, N + 1):
        seen: Set[str] = set()
        made = 0
        attempts = 0
        max_attempts = per_bin * 200 + 500
        while made < per_bin and attempts < max_attempts:
            attempts += 1
            a = [0] * N
            b = [0] * N
            if L >= 1:
                # position 0 generates
                ga, gb = generate_pairs[rng.integers(len(generate_pairs))]
                a[0], b[0] = ga, gb
                # positions 1..L-1 propagate
                for i in range(1, L):
                    pa, pb = propagate_pairs[rng.integers(len(propagate_pairs))]
                    a[i], b[i] = pa, pb
            # positions L..N-1 are safe (never generate, absorb any carry)
            for i in range(L, N):
                sa, sb = safe_pairs[rng.integers(len(safe_pairs))]
                a[i], b[i] = sa, sb

            # Sanity: verify the constructed chain length is exactly L.
            if carry_chain_length(a, b) != L:
                continue

            inp, tgt = build_example(task, a, b)
            k = example_key(inp)
            if k in seen:
                continue
            seen.add(k)
            out.append((inp, tgt))
            made += 1
    return out


def chain_lengths(examples: Sequence[Tuple[np.ndarray, np.ndarray]], N: int) -> List[int]:
    """Carry-chain length L(x) for each example (operands recovered from ids)."""
    out = []
    for inp, _ in examples:
        a, b = _operands_from_ids(inp, N)
        out.append(carry_chain_length(a, b))
    return out


# --------------------------------------------------------------------------
# M1 acceptance-check driver
# --------------------------------------------------------------------------
def _acceptance_checks() -> None:
    import os

    print("=" * 70)
    print("MILESTONE 1 ACCEPTANCE CHECKS (src/data.py)")
    print("=" * 70)

    N = 8
    rng = np.random.default_rng(0)

    print(f"\n--- 20 SEQ examples (N={N}), decoded to human-readable ---")
    print("(operator '+' = integer addition with carries)")
    for _ in range(20):
        inp, tgt = make_example_seq(N, rng)
        print("  ", example_to_readable(inp, N, "SEQ"))

    print(f"\n--- 20 INDEP examples (N={N}), decoded to human-readable ---")
    print("(operator '(+)' = per-digit sum mod 10, no carry)")
    for _ in range(20):
        inp, tgt = make_example_indep(N, rng)
        print("  ", example_to_readable(inp, N, "INDEP"))

    print("\n--- raw token surface form of one SEQ example (reversed digits) ---")
    inp, tgt = make_example_seq(N, np.random.default_rng(42))
    print("   ids   :", decode_ids(inp))
    print("   labels:", decode_ids([t if t != IGNORE_INDEX else PAD_ID for t in tgt]),
          "(PAD shown where loss is ignored)")
    print("   seq_len:", len(inp), "== 3N+5 ==", seq_len_for(N))

    print("\n--- manual verification: 3 additions ---")
    manual = [(999, 1), (555, 555), (123, 456)]
    for a_val, b_val in manual:
        Nm = max(len(str(a_val)), len(str(b_val)))
        a = int_to_digits(a_val, Nm)
        b = int_to_digits(b_val, Nm)
        got = digits_to_int(a) + digits_to_int(b)
        ok = "OK" if got == a_val + b_val else "MISMATCH"
        print(f"   {a_val} + {b_val} = {got}  (expected {a_val + b_val})  [{ok}]")

    print("\n--- manual verification: 3 carry_chain_length values ---")
    cases = [
        # (a, b, expected, note)
        (999, 1, 3, "9,9,9 + 1 -> carry travels all 3 positions"),
        (123, 456, 0, "3+6,2+5,1+4 -> no carries"),
        (1999, 1, 3, "carry from units through three 9s"),
    ]
    for a_val, b_val, expected, note in cases:
        Nm = max(len(str(a_val)), len(str(b_val)))
        a = int_to_digits(a_val, Nm)
        b = int_to_digits(b_val, Nm)
        L = carry_chain_length(a, b)
        ok = "OK" if L == expected else "MISMATCH"
        print(f"   L({a_val} + {b_val}) = {L}  (expected {expected})  [{ok}]  # {note}")

    # ---- histogram of L(x): uniform vs balanced ----
    print("\n--- histogram of L(x): uniform vs balanced sampler ---")
    Nh = 16
    n_uniform = 20_000
    per_bin = 500
    urng = np.random.default_rng(7)
    uniform_examples = [make_example_seq(Nh, urng) for _ in range(n_uniform)]
    uniform_L = chain_lengths(uniform_examples, Nh)
    balanced_examples = sample_balanced_by_chain(Nh, per_bin=per_bin, seed=7)
    balanced_L = chain_lengths(balanced_examples, Nh)

    bins = list(range(0, Nh + 1))
    uni_hist = [uniform_L.count(k) for k in bins]
    bal_hist = [balanced_L.count(k) for k in bins]
    print(f"   N={Nh}, uniform n={n_uniform}, balanced per_bin={per_bin}")
    print(f"   {'L':>3} | {'uniform':>8} | {'balanced':>9}")
    print("   " + "-" * 26)
    for k in bins:
        print(f"   {k:>3} | {uni_hist[k]:>8} | {bal_hist[k]:>9}")

    bal_nonempty = [c for c in bal_hist if c > 0]
    flat = (max(bal_hist) - min(bal_nonempty)) <= 1 if bal_nonempty else False
    print(f"   balanced flat across bins? {flat} "
          f"(min={min(bal_nonempty) if bal_nonempty else 0}, max={max(bal_hist)})")

    # Save figure.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True)
        axes[0].bar(bins, uni_hist, color="#4477aa")
        axes[0].set_title(f"Uniform sampling (n={n_uniform})")
        axes[0].set_xlabel("carry-chain length L(x)")
        axes[0].set_ylabel("count")
        axes[1].bar(bins, bal_hist, color="#ee6677")
        axes[1].set_title(f"Balanced sampler (per_bin={per_bin})")
        axes[1].set_xlabel("carry-chain length L(x)")
        fig.suptitle(f"L(x) distribution, SEQ, N={Nh}")
        fig.tight_layout()
        os.makedirs("figures", exist_ok=True)
        path = os.path.join("figures", "m1_chain_length_hist.png")
        fig.savefig(path, dpi=120)
        print(f"\n   histogram figure saved -> {path}")
    except Exception as e:  # pragma: no cover
        print(f"   (figure not saved: {e})")

    print("\nAll M1 checks executed.")


if __name__ == "__main__":
    _acceptance_checks()
