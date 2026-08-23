"""Parameter-budget calculator and (L, d_model) grid generation.

count_params mirrors the exact architecture in model.GPT (pre-LN blocks,
bias in every Linear and LayerNorm, weight-tied lm_head).

For target budgets {~0.5M, ~2M, ~8M} we build grids over depths
L in {2, 4, 6, 8, 12}, shrinking d_model as L grows so total params land
within +/-10% of target. A separate fixed-width grid holds d_model=128 and
lets params grow with L (to disentangle depth from the budget confound).
"""

from __future__ import annotations

from typing import Dict, List, Optional

# Reference vocab / max sequence length used when sizing the grids.
# (matches src/data.py: VOCAB_SIZE=15; seq_len for N=32 is 3*32+5=101 <= 128)
REF_VOCAB_SIZE = 15
REF_MAX_SEQ_LEN = 128

DEPTHS = [2, 4, 6, 8, 12]
BUDGETS: Dict[str, int] = {
    "0.5M": 500_000,
    "2M": 2_000_000,
    "8M": 8_000_000,
}
FIXED_WIDTH = 128


def default_n_heads(d_model: int) -> int:
    """Largest divisor of d_model giving a head dimension closest to 32.
    (Kept in sync with model.default_n_heads.)"""
    best_h, best_err = 1, float("inf")
    for h in range(1, d_model + 1):
        if d_model % h == 0:
            err = abs((d_model // h) - 32)
            if err < best_err:
                best_err, best_h = err, h
    return best_h


def count_params(
    L: int,
    d_model: int,
    n_heads: Optional[int] = None,  # accepted for API symmetry; unused in count
    d_ff: Optional[int] = None,
    vocab_size: int = REF_VOCAB_SIZE,
    max_seq_len: int = REF_MAX_SEQ_LEN,
    looped: bool = False,
) -> int:
    """Total trainable parameter count for a GPT of the given shape.

    L is the number of distinct blocks; for the looped variant a single block
    is shared, so pass looped=True (block count = 1) regardless of iterations.
    """
    if d_ff is None:
        d_ff = 4 * d_model

    emb = vocab_size * d_model + max_seq_len * d_model  # tok + pos (lm_head tied)

    ln = 2 * d_model  # weight + bias
    attn = (3 * d_model * d_model + 3 * d_model) + (d_model * d_model + d_model)
    mlp = (d_model * d_ff + d_ff) + (d_ff * d_model + d_model)
    block = ln + attn + ln + mlp  # ln1 + attn + ln2 + mlp

    n_blocks = 1 if looped else L
    total = emb + n_blocks * block + 2 * d_model  # + final LayerNorm
    return total


def _best_d_model(
    L: int,
    target: int,
    d_min: int = 16,
    d_max: int = 1024,
    step: int = 8,
) -> int:
    """d_model (over the search grid) minimizing |params - target| for depth L."""
    best_d, best_err = d_min, float("inf")
    for d in range(d_min, d_max + 1, step):
        err = abs(count_params(L, d) - target)
        if err < best_err:
            best_err, best_d = err, d
    return best_d


def make_budget_grid(target: int, depths: List[int] = DEPTHS) -> List[dict]:
    """(L, d_model) grid for a fixed parameter budget."""
    rows = []
    for L in depths:
        d = _best_d_model(L, target)
        p = count_params(L, d)
        pct = 100.0 * (p - target) / target
        rows.append({
            "L": L,
            "d_model": d,
            "n_heads": default_n_heads(d),
            "d_ff": 4 * d,
            "params": p,
            "pct_err": pct,
            "within_10pct": abs(pct) <= 10.0,
        })
    return rows


def make_fixed_width_grid(
    d_model: int = FIXED_WIDTH, depths: List[int] = DEPTHS
) -> List[dict]:
    """Fixed-width grid: d_model held constant, params grow with L."""
    rows = []
    for L in depths:
        p = count_params(L, d_model)
        rows.append({
            "L": L,
            "d_model": d_model,
            "n_heads": default_n_heads(d_model),
            "d_ff": 4 * d_model,
            "params": p,
        })
    return rows


def _fmt(p: int) -> str:
    return f"{p/1e6:.3f}M"


def _print_budget_grid(name: str, target: int) -> bool:
    rows = make_budget_grid(target)
    print(f"\n=== Fixed-budget grid: target {name} ({target:,} params) ===")
    print(f"{'L':>3} | {'d_model':>7} | {'n_heads':>7} | {'d_ff':>5} | "
          f"{'params':>9} | {'pct_err':>8} | within 10%")
    print("-" * 66)
    all_ok = True
    for r in rows:
        ok = "yes" if r["within_10pct"] else "NO"
        all_ok = all_ok and r["within_10pct"]
        print(f"{r['L']:>3} | {r['d_model']:>7} | {r['n_heads']:>7} | "
              f"{r['d_ff']:>5} | {_fmt(r['params']):>9} | "
              f"{r['pct_err']:>+7.2f}% | {ok}")
    return all_ok


def _print_fixed_width_grid() -> None:
    rows = make_fixed_width_grid()
    print(f"\n=== Fixed-width grid: d_model={FIXED_WIDTH} (params grow with L) ===")
    print(f"{'L':>3} | {'d_model':>7} | {'n_heads':>7} | {'d_ff':>5} | {'params':>9}")
    print("-" * 44)
    for r in rows:
        print(f"{r['L']:>3} | {r['d_model']:>7} | {r['n_heads']:>7} | "
              f"{r['d_ff']:>5} | {_fmt(r['params']):>9}")


if __name__ == "__main__":
    print("=" * 66)
    print("PARAMETER BUDGET GRIDS")
    print(f"(reference vocab_size={REF_VOCAB_SIZE}, max_seq_len={REF_MAX_SEQ_LEN})")
    print("=" * 66)
    overall_ok = True
    for name, target in BUDGETS.items():
        overall_ok &= _print_budget_grid(name, target)
    _print_fixed_width_grid()
    print(f"\nAll fixed-budget grid entries within +/-10% of target? {overall_ok}")
