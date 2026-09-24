# Depth sweep (n_layer = 1–4): task parameters and failure/success predictions

Companion to `task_primitives.tex` / `architecture_axes.tex`. For each task:
the `configs/main.yaml`-style params to use, and the predicted outcome at each
depth, with the reasoning tied back to the task's complexity tier. Predictions
are directional, calibrated where possible against literature precedent
(Merrill, Petty & Sabharwal's own depth-vs-length curves for the $A_5$ family)
— not exact numbers for this repo's specific width/hyperparameters. Treat
disagreement with these predictions as informative, not as an experiment
failure.

## The two regimes

- **Task-complexity-bound tasks** (`s5`, `state_based_recall`, `maze`): a real
  wall is predicted. Depth should produce a genuine success→failure
  transition as difficulty grows, and the transition point should move
  outward as depth increases.
- **$TC^0$ tasks** (`kv_retrieval`, `selective_count`, `sorting`, `c5`): **no**
  wall is predicted. Depth 1–4 should look roughly flat across difficulty: if
  a difficulty threshold cripples one of these regardless of depth, that's a
  Capacity or Learnability failure, not a Task-complexity one — depth won't
  fix it either way.

---

## `s5` — sequential state tracking, $NC^1$-complete (Barrington 1986)

```yaml
task:
  name: s5
  params:
    n_permutations: [4, 25]   # eval checkpoints: 4, 8, 12, 16, 20, 24; OOD tail 28-40
```

Calibration: Merrill/Petty/Sabharwal's own $A_5$ curves used lengths 5–20 and
saw depth requirements sweep 1→4 across exactly that range. Using their fitted
relation $n(d) \approx 2^{(d+15.8)/4.8}$ as a rough guide (not an exact
prediction for this repo's hyperparameters):

| Depth | Predicted max solvable $n$ | Prediction |
|---|---|---|
| 1 | $\approx 11$ | succeeds up to $n\approx 8$–11, fails beyond |
| 2 | $\approx 13$ | succeeds up to $n\approx 12$–13 |
| 3 | $\approx 15$ | succeeds up to $n\approx 14$–15 |
| 4 | $\approx 17$ | succeeds up to $n\approx 16$–17, still fails by $n=24$+ |

No depth in {1,2,3,4} should solve the OOD tail ($n\geq28$) — that's the
point: the wall should persist and merely shift, not disappear.

**Do not extend depth beyond 4 for this task** — range is precedented.

---

## `c5` — matched control for `s5`, $TC^0$ (solvable/abelian group)

```yaml
task:
  name: c5
  params:
    n_permutations: [4, 25]   # MUST match s5's range exactly — this is the control
```

**Prediction: flat, near-ceiling accuracy at every depth 1–4, across the
entire range where `s5` collapses.** If `c5` is not flat here, the `s5`
result cannot be attributed to non-solvability — the control has failed and
the causal claim needs to be re-examined before citing the `s5` result at all.

---

## `state_based_recall` — composition, unsolvable by any pure architecture (Olmo Hybrid Thm. 1)

```yaml
task:
  name: state_based_recall
  params:
    n_bits: [16, 17]     # fixed narrow — not the axis under test
    n_swaps: [4, 25]      # mirrors s5's range; this is the state-tracking axis
    max_bits: 32
```

| Depth | Prediction | Reasoning |
|---|---|---|
| 1–4 | should track `s5`'s collapse curve on `n_swaps`, or fail *earlier* | every model tested here is a pure transformer (no linear-RNN/hybrid exists yet in this repo); Thm. 1 says no pure architecture solves this as `n_swaps` grows, so failure is expected at least as early as `s5`'s |
| 4 | **may still fail even where `s5` at depth 4 would succeed** | composing state-tracking with retrieval is strictly harder than either alone; if depth-4 `state_based_recall` fails at a lower `n_swaps` than depth-4 `s5`, that's the composition penalty made visible |

Held `n_bits` fixed deliberately — varying it too would confound the
state-tracking (depth) story with a retrieval (capacity) story.

**Be ready to extend to depth 5–6 specifically for this task** if depth 4
doesn't crack it anywhere in the tested range — there's no external
precedent (unlike `s5`) to say 1–4 is enough.

---

## `maze` — planning/graph reachability, $NC^1$-hard, $\subseteq L$, exact class open (Allender et al. 2006)

```yaml
task:
  name: maze
  params:
    grid_size: [4, 12]
    max_grid_size: 12
    min_path_length: 2   # let path length vary naturally; bucket eval by it, not grid_size
```

Bucket evaluation by the task's own `path_length` metadata, **not** by
`grid_size` — a large grid can still have a short path, and the theoretical
claim ("one plan step per layer") is about path length specifically.

| Depth | Predicted solvable `path_length` | Prediction |
|---|---|---|
| 1 | $\leq 1$–2 | only trivial, near-adjacent origin/target pairs |
| 2 | $\leq 2$–4 | short paths only |
| 3 | $\leq 4$–6 | |
| 4 | $\leq 5$–8 | |
| longer paths | **fail regardless of depth 1–4** | if `grid_size` up to 12 produces paths well past 8, depth 4 should still fail on those — expected, not a bug |

**Most likely of the four to need depth beyond 4.** No external calibration
exists for this task the way it does for `s5`; if paths in the tested grid
range exceed ~8 (likely once `grid_size` > 6–7), extend to depth 5–8 to find
where the wall actually sits, rather than reading "fails past depth-4-worth
of path length" as the wall itself.

---

## `kv_retrieval` — content-based retrieval, trivial $TC^0$

```yaml
task:
  name: kv_retrieval
  params:
    k_card: 64
    v_card: 64
    n_pairs: [8, 33]   # widened from the [4,9] default — too easy to stress capacity
    depth: 1
```

**Prediction: flat across depth 1–4 at every `n_pairs` value tested.** Any
accuracy drop as `n_pairs` grows is a Capacity story (state/cache size), not
a depth story — depth should not visibly help or hurt.

---

## `selective_count` — aggregation/counting, trivial $TC^0$ (counting is native to $TC^0$)

```yaml
task:
  name: selective_count
  params:
    length_range: [16, 65]
    relevant_count_range: [1, 17]
```

**Prediction: flat across depth 1–4.** Same logic as `kv_retrieval` — counting
is TC⁰-native, no depth wall predicted at any tested length.

---

## `sorting` — aggregation, $TC^0$ but $O(\log n)$-depth in the classical (AKS) construction

```yaml
task:
  name: sorting
  params:
    n_items: [4, 17]   # log2(n_items) spans 1-4 across ~2-16 items
    item_len: [2, 5]
```

| Depth | Prediction |
|---|---|
| 1–4 | **mild** possible benefit as `n_items` grows past $2^d$, much weaker than `s5`'s collapse |

This is the one $TC^0$ task where a *small* depth effect wouldn't be a
surprise — sorting networks are $O(\log n)$ depth, not $O(1)$. If `sorting`
shows a real (if weak) depth trend here while `kv_retrieval`/`selective_count`
stay flat, that's a clean secondary confirmation of the log-depth vs.
no-depth-requirement distinction already in `task_primitives.tex`.

---

## Summary: extend past depth 4?

| Task | Extend beyond 4? |
|---|---|
| `s5` / `c5` | No — precedented range |
| `kv_retrieval` / `selective_count` / `sorting` | No — no wall predicted either way |
| `state_based_recall` | Maybe — extend to 5–6 if depth 4 doesn't crack it anywhere tested |
| `maze` | Probably — extend to 5–8 if paths past length ~8 remain unsolved at depth 4 |

## Open item

`dyck` is not in the current task registry (`tasks/__init__.py` lists only
`kv_retrieval, selective_count, sorting, c5, ruler, s5, state_based_recall,
maze`). It was argued for keeping in the last round of task-list pruning
(strongest "solved by LLMs today" fit of anything cut) — confirm whether its
removal from the registry was intentional before finalizing the task list
this depth sweep is meant to validate.
