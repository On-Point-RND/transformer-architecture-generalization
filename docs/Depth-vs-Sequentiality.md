# Depth vs. Sequentiality: pipeline guide

**What this project is about (one sentence):** we train small transformers on
three toy tasks (addition, permutation composition, sorting) and measure **how
many layers each task needs**. The hypothesis: a task needs transformer depth in
proportion to the length of its *sequential* (irreducible) dependency chain;
parallelizable tasks need almost no depth.

- **Sequential task:** permutation composition on **S5** (the symmetric group;
  its word problem is NC¹-complete, so it cannot be collapsed to constant depth).
- **Parallel controls:** addition (carry-lookahead) and sorting (rank counting),
  plus **C5** (the abelian cyclic control for the permutation task).

Every example carries a **difficulty `L(x)`** — the length of its sequential
chain: carry-chain length for addition, number of non-identity permutations for
S5/C5, number of inversions for sorting. The main plots are accuracy vs. `L(x)`.

---

## Repository layout

```
configs/     experiment configs (YAML) — one per experiment
src/         all the code (see "What each file does")
tests/       smoke_test.py — fast end-to-end check
results/     CSV outputs (created on first run)
checkpoints/ trained models (.pt, created on first run)
```

All commands are run **from the repository root**. Training writes `results/`
and `checkpoints/` relative to the current directory.

```bash
pip install -r requirements.txt      # torch, numpy, pyyaml, pandas, matplotlib, tqdm
python tests/smoke_test.py           # ~70s on CPU; prints "ALL GREEN" if healthy
```

---

## What each file does

Think of the files in two groups: **things you run by hand**, and **machinery
that runs under the hood**.

### You run these by hand

**`src/run.py` — the experiment launcher.**
Takes one config (`.yaml`), trains a batch of models (different depths/widths)
according to it, and writes the results as CSV tables.
*In → out:* `configs/<name>.yaml` → `results/*.csv` (metrics) + `checkpoints/*.pt`
(trained models). *This is your main command:*
`python src/run.py configs/<name>.yaml`.

**`src/probe_linear.py` — the "honest X-ray" across layers.**
Takes an already-trained model and checks **at which internal layer the answer
already becomes readable**, by training a tiny linear read-out on top of each
layer. This shows that a parallel task (sorting) is readable from layer 1, while
a sequential task (S5) is only built up layer by layer.
*In → out:* `checkpoints/*.pt` → `results/<name>_probe.csv`.

**`src/probe_layers.py` — the "simple X-ray" (baseline).**
Same spirit but cruder: it just truncates the model to `k` layers and reads off
the answer. This method is slightly "confounded" (the head was trained only for
the full depth — see the file's docstring), so its main use is as a **contrast**
to the honest `probe_linear.py`. Bonus: it supports **transfer** — e.g. take
models trained on S5 and evaluate them on C5 (`--eval-variant C5`).
*In → out:* `checkpoints/*.pt` → `results/<name>_probe.csv`.

**`src/train.py` — train a single model.**
The heart of training: one universal loop that works for all three tasks.
`run.py` calls it many times; you can also train a single model directly from
the command line for a quick check.

**`tests/smoke_test.py` — "is everything alive?"**
In ~70 seconds on CPU it runs tiny versions of **every** experiment and reports
`ALL GREEN` or exactly where something broke. Run it after any code/config
change, before a real run.

### Under the hood (you rarely touch these)

**`src/model.py` — the neural network itself.**
The transformer (a GPT). Plus three knobs specific to our experiments: a
"repeat one block T times" mode (looped), "truncate to `k` layers" (for
`probe_layers`), and "hand back the internal representation of every layer"
(for `probe_linear`).

**`src/datasets/` — the task generators (where the data comes from).**
- `addition.py` — **addition** examples (two modes: with carries / per-digit
  without carries).
- `permutation.py` — **permutation composition** examples (S5 = the hard,
  sequential one; C5 = the easy, control one).
- `sorting.py` — **number-sorting** examples.
- `base.py` + `tasks.py` + `__init__.py` — the "adapter" that puts all three
  tasks behind one common interface, so `train.py` / `run.py` / the probes work
  with any task the same way without knowing its details. `base.py` is the
  shared contract (`Task`), `tasks.py` has the three implementations,
  `__init__.py` is the registry (`get_task("permutation")`).
- Every task exposes the key quantity **`L(x)` (example difficulty)**: carry-chain
  length for addition, number of real composition steps for permutations, number
  of inversions for sorting. All the main plots are stratified by `L(x)`.

**`src/budgets.py` — equalize models by parameter count.**
Computes what width to use at each depth so all models have roughly the same
number of parameters (a fair "deeper vs. wider" comparison). Used only by the
`fixed_budget` configs.

---

## Experiments — what exists and how to run them

The workflow is always the same: **1) train models with a config → 2) (optional)
look inside with a probe → 3) plot from the CSV.** All commands from the repo root.

### Step 1. Train models (the core)

```bash
python src/run.py configs/<name>.yaml            # add --quiet for less logging
```

Produces `results/<sweep>_summary.csv` (one row per model) and
`results/<sweep>_by_chain.csv` (or `_by_inv.csv` for sorting) — accuracy split
by difficulty `L(x)`.

| What it shows | Config(s) | Task |
|---|---|---|
| **Main "depth vs. difficulty" plot** for the sequential task | `perm_depth_fixedwidth_v2` | permutation |
| Depth is **not** needed for addition (control) | `add_pilot_budget`, `add_depth_fixedwidth`, `add_depth_bigN` | addition |
| Width alone does **not** rescue it (depth is what matters) | `perm_width` | permutation |
| Fill in the missing cells of the depth×width grid | `perm_grid_fill` | permutation |

### Step 2. Look inside the model (per-layer probes)

First train a grid of models (this saves the checkpoints):

```bash
python src/run.py configs/perm_grid.yaml         # or add_grid / sort_grid
```

Then the **honest probe**:

```bash
python src/probe_linear.py --ckpts 'checkpoints/perm_grid_*_S5_*.pt' --out perm_grid
```

→ `results/perm_grid_probe.csv`: for each layer, from which difficulty `L(x)` the
answer is already linearly readable (a row with the difficulty column = `-1` is
the overall exact-match at that layer).

Also available:
- **Truncation baseline:** `python src/probe_layers.py --config configs/add_grid_probe.yaml`
- **Cross-variant transfer (S5 models on C5):**
  `python src/probe_layers.py --config configs/perm_grid_cross_probe.yaml`

### Additional experiments (mechanisms)

| Question | Config | Extra output |
|---|---|---|
| Does the number of attention heads matter? | `add_head_ablation`, `perm_head_ablation` | — |
| Does "repeating one block" substitute for depth? | `add_loop` | — |
| Does the model work on lengths it never saw? | `add_length_gen` | extra `results/<sweep>_lengthgen.csv` |
| Sanity: does the pipeline learn at all? | `add_solved_repro` | — |

### Step 3. Process the results

All outputs are plain CSV, loadable with `pandas.read_csv`:

- **Main plot (money-plot):** take `*_by_chain.csv`, group by `(variant,
  n_layers)`, and plot `exact_match` against `chain_length`. Expectation: S5
  shows a "staircase" (a deeper layer solves a longer `L(x)`); addition/sorting
  stay flat.
- **Depth summary:** `*_summary.csv`, column `id_test_em_ag` (the honest
  autoregressive exact-match) against `n_layers`.

---

## Config reference (full list)

**Addition (parallel controls):**
| Config | Experiment |
|---|---|
| `add_pilot_budget` | fixed-parameter-budget pilot (the fastest money-plot) |
| `add_pilot_figa` | reduced pilot that runs on a laptop GPU |
| `add_depth_fixedwidth` | depth sweep at fixed width, N=16 |
| `add_depth_bigN` | depth × length sweep (N ∈ {16,24,32}) |
| `add_head_ablation` | vary only the number of attention heads |
| `add_length_gen` | train at N=8, test on larger N without retraining |
| `add_loop` | one shared block applied T times |
| `add_solved_repro` | train addition to ~1.0 (sanity / reproduction) |
| `add_grid` | depth × width grid — **trains** checkpoints for the probes |
| `add_grid_probe` | **runs** the truncation probe over the `add_grid` checkpoints |

**Permutation composition (sequential task + C5 control):**
| Config | Experiment |
|---|---|
| `perm_depth_fixedwidth` | depth sweep S5/C5, K=16 |
| `perm_depth_fixedwidth_v2` | improved (K=8, 5 seeds, 1M examples) — cleanest money-plot |
| `perm_width` | width sweep at fixed depth |
| `perm_grid_fill` | fill missing depth×width cells |
| `perm_head_ablation` | vary only the number of attention heads |
| `perm_grid` | depth × width grid — **trains** checkpoints for the probes |
| `perm_grid_cross_probe` | **runs** the S5→C5 transfer probe |

**Sorting (parallel control):**
| Config | Experiment |
|---|---|
| `sort_grid` | depth × width grid + checkpoints for the probe |

> Config naming convention: `<task>_<experiment>.yaml`, with `task ∈
> {add, perm, sort}`. Configs that **run a probe** (they contain a `ckpts:`
> glob and drive `probe_layers.py --config`) end in `_probe`.

---

## Notes / gotchas

- **Run from the repo root**, always (`python src/run.py configs/...`). The
  scripts add `src/` to the import path; the working directory stays the root so
  `results/` and `checkpoints/` land there.
- **Resumable:** `run.py` skips any run whose tag is already in the summary CSV,
  so an interrupted sweep continues where it left off. Use `--limit N` to run
  only the first N pending runs.
- **Device:** picks CUDA → MPS → CPU automatically.
- Adding a **new task** only requires implementing the `Task` interface
  (`src/datasets/base.py`) and registering it in `src/datasets/__init__.py`;
  training, sweeps and probes then work on it for free.
```
