# Positional encoding continuation experiments

This directory adapts the useful EVE-PE pilots to the current `main` APIs. It
does not merge the old branch wholesale.

## Audit summary

| Area | EVE-PE pilot | Current implementation |
|---|---|---|
| KV task | `k_card`, `v_card`, `n_pairs`, `depth` | `vocab_size`, `input_seq_len`, `num_kv_pairs`, `power_a` |
| Config | full standalone pilot YAML | current dataclass schema plus relative `extends` |
| PE selection | `model.name: positional`, `pos_encoding` | retained; no task-specific PE switch |
| Checkpoint | same-run strict resume | strict resume plus explicit external continuation |
| Optimizer | whole state dict only | whole-state resume plus name-based state transfer |
| Evaluation | validation during training | fixed ID/OOD seed and length table from the launcher |
| Tracking | local JSONL/summary and basic MLflow sink | provenance, continuation links, evaluation metrics and artifacts |

`models/positional.py` remains the single implementation for NoPE, WPE, RoPE,
ALiBi, relative bias, CoPE, CAPE and FoPE. The validation added by the EVE-PE
pilot was retained. Obsolete task arguments and the old depth-calibration
matrix were not copied.

## Continuation semantics

`train.init: checkpoint` loads the file named by `train.checkpoint_path` into a
new run directory. Every non-positional architecture field must match. Every
non-positional state-dict tensor must exist with the same shape. Positional
keys may be added or removed.

Optimizer moments are checkpointed by parameter name. Shared Transformer
blocks, token embeddings and output head retain their moments. New PE-specific
parameters, such as WPE embeddings, start with empty optimizer state. Removed
PE-specific parameters are discarded. RNG and task-generator state are
restored, so stage 2 continues the stage-1 sample stream. Stage 2 retains the
global step but resets its local best-validation checkpoint criterion.

The four-regime launcher uses an equal total budget. Single-stage NoPE and RoPE
train for `max_iters`. Transition runs train the source PE to `switch_iter`,
then train the target PE until the same global `max_iters`.

## MLflow

Set a standard MLflow tracking URI. Authentication variables supported by the
MLflow client can be set normally. `MLFLOW_WORKSPACE` is optional and is used
only by a workspace-enabled client.

```bash
export MLFLOW_TRACKING_URI=https://your-mlflow-server
export MLFLOW_EXPERIMENT_NAME=transformer-architecture-generalization
```

Every training stage is a separate MLflow run. Stage 2 carries the
`mlflow.parentRunId` tag and the parameters `source_run_id`,
`source_checkpoint`, `source_positional_encoding`, and `source_global_step`.
Large checkpoints are not uploaded by default; set
`MLFLOW_LOG_CHECKPOINTS=1` to upload `last.pt` and `best.pt`.

## KV pilot

Preview one run without training:

```bash
python main.py --config configs/pe_experiments/kv_retrieval/rope.yaml --dry-run
```

Run the required four regimes on seed 1337, including fixed ID/OOD
evaluation at lengths 64, 128 and 256:

```bash
python scripts/run_pe_kv_pilot.py \
  --seeds 1337 \
  --regimes nope rope rope_to_nope nope_to_rope
```

Optional WPE and ALiBi controls:

```bash
python scripts/run_pe_kv_pilot.py --regimes wpe alibi --seeds 1337
```

Existing completed runs are reused, while an incomplete non-empty directory is
refused. Pass `--rerun` to overwrite run outputs deliberately. The launcher
prints every MLflow run ID and writes `manifest.json` under the selected
`--run-root`.

Direct two-command continuation is also supported:

```bash
python main.py --config configs/pe_experiments/kv_retrieval/rope.yaml \
  --set train.stage=stage_1 --set train.max_iters=7500 \
  --set paths.run_dir=runs/manual/rope_stage1

python main.py --config configs/pe_experiments/kv_retrieval/rope_to_nope.yaml \
  --set train.checkpoint_path=runs/manual/rope_stage1/last.pt \
  --set train.source_checkpoint=runs/manual/rope_stage1/last.pt \
  --set train.max_iters=15000 \
  --set paths.run_dir=runs/manual/rope_to_nope
```

## Analysis

The analysis reads runs and evaluation artifacts from MLflow, groups by task,
regime, seed and sequence length, and writes aggregate CSV files, mean ± std
accuracy curves, mean ± std loss curves, and pre/post-switch degradation.

```bash
python scripts/analyze_pe_mlflow.py \
  --experiment transformer-architecture-generalization \
  --task kv_retrieval
```

The analysis outputs are also logged to a dedicated MLflow analysis run.

## Additional task templates

`c5/`, `s5/`, and `state_based_recall/` contain the same
base/NoPE/RoPE/transition structure using the current task APIs. They are
intentionally templates until the KV pilot validates the protocol; no broad
task sweep is launched automatically. `relative_offset_copy` and `indexing`
are not registered tasks in the current `main`, so no fictitious configs are
added for them.
