#!/usr/bin/env python3
"""Run the reproducible KV positional-encoding pilot and its fixed evaluations."""

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs/pe_experiments/kv_retrieval"
DEFAULT_REGIMES = ("nope", "rope", "rope_to_nope", "nope_to_rope")
VALID_REGIMES = (*DEFAULT_REGIMES, "wpe", "alibi")
TRANSITIONS = {
    "rope_to_nope": ("rope", "nope"),
    "nope_to_rope": ("nope", "rope"),
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=[1337])
    parser.add_argument("--regimes", nargs="+", choices=VALID_REGIMES,
                        default=list(DEFAULT_REGIMES))
    parser.add_argument("--run-root", type=Path,
                        default=Path("runs/pe_experiments/kv_retrieval/pilot"))
    parser.add_argument("--max-iters", type=int, default=15000,
                        help="total optimizer steps for every final regime")
    parser.add_argument("--switch-iter", type=int, default=7500,
                        help="global step at which transition regimes switch PE")
    parser.add_argument("--eval-lengths", nargs="+", type=int, default=[64, 128, 256])
    parser.add_argument("--eval-seed", type=int, default=987654)
    parser.add_argument("--n-eval", type=int, default=2000)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--eval-device", default="auto")
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-no-mlflow", action="store_true",
                        help="explicitly permit local-only runs")
    args = parser.parse_args()
    if not 0 < args.switch_iter < args.max_iters:
        parser.error("require 0 < --switch-iter < --max-iters")
    if not args.allow_no_mlflow and not os.environ.get("MLFLOW_TRACKING_URI"):
        parser.error("MLFLOW_TRACKING_URI is required (or pass --allow-no-mlflow)")
    return args


def run_dir(root, regime, seed):
    return (root / regime / f"seed={seed}").resolve()


def run_id(directory):
    path = directory / "mlflow-run-id"
    return path.read_text(encoding="utf-8").strip() if path.is_file() else ""


def assert_writable(directory, rerun):
    if rerun or not directory.exists():
        return
    if (directory / "summary.csv").is_file():
        return
    if any(directory.iterdir()):
        raise RuntimeError(
            f"refusing to overwrite incomplete run {directory}; use --rerun explicitly"
        )


def main_command(config, directory, seed, max_iters, stage, regime, extra=()):
    return [
        sys.executable, str(ROOT / "main.py"), "--config", str(config),
        "--set", f"train.seed={seed}",
        "--set", "train.data_seed=424242",
        "--set", f"train.max_iters={max_iters}",
        "--set", f"optimizer.lr_decay_iters={max_iters}",
        "--set", f"train.stage={stage}",
        "--set", f"train.regime={regime}",
        "--set", f"paths.run_dir={directory}",
        *extra,
    ]


def execute(command, directory, rerun, dry_run, name):
    assert_writable(directory, rerun)
    if rerun:
        command.append("--rerun")
    print("+", " ".join(command), flush=True)
    if dry_run:
        return ""
    environment = os.environ.copy()
    environment["MLFLOW_RUN_NAME"] = name
    subprocess.run(command, cwd=ROOT, env=environment, check=True)
    mlflow_id = run_id(directory)
    print(f"{name}: MLflow run id = {mlflow_id or '(MLflow disabled)'}", flush=True)
    return mlflow_id


def ensure_source(args, source_encoding, seed, manifest):
    logical = f"stage1_{source_encoding}"
    directory = run_dir(args.run_root, logical, seed)
    command = main_command(
        CONFIGS / f"{source_encoding}.yaml", directory, seed,
        args.switch_iter, "stage_1", logical,
        extra=("--set", f"optimizer.lr_decay_iters={args.max_iters}"),
    )
    mlflow_id = execute(command, directory, args.rerun, args.dry_run,
                        f"kv-{logical}-seed-{seed}")
    entry = {
        "task": "kv_retrieval", "regime": logical, "seed": seed,
        "stage": "stage_1", "positional_encoding": source_encoding,
        "run_dir": str(directory), "mlflow_run_id": mlflow_id,
        "checkpoint": str(directory / "last.pt"),
        "global_step": args.switch_iter,
    }
    manifest.append(entry)
    evaluation_exists = (directory / "id_ood_evaluation.csv").is_file()
    if (not args.dry_run and not args.skip_eval
            and (args.rerun or not evaluation_exists)):
        evaluate_run(directory, args)
    return entry


def evaluate_run(directory, args):
    sys.path.insert(0, str(ROOT))
    import evaluate

    device = evaluate.pick_device(args.eval_device)
    rows = []
    for length in args.eval_lengths:
        started = time.time()
        namespace = SimpleNamespace(
            checkpoint=["last.pt"], task=None,
            params=repr({"input_seq_len": length}),
            label="id" if length == args.eval_lengths[0] else "ood",
            n_eval=args.n_eval, seed=args.eval_seed,
            batch_size=args.eval_batch_size, device=device, dtype=None,
            autoregressive=False, by=None, bins=0, range=None,
        )
        produced = evaluate.evaluate_run(directory, namespace, device, "last.pt")
        elapsed = time.time() - started
        for row in produced:
            row["sequence_length"] = length
            row["evaluation_time_seconds"] = elapsed
        rows.extend(produced)
    table = evaluate.write_table(rows, directory / "id_ood_evaluation.csv")
    plot = plot_accuracy(rows, directory / "accuracy-vs-length.png")
    log_evaluation(directory, rows, (table, plot))
    return rows


def plot_accuracy(rows, path):
    ordered = sorted(rows, key=lambda row: int(row["sequence_length"]))
    lengths = [int(row["sequence_length"]) for row in ordered]
    figure, axis = plt.subplots(figsize=(6.5, 4.2))
    for metric, label in (("acc", "sequence accuracy"),
                          ("token_acc", "token accuracy")):
        if all(metric in row for row in ordered):
            axis.plot(lengths, [float(row[metric]) for row in ordered], marker="o", label=label)
    axis.axvline(lengths[0], color="black", linestyle="--", alpha=0.35, label="train length")
    axis.set(xlabel="sequence length", ylabel="accuracy", ylim=(-0.02, 1.02))
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path


def configure_mlflow():
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        return None
    import mlflow
    mlflow.set_tracking_uri(tracking_uri)
    workspace = os.environ.get("MLFLOW_WORKSPACE")
    if workspace:
        if not hasattr(mlflow, "set_workspace"):
            raise RuntimeError("MLFLOW_WORKSPACE requires a workspace-enabled MLflow client")
        mlflow.set_workspace(workspace)
    return mlflow


def log_evaluation(directory, rows, artifacts):
    mlflow = configure_mlflow()
    mlflow_id = run_id(directory)
    if mlflow is None or not mlflow_id:
        return
    client = mlflow.tracking.MlflowClient()
    step = int(rows[0].get("iter") or 0)
    for row in rows:
        length = int(row["sequence_length"])
        prefix = "id" if row["label"] == "id" else "ood"
        excluded = {"iter", "n", "sequence_length"}
        for key, value in row.items():
            if key in excluded or isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            client.log_metric(mlflow_id, f"{prefix}_{key}_length_{length}",
                              float(value), step=step)
    for path in artifacts:
        client.log_artifact(mlflow_id, str(path), artifact_path="evaluation")


def write_manifest(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main():
    args = parse_args()
    args.run_root = (ROOT / args.run_root).resolve() if not args.run_root.is_absolute() else args.run_root
    entries = []
    sources = {}
    for seed in args.seeds:
        for regime in args.regimes:
            if regime in TRANSITIONS:
                source_encoding, target_encoding = TRANSITIONS[regime]
                key = (source_encoding, seed)
                if key not in sources:
                    sources[key] = ensure_source(args, source_encoding, seed, entries)
                source = sources[key]
                directory = run_dir(args.run_root, regime, seed)
                checkpoint_path = Path(source["checkpoint"])
                extra = (
                    "--set", f"train.checkpoint_path={checkpoint_path}",
                    "--set", f"train.source_checkpoint={checkpoint_path}",
                    "--set", f"train.source_run_id={source['mlflow_run_id']}",
                    "--set", f"train.source_global_step={args.switch_iter}",
                )
                command = main_command(
                    CONFIGS / f"{regime}.yaml", directory, seed,
                    args.max_iters, "stage_2", regime, extra=extra,
                )
                encoding = target_encoding
            else:
                directory = run_dir(args.run_root, regime, seed)
                command = main_command(
                    CONFIGS / f"{regime}.yaml", directory, seed,
                    args.max_iters, "single", regime,
                )
                encoding = regime
            mlflow_id = execute(command, directory, args.rerun, args.dry_run,
                                f"kv-{regime}-seed-{seed}")
            entry = {
                "task": "kv_retrieval", "regime": regime, "seed": seed,
                "stage": "stage_2" if regime in TRANSITIONS else "single",
                "positional_encoding": encoding, "run_dir": str(directory),
                "mlflow_run_id": mlflow_id,
                "checkpoint": str(directory / "last.pt"),
                "global_step": args.max_iters,
            }
            entries.append(entry)
            evaluation_exists = (directory / "id_ood_evaluation.csv").is_file()
            if (not args.dry_run and not args.skip_eval
                    and (args.rerun or not evaluation_exists)):
                evaluate_run(directory, args)
            write_manifest(args.run_root / "manifest.json", {
                "task": "kv_retrieval", "seeds": args.seeds,
                "regimes": args.regimes, "max_iters": args.max_iters,
                "switch_iter": args.switch_iter,
                "evaluation": {"lengths": args.eval_lengths,
                               "seed": args.eval_seed, "n": args.n_eval},
                "runs": entries,
            })
    print(f"manifest: {args.run_root / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
