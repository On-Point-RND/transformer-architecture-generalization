#!/usr/bin/env python3
"""Aggregate PE-pilot curves and fixed-length evaluations directly from MLflow."""

import argparse
import csv
import json
import os
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", default=os.environ.get(
        "MLFLOW_EXPERIMENT_NAME", "transformer-architecture-generalization"))
    parser.add_argument("--task", default="kv_retrieval")
    parser.add_argument("--out", type=Path, default=Path("analysis/pe_kv_mlflow"))
    parser.add_argument("--run-name", default="pe-kv-aggregate-analysis")
    parser.add_argument("--no-log-analysis", action="store_true")
    return parser.parse_args()


def configure_mlflow():
    uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not uri:
        raise RuntimeError("MLFLOW_TRACKING_URI is required")
    import mlflow
    mlflow.set_tracking_uri(uri)
    workspace = os.environ.get("MLFLOW_WORKSPACE")
    if workspace:
        if not hasattr(mlflow, "set_workspace"):
            raise RuntimeError("MLFLOW_WORKSPACE requires a workspace-enabled MLflow client")
        mlflow.set_workspace(workspace)
    return mlflow


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row)) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        if fields:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    return path


def read_evaluations(client, run, cache):
    try:
        local = Path(client.download_artifacts(
            run.info.run_id, "evaluation/id_ood_evaluation.csv",
            str(cache / run.info.run_id)
        ))
    except Exception:
        return []
    rows = list(csv.DictReader(local.open(encoding="utf-8")))
    regime = run.data.params.get("regime") or run.data.params.get("train.regime", "")
    seed = run.data.params.get("seed") or run.data.params.get("train.seed", "")
    stage = (run.data.params.get("training_stage")
             or run.data.params.get("train.stage", "single"))
    return [{**row, "run_id": run.info.run_id, "regime": regime,
             "seed": int(seed), "stage": stage} for row in rows]


def aggregate(rows):
    grouped = defaultdict(list)
    for row in rows:
        length = int(row["sequence_length"])
        for metric in ("acc", "token_acc", "loss"):
            if row.get(metric, "") != "":
                grouped[(row["regime"], length, metric)].append(float(row[metric]))
    output = []
    for (regime, length, metric), values in sorted(grouped.items()):
        output.append({
            "task": rows[0].get("task", "") if rows else "",
            "regime": regime, "sequence_length": length, "metric": metric,
            "n_seeds": len(values), "mean": statistics.fmean(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        })
    return output


def plot_accuracy(summary, path):
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharex=True, sharey=True)
    for axis, metric in zip(axes, ("acc", "token_acc")):
        regimes = sorted({row["regime"] for row in summary if row["metric"] == metric
                          and not row["regime"].startswith("stage1_")})
        for regime in regimes:
            picked = sorted((row for row in summary
                             if row["regime"] == regime and row["metric"] == metric),
                            key=lambda row: row["sequence_length"])
            x = [row["sequence_length"] for row in picked]
            y = [row["mean"] for row in picked]
            e = [row["std"] for row in picked]
            axis.plot(x, y, marker="o", label=regime)
            axis.fill_between(x, [a - b for a, b in zip(y, e)],
                              [a + b for a, b in zip(y, e)], alpha=0.15)
        axis.set_title("sequence accuracy" if metric == "acc" else "token accuracy")
        axis.set(xlabel="sequence length", ylim=(-0.02, 1.02))
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("mean ± std across seeds")
    axes[1].legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path


def metric_histories(client, runs, metric):
    histories = []
    for run in runs:
        regime = run.data.params.get("regime") or run.data.params.get("train.regime", "")
        seed = run.data.params.get("seed") or run.data.params.get("train.seed", "")
        history = client.get_metric_history(run.info.run_id, metric)
        if history:
            histories.append((regime, int(seed), [(point.step, point.value) for point in history]))
    return histories


def plot_histories(client, runs, path):
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    for axis, metric, title in zip(
        axes, ("train_batch_loss", "val_loss"), ("training loss", "validation loss")
    ):
        grouped = defaultdict(lambda: defaultdict(list))
        for regime, seed, points in metric_histories(client, runs, metric):
            if not regime.startswith("stage1_"):
                for step, value in points:
                    grouped[regime][step].append(value)
        for regime, by_step in sorted(grouped.items()):
            steps = sorted(by_step)
            means = [statistics.fmean(by_step[step]) for step in steps]
            stds = [statistics.stdev(by_step[step]) if len(by_step[step]) > 1 else 0.0
                    for step in steps]
            axis.plot(steps, means, label=regime)
            axis.fill_between(steps, [a - b for a, b in zip(means, stds)],
                              [a + b for a, b in zip(means, stds)], alpha=0.15)
        axis.set(title=title, xlabel="global training step", ylabel="loss")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path


def degradation(rows):
    lookup = {(row["regime"], int(row["seed"]), int(row["sequence_length"])): row
              for row in rows}
    output = []
    for transition, source in (("rope_to_nope", "stage1_rope"),
                               ("nope_to_rope", "stage1_nope")):
        for (regime, seed, length), target in sorted(lookup.items()):
            if regime != transition or (source, seed, length) not in lookup:
                continue
            before = lookup[(source, seed, length)]
            for metric in ("acc", "token_acc"):
                output.append({
                    "regime": transition, "seed": seed, "sequence_length": length,
                    "metric": metric, "before_switch": float(before[metric]),
                    "after_stage_2": float(target[metric]),
                    "delta_after_minus_before": float(target[metric]) - float(before[metric]),
                })
    return output


def plot_degradation(rows, path):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["regime"], int(row["sequence_length"]), row["metric"])].append(
            float(row["delta_after_minus_before"])
        )
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharex=True)
    for axis, metric in zip(axes, ("acc", "token_acc")):
        for regime in ("rope_to_nope", "nope_to_rope"):
            points = sorted((length, values) for (name, length, key), values in grouped.items()
                            if name == regime and key == metric)
            if not points:
                continue
            x = [length for length, _ in points]
            y = [statistics.fmean(values) for _, values in points]
            e = [statistics.stdev(values) if len(values) > 1 else 0.0
                 for _, values in points]
            axis.errorbar(x, y, yerr=e, marker="o", capsize=3, label=regime)
        axis.axhline(0.0, color="black", linewidth=1, alpha=0.5)
        axis.set(title=metric, xlabel="sequence length",
                 ylabel="stage 2 minus stage 1 accuracy")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path


def main():
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    mlflow = configure_mlflow()
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(args.experiment)
    if experiment is None:
        raise ValueError(f"MLflow experiment not found: {args.experiment}")
    runs = client.search_runs(
        [experiment.experiment_id],
        filter_string=f"params.task = '{args.task}'",
        order_by=["attributes.start_time ASC"],
    )
    cache = args.out / ".mlflow-downloads"
    rows = [row for run in runs for row in read_evaluations(client, run, cache)]
    if not rows:
        raise RuntimeError("no PE evaluation artifacts found in the selected MLflow runs")
    summary = aggregate(rows)
    degradation_rows = degradation(rows)
    files = [
        write_csv(args.out / "all_evaluations.csv", rows),
        write_csv(args.out / "summary.csv", summary),
        write_csv(args.out / "degradation_after_switch.csv", degradation_rows),
        plot_accuracy(summary, args.out / "accuracy-vs-length.png"),
        plot_histories(client, runs, args.out / "loss-curves.png"),
        plot_degradation(degradation_rows, args.out / "degradation-after-switch.png"),
    ]
    (args.out / "analysis-metadata.json").write_text(json.dumps({
        "experiment": args.experiment, "task": args.task,
        "source_run_ids": [run.info.run_id for run in runs],
    }, indent=2) + "\n", encoding="utf-8")
    files.append(args.out / "analysis-metadata.json")
    if not args.no_log_analysis:
        mlflow.set_experiment(args.experiment)
        with mlflow.start_run(run_name=args.run_name,
                              tags={"task": args.task, "purpose": "pe-analysis"}) as run:
            mlflow.log_params({"task": args.task, "source_runs": len(runs)})
            for path in files:
                mlflow.log_artifact(str(path), artifact_path="analysis")
            print(f"analysis MLflow run id: {run.info.run_id}")
    print(f"analysis: {args.out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
