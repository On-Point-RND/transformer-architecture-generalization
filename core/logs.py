import csv
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg") 
import matplotlib.pyplot as plt  


def optional_mlflow_sink(config, logs_dir, metadata):
    """Build the optional sink without making MLflow a training dependency."""
    from core.mlflow_sink import MLflowSink
    return MLflowSink.from_environment(config, logs_dir, metadata)

def parameter_norms(model):

    module_types = {
        path: module.__class__.__name__
        for path, module in model.named_modules()
    }
    weight, grad = {"all": 0.0}, {"all": 0.0}
    for name, parameter in model.named_parameters():
        owner = name.rpartition(".")[0]
        group = module_types.get(owner, model.__class__.__name__)
        weight_sq = parameter.detach().float().pow(2).sum().item()
        weight["all"] += weight_sq
        weight[group] = weight.get(group, 0.0) + weight_sq
        if parameter.grad is None:
            continue
        grad_sq = parameter.grad.detach().float().pow(2).sum().item()
        grad["all"] += grad_sq
        grad[group] = grad.get(group, 0.0) + grad_sq
    return {"weight_norm": {k: v ** 0.5 for k, v in weight.items()},
            "grad_norm": {k: v ** 0.5 for k, v in grad.items()}}


class RunLogger:
    def __init__(self, paths, metadata, config, resume=False):
        self.logs_dir, self.results_dir = Path(paths.logs), Path(paths.results)
        self.checkpoints_dir = Path(paths.checkpoints)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.logs_dir / "metrics.jsonl"
        self.metadata = metadata
        self.history = []
        if not resume:
            self.metrics_path.write_text("", encoding="utf-8")
        self.mlflow = optional_mlflow_sink(config, self.logs_dir, metadata)

    def log(self, event, **fields):
        record = {"event": event, **fields}
        self.history.append(record)
        with self.metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=float) + "\n")
        if self.mlflow is not None:
            self.mlflow.log(event, fields)

    def log_eval(self, **fields):
        """Eval records also carry the run metadata, as the old jsonl log did."""
        self.log("eval", **fields, **self.metadata)

    def log_diagnostics(self, model, iter_num, every, eval_count):
        """Weight/grad norms every `every` evals; 0 disables."""
        if not every or eval_count % every:
            return
        self.log("diag", iter=iter_num, **parameter_norms(model))

    def write_summary(self, **fields):
        row = {**self.metadata, **fields}
        summary_path = self.results_dir / "summary.csv"
        temporary_path = self.results_dir / ".summary.csv.tmp"
        with temporary_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row))
            writer.writeheader()
            writer.writerow({k: json.dumps(v) if isinstance(v, dict) else v
                             for k, v in row.items()})
        temporary_path.replace(summary_path)
        if self.mlflow is not None:
            self.mlflow.log("summary", fields)

    def write_curves(self):
        write_curves(self.metrics_path, self.results_dir / "curves.png",
                     title=Path(self.logs_dir).name)

    def close(self, status="FINISHED"):
        if self.mlflow is None:
            return
        artifacts = [
                self.logs_dir / "config.resolved.yaml",
                self.logs_dir / "config.source.yaml",
                self.metrics_path,
                self.logs_dir / "checkpoint-metadata.json",
                self.logs_dir / "git-info.json",
                self.logs_dir / "git.diff",
                self.logs_dir / "RUN.md",
                self.results_dir / "summary.csv",
                self.results_dir / "curves.png",
        ]
        if os.environ.get("MLFLOW_LOG_CHECKPOINTS", "").lower() in {"1", "true", "yes"}:
            artifacts.extend((self.checkpoints_dir / "last.pt",
                              self.checkpoints_dir / "best.pt"))
        self.mlflow.close(artifacts, status=status)


def collect_summaries(run_dirs, out_path):
    """One table from every run's summary.csv — the ablation result of a grid.

    Runs that have not finished yet simply do not contribute a row.
    """
    rows = []
    for run_dir in run_dirs:
        path = Path(run_dir) / "summary.csv"
        rows += list(csv.DictReader(path.open(encoding="utf-8"))) if path.is_file() else []
    if not rows:
        return None
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(out_path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return out_path


def read_metrics(path):
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def series(records, event, x_key, y_key):
    picked = [(r[x_key], r[y_key]) for r in records
              if r.get("event") == event and r.get(y_key) is not None]
    return [x for x, _ in picked], [y for _, y in picked]


PANELS = (
    ("loss", (("eval", "train_loss", "train"), ("eval", "val_loss", "val"))),
    ("exact-match accuracy", (("eval", "train_acc", "train"), ("eval", "val_acc", "val"),
                               ("eval", "train_token_acc", "train token"), ("eval", "val_token_acc", "val token"))),
    ("learning rate", (("eval", "lr", "lr"),)),
    ("grad norm (global)", (("train", "grad_norm", "grad norm"),)),
)


def draw_panel(axis, records, ylabel, lines):
    drawn = 0
    for event, key, label in lines:
        xs, ys = series(records, event, "iter", key)
        drawn += 1 if xs else 0
        axis.plot(xs, ys, label=label, marker="." if len(xs) < 40 else None)
    axis.set_xlabel("iteration")
    axis.set_ylabel(ylabel)
    axis.grid(alpha=0.25)
    if drawn > 1:
        axis.legend(fontsize=8)


def write_curves(metrics_path, out_path, title=None):
    records = read_metrics(metrics_path)
    figure, axes = plt.subplots(2, 2, figsize=(11, 7))
    for axis, (ylabel, lines) in zip(axes.flat, PANELS):
        draw_panel(axis, records, ylabel, lines)
    figure.suptitle(title or Path(out_path).parent.name)
    figure.tight_layout()
    figure.savefig(out_path, dpi=120)
    plt.close(figure)
    return out_path
