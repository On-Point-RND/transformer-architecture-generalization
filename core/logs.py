"""Run logging: one JSONL of every metric, a one-row summary CSV, and curves.

    <logs>/metrics.jsonl        one record per train tick / eval / diagnostic
    <results>/summary.csv       one row: config + latest and best metrics
    <results>/curves.png        loss, accuracy, lr and grad norm, redrawn at eval

where <logs> and <results> come from the paths section of the config and both
default to the run directory.

No wandb, no tensorboard: the JSONL is the source of truth and every plot is
regenerated from it (plot.py).
"""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # cluster nodes have no display
import matplotlib.pyplot as plt  # noqa: E402

# substring of the parameter name -> reporting group, first match wins
GROUPS = (
    ("positioning", "pe"),
    ("wte", "embedding"),
    ("wpe", "embedding"),
    ("attn.c_attn", "attn.qkv"),
    ("attn.c_proj", "attn.proj"),
    ("mlp.", "mlp"),
    ("ln_", "norm"),
    ("lm_head", "head"),
)


def group_of(name):
    for needle, group in GROUPS:
        if needle in name:
            return group
    return "other"


def parameter_norms(model):
    """L2 norm of the weights and of the gradients, per reporting group.

    Weights and gradients only — no forward pass, no extra batch. Pulling one
    more batch here would advance the task's stream and change what the model
    trains on. Everything is detached, so this module stays torch-free at import
    time and plot.py runs anywhere.
    """
    weight, grad = {}, {}
    for name, parameter in model.named_parameters():
        group = group_of(name)
        weight[group] = weight.get(group, 0.0) + parameter.detach().float().pow(2).sum().item()
        if parameter.grad is None:
            continue
        grad[group] = grad.get(group, 0.0) + parameter.grad.detach().float().pow(2).sum().item()
    return {"weight_norm": {k: v ** 0.5 for k, v in weight.items()},
            "grad_norm": {k: v ** 0.5 for k, v in grad.items()}}


class RunLogger:
    """Append-only JSONL plus a summary row and refreshed curves."""

    def __init__(self, paths, metadata, resume=False):
        self.logs_dir, self.results_dir = Path(paths.logs), Path(paths.results)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.logs_dir / "metrics.jsonl"
        self.metadata = metadata
        self.history = []
        if not resume:
            self.metrics_path.write_text("", encoding="utf-8")

    def log(self, event, **fields):
        record = {"event": event, **fields}
        self.history.append(record)
        with self.metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=float) + "\n")

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
        with (self.results_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row))
            writer.writeheader()
            writer.writerow({k: json.dumps(v) if isinstance(v, dict) else v
                             for k, v in row.items()})

    def write_curves(self):
        write_curves(self.metrics_path, self.results_dir / "curves.png",
                     title=Path(self.logs_dir).name)


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
    ("exact-match accuracy", (("eval", "train_acc", "train"), ("eval", "val_acc", "val"))),
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
