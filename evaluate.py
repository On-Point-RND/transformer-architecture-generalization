#!/usr/bin/env python3
"""Score trained checkpoints, optionally on a shifted distribution.

    python evaluate.py runs/EVE-PE/pe_lab/pos_encoding=rope__seed=1337
    python evaluate.py runs/EVE-PE/pe_lab/* --params "{'length_range': (33, 64)}" --label OOD-1
    python evaluate.py runs/kv-run --task nested_kv --label nested

The model and the task come from the checkpoint, so an evaluation reproduces the
run's own validation set by default (same task, same params, same data_seed).
Overriding --task or --params is what makes it an OOD measurement: everything
else stays as trained.

Metrics are whatever the task reports (see Task.metrics), so a task that scores
itself specially is scored the same way here as during training. Rows are
appended to <run_dir>/evaluations.csv and printed; -o also writes one combined
table for all runs.
"""

import argparse
import csv
from ast import literal_eval
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

from core import checkpoint
from models import get_model
from tasks import get_task

DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
CHECKPOINTS = ("best.pt", "last.pt", "ckpt.pt") 
ROW_FIELDS = ("run", "checkpoint", "iter", "model", "positional_encoding", "task",
              "label", "scoring", "slice", "params", "n", "loss")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--checkpoint", help=f"default: first of {', '.join(CHECKPOINTS)}")
    parser.add_argument("--task", help="evaluate on another task (params are inherited)")
    parser.add_argument("--params", default="{}",
                        help="task params to override, e.g. \"{'length_range': (33, 64)}\"")
    parser.add_argument("--label", help="name for this evaluation in the output row")
    parser.add_argument("-n", "--n-eval", type=int, default=2000)
    parser.add_argument("--seed", type=int, help="default: the run's data_seed")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="auto", help="auto | cpu | cuda | cuda:N")
    parser.add_argument("--dtype", help="default: the dtype the run used")
    parser.add_argument("--autoregressive", action="store_true",
                        help="decode the answer token by token instead of scoring a "
                             "teacher-forced argmax")
    parser.add_argument("--by", help="metadata field to break the score down by, "
                                     "e.g. normalized_target_position")
    parser.add_argument("--bins", type=int, default=0,
                        help="equal-width bins for --by; 0 groups by exact value")
    parser.add_argument("--range", type=float, nargs=2, metavar=("LO", "HI"),
                        help="bin edges for --by; default: the observed range")
    parser.add_argument("-o", "--out", type=Path, help="combined table for all runs")
    return parser.parse_args()


def pick_checkpoint(run_dir, requested):
    names = [requested] if requested else list(CHECKPOINTS)
    found = [name for name in names if (Path(run_dir) / name).is_file()]
    if not found:
        raise FileNotFoundError(f"no {' / '.join(names)} in {run_dir}")
    return found[0]


def load_model(run_dir, name, device):
    saved = checkpoint.load(run_dir, name, device)
    fields = dict(checkpoint.config_sections(saved)["model"])
    Config, Model = get_model(fields.pop("name", "positional"))
    model = Model(Config(**fields))
    model.load_state_dict(checkpoint.strip_compile_prefix(saved["model"]))
    return model.eval().to(device), saved


def build_task(sections, task_name, overrides, seed):
    section = sections["task"]
    params = {**(section.get("params") or {}), **overrides}
    if seed is None:
        seed = sections["train"].get("data_seed", 42)
    return get_task(task_name or section["name"], {**params, "seed": seed}), params


def group_items(items, field, bins, edges):
    """[(slice label, items)] split by a metadata field; one group without --by.

    Grouping happens before scoring, so a slice is scored by exactly the same
    ``Task.metrics`` as the whole set — no per-example metric interface needed.
    """
    if not field:
        return [("all", items)]
    missing = [i for i in items if field not in i.metadata]
    if missing:
        raise KeyError(f"the task records no {field!r}; it has "
                       f"{sorted(items[0].metadata)}")
    values = [item.metadata[field] for item in items]
    if not bins:
        keys = sorted(set(values))
        return [(f"{field}={key}", [i for i, v in zip(items, values) if v == key])
                for key in keys]
    lo, hi = edges if edges else (min(values), max(values))
    width = (hi - lo) / bins or 1.0
    labels = [f"{field}[{lo + b * width:.3g},{lo + (b + 1) * width:.3g})"
              for b in range(bins)]
    index = [min(int((v - lo) / width), bins - 1) for v in values]
    groups = [(labels[b], [i for i, k in zip(items, index) if k == b]) for b in range(bins)]
    return [(label, picked) for label, picked in groups if picked]


@torch.no_grad()
def decode_group(model, items, device, ctx):
    """Greedy continuation for prompts that all share one length."""
    tokens = torch.from_numpy(np.stack([i.prompt for i in items])).long().to(device)
    produced = []
    for _ in range(max(len(i.answer) for i in items)):
        with ctx:
            logits, _ = model(tokens[:, -model.config.block_size:])
        nxt = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        produced.append(nxt)
        tokens = torch.cat([tokens, nxt], dim=1)
    return torch.cat(produced, dim=1).cpu().numpy()


def greedy_predictions(model, items, y, device, ctx):
    """Decoded answers laid out exactly like a teacher-forced argmax would be.

    Teacher forcing feeds the true answer prefix back in, so a multi-token answer
    is scored optimistically: one wrong token does not derail the rest. Here the
    model only ever sees its own output. Prompts are grouped by length because a
    causal model cannot be left-padded without shifting every position, so rows
    in one batch must have their answer start at the same index.
    """
    predicted = np.zeros_like(y)
    by_length = defaultdict(list)
    for index, item in enumerate(items):
        by_length[len(item.prompt)].append(index)
    for indices in by_length.values():
        generated = decode_group(model, [items[i] for i in indices], device, ctx)
        for row, index in enumerate(indices):
            answer_len = len(items[index].answer)
            end = len(items[index].prompt) + answer_len - 1
            predicted[index, end - answer_len:end] = generated[row, :answer_len]
    return predicted


@torch.no_grad()
def score(model, task, items, batch_size, device, ctx, autoregressive=False):
    """Loss and the task's metrics over the given examples, in batches.

    Batched on purpose: one forward over thousands of rows materialises a
    [rows, n_head, T, T] score matrix for the mechanisms that need explicit
    scores, which does not fit in memory.
    """
    block_size = model.config.block_size
    totals, rows = {"loss": 0.0}, 0
    for start in range(0, len(items), batch_size):
        chunk = items[start:start + batch_size]
        x_np, y_np = task.collate(chunk, block_size)
        x, y = torch.from_numpy(x_np).to(device), torch.from_numpy(y_np).to(device)
        with ctx:
            logits, loss = model(x, y)
        predicted = (greedy_predictions(model, chunk, y_np, device, ctx) if autoregressive
                     else logits.argmax(dim=-1).cpu().numpy())
        scores = {"loss": loss.item(), **task.metrics(predicted, y_np)}
        totals = {k: totals.get(k, 0.0) + v * len(chunk) for k, v in scores.items()}
        rows += len(chunk)
    return {"n": rows, **{key: value / rows for key, value in totals.items()}}


def pick_device(requested):
    if requested != "auto":
        return requested
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def evaluate_run(run_dir, args, device):
    name = pick_checkpoint(run_dir, args.checkpoint)
    model, saved = load_model(run_dir, name, device)
    sections = checkpoint.config_sections(saved)
    overrides = literal_eval(args.params)
    task, params = build_task(sections, args.task, overrides, args.seed)
    dtype = args.dtype or sections["hardware"].get("dtype", "float32")
    ctx = (nullcontext() if "cuda" not in device else
           torch.amp.autocast(device_type="cuda", dtype=DTYPES[dtype]))
    metadata = saved.get("run_metadata", {})
    shifted = bool(overrides) or bool(args.task)
    row = {
        "run": Path(run_dir).as_posix(),
        "checkpoint": name,
        "iter": saved.get("iter_num"),
        "model": metadata.get("model", sections["model"].get("name")),
        "positional_encoding": metadata.get("positional_encoding", ""),
        "task": args.task or sections["task"]["name"],
        "label": args.label or ("ood" if shifted else "id"),
        "scoring": "autoregressive" if args.autoregressive else "teacher_forced",
        "params": repr(params),
    }
    items = task.generate_val(args.n_eval)
    groups = group_items(items, args.by, args.bins, args.range)
    return [{**row, "slice": label,
             **score(model, task, chunk, args.batch_size, device, ctx, args.autoregressive)}
            for label, chunk in groups]


def write_table(rows, path):
    """Append rows, rewriting the file so a new metric adds a column instead of
    sliding every value one place to the left."""
    path = Path(path)
    existing = list(csv.DictReader(path.open(encoding="utf-8"))) if path.is_file() else []
    everything = existing + [{k: v for k, v in row.items()} for row in rows]
    fields = list(dict.fromkeys(list(ROW_FIELDS) + [k for r in everything for k in r]))
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(everything)
    return path


def main():
    args = parse_args()
    device = pick_device(args.device)
    rows, failures = [], []
    for run_dir in args.runs:
        try:
            produced = evaluate_run(run_dir, args, device)
        except Exception as error:  
            failures.append(run_dir)
            print(f"FAILED {run_dir}: {type(error).__name__}: {error}", flush=True)
            continue
        rows += produced
        for row in produced:
            scores = " ".join(f"{k}={v:.4f}" for k, v in row.items() if isinstance(v, float))
            print(f"{row['positional_encoding'] or row['model']:22s} {row['label']:10s} "
                  f"{row['slice']:34s} n={row['n']:<5} {scores}", flush=True)
        write_table(produced, Path(run_dir) / "evaluations.csv")
    if args.out and rows:
        print(f"combined table: {write_table(rows, args.out)}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
