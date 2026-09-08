import argparse
import csv
from ast import literal_eval
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from evaluate import DTYPES, build_task, load_model, pick_checkpoint, pick_device
from core import checkpoint

ROW_FIELDS = ("run", "checkpoint", "model", "task", "label", "mode", "layer",
              "metric", "value")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--mode", default="linear", choices=("linear", "readout", "stats"))
    parser.add_argument("--checkpoint", help="default: best.pt, then last.pt")
    parser.add_argument("--task", help="probe on another task (params are inherited)")
    parser.add_argument("--params", default="{}", help="task params to override")
    parser.add_argument("--label", help="name for this probe in the output row")
    parser.add_argument("-n", "--n-eval", type=int, default=2000)
    parser.add_argument("--seed", type=int, help="default: the run's data_seed")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--probe-steps", type=int, default=400)
    parser.add_argument("--probe-lr", type=float, default=1e-2)
    parser.add_argument("--sparsity-threshold", type=float, default=0.01)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", help="default: the dtype the run used")
    parser.add_argument("-o", "--out", type=Path, help="combined table for all runs")
    return parser.parse_args()


def capture_layers(model, x, ctx):
    """Residual stream after every block, in the order the blocks actually run.

    Hooks rather than a bespoke forward, so this works for any architecture in
    models/. A weight-shared model lists one block object several times, so the
    hook is registered once per distinct module and fires once per application --
    which gives one capture per iteration, exactly what a looped model should
    report. Registering per list entry instead would fire every hook on every
    call and return n_loops squared captures.
    """
    captured, seen, handles = [], set(), []
    for block in model.transformer.h:
        if id(block) in seen:
            continue
        seen.add(id(block))
        handles.append(block.register_forward_hook(
            lambda _module, _inputs, output: captured.append(output)))
    try:
        with torch.no_grad(), ctx:
            model(x)
    finally:
        for handle in handles:
            handle.remove()
    return captured


def answer_positions(y):
    """(rows, columns, tokens) of the supervised positions, read off the targets.

    Derived from the mask rather than recomputed from prompt/answer lengths, so
    it cannot drift away from Task.collate.
    """
    rows, cols = np.nonzero(y != -1)
    return rows, cols, y[rows, cols]


def gather(model, task, items, args, device, ctx):
    """(features per layer at the answer positions, labels, per-layer stats).

    Reduced batch by batch: keeping every activation would be
    n_eval * block_size * n_embd * n_layer floats, gigabytes for a normal run,
    while the probes only ever need the answer positions and running moments.
    """
    block_size = model.config.block_size
    features, labels, positions = None, [], 0
    below, sums, squares = None, None, None
    for start in range(0, len(items), args.batch_size):
        chunk = items[start:start + args.batch_size]
        x_np, y_np = task.collate(chunk, block_size)
        layers = capture_layers(model, torch.from_numpy(x_np).to(device), ctx)
        if features is None:
            features = [[] for _ in layers]
            below, sums, squares = [0.0] * len(layers), [0.0] * len(layers), [0.0] * len(layers)
        rows, cols, tokens = answer_positions(y_np)
        labels.append(tokens)
        lengths = np.array([len(i.prompt) + len(i.answer) - 1 for i in chunk])
        real = torch.from_numpy(np.arange(block_size)[None, :] < lengths[:, None]).to(device)
        positions += int(lengths.sum())
        for index, activation in enumerate(layers):
            features[index].append(activation[rows, cols].float().cpu().numpy())
            live = activation[real].float()  
            below[index] += float((live.abs() < args.sparsity_threshold).sum())
            sums[index] += live.sum(0).cpu().numpy()
            squares[index] += live.pow(2).sum(0).cpu().numpy()
    width = len(sums[0])
    stats = [{"sparsity": below[i] / (positions * width),
              "variance": float(np.mean(squares[i] / positions - (sums[i] / positions) ** 2))}
             for i in range(len(features))]
    return [np.concatenate(f) for f in features], np.concatenate(labels), stats


def train_linear_probe(features, labels, vocab_size, args, device):
    """Fit a linear read-out on frozen features; return held-out accuracy."""
    generator = torch.Generator().manual_seed(0)
    order = torch.randperm(len(labels), generator=generator)
    split = int(0.8 * len(labels))
    train_idx, test_idx = order[:split].to(device), order[split:].to(device)
    x = torch.from_numpy(features).float().to(device)
    y = torch.from_numpy(labels).long().to(device)
    probe = torch.nn.Linear(x.size(1), vocab_size).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=args.probe_lr)
    for _ in range(args.probe_steps):
        loss = F.cross_entropy(probe(x[train_idx]), y[train_idx])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        return float((probe(x[test_idx]).argmax(-1) == y[test_idx]).float().mean())


@torch.no_grad()
def head_readout(model, features, labels, device):
    """The model's own final norm and head, applied to an intermediate layer."""
    x = torch.from_numpy(features).to(device).to(model.lm_head.weight.dtype)
    predicted = model.lm_head(model.transformer.ln_f(x)).argmax(-1).cpu().numpy()
    return float((predicted == labels).mean())


def probe_run(run_dir, args, device):
    name = pick_checkpoint(run_dir, args.checkpoint)
    model, saved = load_model(run_dir, name, device)
    sections = checkpoint.config_sections(saved)
    overrides = literal_eval(args.params)
    task, params = build_task(sections, args.task, overrides, args.seed)
    dtype = args.dtype or sections["hardware"].get("dtype", "float32")
    ctx = (nullcontext() if "cuda" not in device else
           torch.amp.autocast(device_type="cuda", dtype=DTYPES[dtype]))
    items = task.generate_val(args.n_eval)
    features, labels, stats = gather(model, task, items, args, device, ctx)

    row = {"run": Path(run_dir).as_posix(), "checkpoint": name,
           "model": sections["model"].get("name"),
           "task": args.task or sections["task"]["name"],
           "label": args.label or ("ood" if overrides or args.task else "id"),
           "mode": args.mode}
    if args.mode == "stats":
        return [{**row, "layer": i, "metric": metric, "value": value}
                for i, per_layer in enumerate(stats)
                for metric, value in per_layer.items()]
    if args.mode == "readout":
        return [{**row, "layer": i, "metric": "acc",
                 "value": head_readout(model, f, labels, device)}
                for i, f in enumerate(features)]
    return [{**row, "layer": i, "metric": "acc",
             "value": train_linear_probe(f, labels, model.config.vocab_size, args, device)}
            for i, f in enumerate(features)]


def write_table(rows, path):
    path = Path(path)
    existing = list(csv.DictReader(path.open(encoding="utf-8"))) if path.is_file() else []
    everything = existing + rows
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(ROW_FIELDS))
        writer.writeheader()
        writer.writerows(everything)
    return path


def main():
    args = parse_args()
    device = pick_device(args.device)
    rows, failures = [], []
    for run_dir in args.runs:
        try:
            produced = probe_run(run_dir, args, device)
        except Exception as error:  
            failures.append(run_dir)
            print(f"FAILED {run_dir}: {type(error).__name__}: {error}", flush=True)
            continue
        rows += produced
        for r in produced:
            print(f"{r['run']:44s} {r['mode']:8s} layer {r['layer']:<3} "
                  f"{r['metric']:9s} {r['value']:.4f}", flush=True)
        write_table(produced, Path(run_dir) / "probe.csv")
    if args.out and rows:
        print(f"combined table: {write_table(rows, args.out)}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
