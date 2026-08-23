"""Layer-truncation probe, task-agnostic.

For a trained model of depth L, run inference using only the first k blocks
(k = 1..L) and measure how long a dependency chain L(x) each such "sub-model"
can solve. Asks whether the model builds its answer LAYER BY LAYER.

Truncation = embedding -> blocks[:k] -> final LayerNorm -> tied head (via the
model's stop_at_layer forward arg). NOTE: the head/final-LN were trained for the
FULL depth, so a truncated read-out is an UNDERESTIMATE and this probe is
confounded ("the head only reads the final layer"). For the artifact-free
measurement, use probe_linear.py (trains a fresh read-out per layer). This script
is kept as the documented baseline that motivates the honest probe.

All task-specific behaviour comes from the shared ``Task`` interface, selected
from each checkpoint's own config; works for addition, permutation, sorting, ...

CSV schema (results/<out>_probe.csv):
  tag, task, train_variant, eval_variant, length, full_depth, d_model, seed,
  layers_used, <difficulty_col>, n_solved, n_total, exact_match
(<difficulty_col> = -1 -> overall exact-match).

Usage:
  python src/probe_layers.py --ckpts 'checkpoints/perm_grid_*_S5_*.pt' --out perm_grid --per-bin 200
  python src/probe_layers.py --ckpts 'checkpoints/perm_grid_*_S5_*.pt' --eval-variant C5   # transfer
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
from typing import Dict, List, Optional

import numpy as np
import torch

from probe_linear import load_model, pick_device  # shared schema-aware loader

RESULTS_DIR = "results"


@torch.no_grad()
def decode_ok(model, inputs, task, length, device, k, batch_size=512) -> np.ndarray:
    """Per-example correctness, decoding autoregressively with only the first k blocks."""
    a_start, a_len = task.answer_span(length)
    ok = []
    for i in range(0, inputs.size(0), batch_size):
        b = inputs[i:i + batch_size].to(device)
        gen = b[:, :a_start].clone()
        for _ in range(a_len):
            lg, _ = model(gen, stop_at_layer=k)
            gen = torch.cat([gen, lg[:, -1, :].argmax(dim=-1, keepdim=True)], dim=1)
        good = (gen[:, a_start:a_start + a_len] == b[:, a_start:a_start + a_len]).all(dim=1)
        ok.append(good.cpu().numpy())
    return np.concatenate(ok)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Layer-truncation probe. Settings from a YAML (--config) or "
                    "CLI flags (CLI overrides the config).")
    ap.add_argument("--config", default=None, help="YAML with keys: ckpts, out, per_bin, eval_variant")
    ap.add_argument("--ckpts", default=None, help="glob for checkpoints")
    ap.add_argument("--out", default=None, help="results/<out>_probe.csv")
    ap.add_argument("--per-bin", type=int, default=None)
    ap.add_argument("--eval-variant", "--eval-task", dest="eval_variant", default=None,
                    help="evaluate on THIS variant instead of the checkpoint's own "
                         "(e.g. --eval-variant C5 on S5-trained models)")
    args = ap.parse_args()

    conf = {}
    if args.config:
        import yaml
        conf = yaml.safe_load(open(args.config)) or {}
    ckpts = args.ckpts or conf.get("ckpts")
    out = args.out or conf.get("out", "probe")
    per_bin = args.per_bin if args.per_bin is not None else conf.get("per_bin", 200)
    eval_override = args.eval_variant or conf.get("eval_variant") or conf.get("eval_task")
    if not ckpts:
        raise SystemExit("need --ckpts or a config with 'ckpts'")

    device = pick_device()
    paths = sorted(glob.glob(ckpts))
    if not paths:
        raise SystemExit(f"no checkpoints match {ckpts}")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, f"{out}_probe.csv")
    rows: List[dict] = []
    diff_col = "difficulty"
    print(f"probing {len(paths)} checkpoints on {device}"
          + (f"  (eval_variant override = {eval_override})" if eval_override else "") + "\n")

    for p in paths:
        model, task, train_variant, length, cfg = load_model(p, device)
        train_variant = train_variant if train_variant is not None else task.default_variant()
        eval_variant = eval_override or train_variant
        L, d, seed = cfg["n_layers"], cfg["d_model"], cfg.get("seed", 0)
        diff_col = task.difficulty_col
        tag = os.path.basename(p)[:-3]

        # one balanced set (of eval_variant), reused for every k so counts compare
        ex = task.sample_balanced(length, per_bin=per_bin, seed=123, variant=eval_variant)
        inputs = torch.from_numpy(np.stack([e[0] for e in ex])).long()
        Ls = np.array([task.difficulty(e[0], length) for e in ex])

        xfer = f"  [train={train_variant} -> eval={eval_variant}]" if eval_variant != train_variant else ""
        print(f"{tag}  (task={task.name}, depth={L}, d={d}, len={length}){xfer}")
        base = {"tag": tag, "task": task.name, "train_variant": train_variant,
                "eval_variant": eval_variant, "length": length, "full_depth": L,
                "d_model": d, "seed": seed}
        for k in range(1, L + 1):
            ok = decode_ok(model, inputs, task, length, device, k)
            for Lx in sorted(set(Ls.tolist())):
                m = Ls == Lx
                rows.append({**base, "layers_used": k, diff_col: int(Lx),
                             "n_solved": int(ok[m].sum()), "n_total": int(m.sum()),
                             "exact_match": round(float(ok[m].mean()), 5)})
            rows.append({**base, "layers_used": k, diff_col: -1,
                         "n_solved": int(ok.sum()), "n_total": len(ok),
                         "exact_match": round(float(ok.mean()), 5)})
            print(f"  layer {k:>2}: overall = {ok.mean():.4f}")
        print()
        del model
        if device == "mps":
            torch.mps.empty_cache()

    fields = ["tag", "task", "train_variant", "eval_variant", "length", "full_depth",
              "d_model", "seed", "layers_used", diff_col, "n_solved", "n_total", "exact_match"]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
