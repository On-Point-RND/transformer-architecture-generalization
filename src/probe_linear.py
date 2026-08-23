"""HONEST layer probe (logit lens), task-agnostic.

For a frozen trained model, train a separate linear read-out on EACH layer's raw
representation (before final LN / head) and measure how much of the answer is
linearly decodable there, stratified by the task's dependency length L(x). This
removes the confound of the model's own head being trained only for the final
layer.

If the answer is present in early layers -> parallel-ish; if it genuinely builds
up across layers (later layers unlock longer L(x)) -> artifact-free evidence of
sequential, depth-dependent computation.

All task-specific behaviour (vocabulary, answer span, difficulty L(x), data)
comes from the shared ``Task`` interface (see datasets/), selected from each
checkpoint's own config. Works for addition, permutation, sorting, ...

CSV schema (results/<out>_probe.csv):
  tag, task, variant, length, full_depth, d_model, seed, layers_used,
  <difficulty_col>, n_solved, n_total, exact_match
where <difficulty_col> is chain_length (addition/perm) or inversions (sorting),
and a row with <difficulty_col> = -1 is the overall (all-L) exact-match.

Usage:
  python src/probe_linear.py --ckpts 'checkpoints/perm_grid_*_S5_*.pt' --out perm_linear
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from datasets import get_task
from datasets.base import Task
from model import GPT

RESULTS_DIR = "results"


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def read_ckpt_spec(cfg: dict) -> Tuple[str, Optional[str], int]:
    """(task_name, variant, length) from a checkpoint config, new- or legacy-schema."""
    if "length" in cfg:                       # unified TrainConfig schema
        return cfg["task"], cfg.get("variant"), cfg["length"]
    # legacy per-task schema: cfg["task"] held the VARIANT, length was K / n / N
    variant = cfg.get("task")
    for key, name in (("K", "permutation"), ("n", "sorting"), ("N", "addition")):
        if key in cfg:
            return name, variant, cfg[key]
    raise KeyError(f"cannot read (task, length) from checkpoint config: {sorted(cfg)}")


def load_model(path: str, device: str) -> Tuple[GPT, Task, Optional[str], int, dict]:
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = ck["config"]
    task_name, variant, length = read_ckpt_spec(cfg)
    task = get_task(task_name)
    model = GPT(n_layers=cfg["n_layers"], d_model=cfg["d_model"], vocab_size=task.vocab_size,
                max_seq_len=task.seq_len(length), n_heads=cfg.get("n_heads"),
                looped=cfg.get("looped", False), n_loops=cfg.get("n_loops", 1)).to(device)
    model.load_state_dict(ck["model_state"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, task, variant, length, cfg


def probe_one(model, task: Task, variant, length, cfg, device,
              per_bin, n_train, steps, batch_size, lr) -> List[dict]:
    L, d = cfg["n_layers"], cfg["d_model"]
    seed = cfg.get("seed", 0)
    variant = variant if variant is not None else task.default_variant()
    a_start, a_len = task.answer_span(length)
    ans = list(range(a_start - 1, a_start - 1 + a_len))   # positions predicting the answer
    tgt_pos = list(range(a_start, a_start + a_len))        # answer-token positions
    diff_col = task.difficulty_col

    # training data for the probes: fresh random examples (model stays frozen)
    tr, _ = task.make_splits(length, n_train=n_train, n_test=0, seed=seed + 100, variant=variant)
    Xtr = torch.from_numpy(np.stack([e[0] for e in tr])).long().to(device)

    probes = [nn.Linear(d, task.vocab_size).to(device) for _ in range(L)]
    opt = torch.optim.Adam([p for pr in probes for p in pr.parameters()], lr=lr)
    ce = nn.CrossEntropyLoss()
    gen = torch.Generator(device="cpu").manual_seed(0)
    for _ in range(steps):
        bi = torch.randint(0, Xtr.size(0), (batch_size,), generator=gen)
        xb = Xtr[bi]
        hs = model.hidden_states(xb)
        targets = xb[:, tgt_pos].reshape(-1)
        opt.zero_grad(set_to_none=True)
        loss = 0.0
        for k in range(L):
            loss = loss + ce(probes[k](hs[k][:, ans, :].reshape(-1, d)), targets)
        loss.backward()
        opt.step()

    # eval on a difficulty-balanced set, stratified by L(x)
    ex = task.sample_balanced(length, per_bin=per_bin, seed=123, variant=variant)
    Xte = torch.from_numpy(np.stack([e[0] for e in ex])).long()
    Ls = np.array([task.difficulty(e[0], length) for e in ex])
    tag = cfg.get("_tag", "model")
    ok = [np.zeros(Xte.size(0), dtype=bool) for _ in range(L)]
    with torch.no_grad():
        for i in range(0, Xte.size(0), 512):
            xb = Xte[i:i + 512].to(device)
            targets = xb[:, tgt_pos]
            hs = model.hidden_states(xb)
            for k in range(L):
                pred = probes[k](hs[k][:, ans, :]).argmax(-1)
                ok[k][i:i + xb.size(0)] = (pred == targets).all(dim=1).cpu().numpy()

    rows = []
    for k in range(L):
        base = {"tag": tag, "task": task.name, "variant": variant, "length": length,
                "full_depth": L, "d_model": d, "seed": seed, "layers_used": k + 1}
        for Lx in sorted(set(Ls.tolist())):
            m = Ls == Lx
            rows.append({**base, diff_col: int(Lx), "n_solved": int(ok[k][m].sum()),
                         "n_total": int(m.sum()),
                         "exact_match": round(float(ok[k][m].mean()), 5)})
        rows.append({**base, diff_col: -1, "n_solved": int(ok[k].sum()),
                     "n_total": len(ok[k]), "exact_match": round(float(ok[k].mean()), 5)})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", required=True, help="glob for checkpoints")
    ap.add_argument("--out", default="linear")
    ap.add_argument("--per-bin", type=int, default=100)
    ap.add_argument("--n-train", type=int, default=40000)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    device = pick_device()
    paths = sorted(glob.glob(args.ckpts))
    if not paths:
        raise SystemExit(f"no checkpoints match {args.ckpts}")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, f"{args.out}_probe.csv")
    all_rows: List[dict] = []
    diff_col = "difficulty"
    print(f"linear-probing {len(paths)} checkpoints on {device}\n")

    for p in paths:
        model, task, variant, length, cfg = load_model(p, device)
        cfg["_tag"] = os.path.basename(p)[:-3]
        diff_col = task.difficulty_col
        rows = probe_one(model, task, variant, length, cfg, device, args.per_bin,
                         args.n_train, args.steps, args.batch_size, args.lr)
        print(f"{cfg['_tag']}  (task={task.name}, variant={variant}, "
              f"depth={cfg['n_layers']}, d={cfg['d_model']}, len={length})")
        for r in rows:
            if r[diff_col] == -1:
                print(f"  layer {r['layers_used']:>2}: linear-decode overall = {r['exact_match']}")
        print()
        all_rows += rows
        del model
        if device == "mps":
            torch.mps.empty_cache()

    fields = ["tag", "task", "variant", "length", "full_depth", "d_model", "seed",
              "layers_used", diff_col, "n_solved", "n_total", "exact_match"]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(all_rows)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
