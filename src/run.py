"""Task-agnostic sweep orchestrator: run a grid of training runs from a YAML
config and write results CSVs. One runner for every task (addition, permutation,
sorting, ...); all task-specific behaviour comes from the shared ``Task``
interface (see datasets/).

Two CSVs are written per sweep (append/resume-safe):
  results/<sweep>_summary.csv       : one row per run (config + final/best metrics)
  results/<sweep>_<by_suffix>.csv   : long format, exact-match stratified by the
                                      task's difficulty L(x) per run (money-plot).
                                      <by_suffix> is by_chain (addition/perm) or
                                      by_inv (sorting).
A third CSV, results/<sweep>_lengthgen.csv, is written for the "length_gen" kind.

Runs are independent; an interrupted sweep resumes by skipping tags already
present in the summary CSV.

Config schema (all keys optional unless noted):
  sweep:    str, name of this sweep (required)
  task:     registry name (addition/permutation/sorting). If omitted it is
            inferred from which length key is present (Ns->addition,
            Ks->permutation, ns->sorting).
  lengths:  list of length params. Aliases: Ns / Ks / ns.
  variants: list of task variants (SEQ/INDEP, S5/C5). Alias: tasks. Default [None].
  depths:   list of n_layers.        widths: list of d_model.
  seeds:    list of seeds (required).
  kinds:    list of sweep kinds (default ["grid"]).
  train:    dict of training hyperparameters passed through to TrainConfig.
  eval:     {chain_per_bin / inv_per_bin: int}.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import os
from typing import Dict, List, Optional

import numpy as np
import torch
import yaml

import budgets as budgets_mod
from datasets import get_task
from datasets.base import Task
from model import GPT, default_n_heads
from train import RESULTS_DIR, TrainConfig, exact_match_accuracy, pick_device, train_run

SUMMARY_FIELDS = [
    "sweep", "tag", "sweep_kind", "task", "variant", "N", "budget", "target_params",
    "n_layers", "d_model", "n_heads", "looped", "n_loops", "seed",
    "params", "steps", "best_id_test_acc", "final_id_test_acc", "stopped_early",
    "id_test_em_ag",  # authoritative: autoregressive exact-match on full id-test
]
LENGTHGEN_FIELDS = [
    "sweep", "tag", "task", "variant", "train_N", "n_layers", "d_model", "seed",
    "eval_N", "exact_match",
]


def _by_diff_fields(diff_col: str) -> List[str]:
    return [
        "sweep", "tag", "sweep_kind", "task", "variant", "N", "budget",
        "n_layers", "d_model", "n_heads", "looped", "n_loops", "seed",
        diff_col, "exact_match", "n_examples",
    ]


# --------------------------------------------------------------------------
# Config -> list of run specs
# --------------------------------------------------------------------------
def _infer_task(cfg: dict) -> str:
    if "task" in cfg:
        return cfg["task"]
    for key, name in (("Ns", "addition"), ("Ks", "permutation"), ("ns", "sorting")):
        if key in cfg:
            return name
    raise ValueError("config must set 'task' (or a length key Ns/Ks/ns to infer it)")


def _lengths(cfg: dict) -> List[int]:
    for key in ("lengths", "Ns", "Ks", "ns"):
        if key in cfg:
            return cfg[key]
    return []


def _variants(cfg: dict) -> List[Optional[str]]:
    return cfg.get("variants") or cfg.get("tasks") or [None]


def build_run_specs(cfg: dict) -> List[dict]:
    """Expand a sweep config into concrete run specifications."""
    sweep = cfg["sweep"]
    task_name = _infer_task(cfg)
    variants = _variants(cfg)
    lengths = _lengths(cfg)
    depths = cfg.get("depths", [])
    seeds = cfg["seeds"]
    tr = cfg["train"]
    specs: List[dict] = []

    def base(**kw) -> dict:
        d = dict(sweep=sweep, task=task_name, looped=False, n_loops=1, tr=tr,
                 target_params=0)
        d.update(kw)
        return d

    for kind in cfg.get("kinds", ["grid"]):
        if kind == "fixed_budget":
            for bud in cfg["budgets"]:
                dm = {r["L"]: r for r in budgets_mod.make_budget_grid(bud["target"], depths)}
                for v, N, L, seed in itertools.product(variants, lengths, depths, seeds):
                    g = dm[L]
                    specs.append(base(sweep_kind=kind, variant=v, length=N,
                                      budget=bud["name"], target_params=bud["target"],
                                      n_layers=L, d_model=g["d_model"],
                                      n_heads=g["n_heads"], seed=seed))
        elif kind == "fixed_width":
            d_model = cfg.get("fixed_width", 128)
            for v, N, L, seed in itertools.product(variants, lengths, depths, seeds):
                specs.append(base(sweep_kind=kind, variant=v, length=N,
                                  budget=f"fw{d_model}", n_layers=L, d_model=d_model,
                                  n_heads=default_n_heads(d_model), seed=seed))
        elif kind in ("grid", "width_sweep"):
            # full depth x width grid (or width-only at a fixed depth).
            fixed_depth = cfg.get("depth")
            depth_iter = [fixed_depth] if (kind == "width_sweep" and fixed_depth) else depths
            for v, N, L, w, seed in itertools.product(
                    variants, lengths, depth_iter, cfg["widths"], seeds):
                specs.append(base(sweep_kind=kind, variant=v, length=N, budget=f"d{w}",
                                  n_layers=L, d_model=w, n_heads=default_n_heads(w),
                                  seed=seed))
        elif kind == "cells":
            # explicit (variant, depth, width) cells to fill grid gaps.
            N = cfg.get("length", cfg.get("K", lengths[0] if lengths else 16))
            for cell in cfg["cells"]:
                w = cell["width"]
                v = cell.get("task", cell.get("variant", variants[0]))
                for seed in seeds:
                    specs.append(base(sweep_kind="width_sweep", variant=v, length=N,
                                      budget=f"d{w}", n_layers=cell["depth"], d_model=w,
                                      n_heads=default_n_heads(w), seed=seed))
        elif kind == "length_gen":
            width = cfg.get("lg_width", 128)
            train_N = cfg.get("lg_train_N", 8)
            eval_Ns = sorted(set(cfg.get("lg_eval_Ns", [10, 12, 16, 24, 32])) | {train_N})
            for v, L, seed in itertools.product(variants, cfg.get("lg_depths", [2, 12]), seeds):
                specs.append(base(sweep_kind=kind, variant=v, length=train_N,
                                  budget=f"fw{width}", n_layers=L, d_model=width,
                                  n_heads=default_n_heads(width), seed=seed,
                                  max_eval_length=max(eval_Ns), eval_lengths=list(eval_Ns)))
        elif kind == "head_ablation":
            width = cfg.get("ha_width", 128)
            depth = cfg.get("ha_depth", 2)
            N = cfg.get("ha_N", cfg.get("ha_K", 16))
            for v, H, seed in itertools.product(variants, cfg.get("ha_heads", [1, 2, 4, 8]), seeds):
                assert width % H == 0, f"d_model {width} not divisible by H={H}"
                specs.append(base(sweep_kind=kind, variant=v, length=N, budget=f"H{H}",
                                  n_layers=depth, d_model=width, n_heads=H, seed=seed))
        elif kind == "loop":
            d_model = cfg.get("loop_width", 128)
            N = cfg.get("loop_N", cfg.get("loop_K", 16))
            for v, T, seed in itertools.product(variants, cfg.get("loop_Ts", [1, 2, 4, 8, 12]), seeds):
                specs.append(base(sweep_kind=kind, variant=v, length=N, budget=f"loop{d_model}",
                                  n_layers=1, d_model=d_model, n_heads=default_n_heads(d_model),
                                  looped=True, n_loops=T, seed=seed))
        else:
            raise ValueError(f"unknown sweep kind {kind!r}")
    return specs


def spec_tag(s: dict) -> str:
    v = s["variant"] if s["variant"] is not None else s["task"]
    if s["looped"]:
        return f"{s['sweep']}_{s['sweep_kind']}_{v}_N{s['length']}_T{s['n_loops']}_s{s['seed']}"
    return (f"{s['sweep']}_{s['sweep_kind']}_{v}_N{s['length']}_"
            f"{s['budget']}_L{s['n_layers']}_s{s['seed']}")


# --------------------------------------------------------------------------
# CSV helpers (append-safe)
# --------------------------------------------------------------------------
def _append_rows(path: str, fields: List[str], rows: List[dict]) -> None:
    exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if not exists:
            w.writeheader()
        w.writerows(rows)


def _done_tags(path: str) -> set:
    if not os.path.exists(path):
        return set()
    with open(path) as f:
        return {row["tag"] for row in csv.DictReader(f)}


# --------------------------------------------------------------------------
# Task-agnostic metrics
# --------------------------------------------------------------------------
@torch.no_grad()
def accuracy_by_difficulty(model: GPT, task: Task, length: int, variant, device: str,
                           per_bin: int, seed: int = 123) -> Dict[int, tuple]:
    """Autoregressive exact-match stratified by the task's difficulty L(x).
    Returns {difficulty: (exact_match, n_examples)}."""
    ex = task.sample_balanced(length, per_bin=per_bin, seed=seed, variant=variant)
    inputs = torch.from_numpy(np.stack([e[0] for e in ex])).long()
    diffs = np.array([task.difficulty(e[0], length) for e in ex])
    a_start, a_len = task.answer_span(length)
    model.eval()
    ok = []
    for i in range(0, inputs.size(0), 512):
        b = inputs[i:i + 512].to(device)
        gen = b[:, :a_start].clone()
        for _ in range(a_len):
            lg, _l = model(gen)
            gen = torch.cat([gen, lg[:, -1, :].argmax(dim=-1, keepdim=True)], dim=1)
        good = (gen[:, a_start:a_start + a_len] == b[:, a_start:a_start + a_len]).all(dim=1)
        ok.append(good.cpu().numpy())
    ok = np.concatenate(ok)
    return {int(d): (float(ok[diffs == d].mean()), int((diffs == d).sum()))
            for d in sorted(set(diffs.tolist()))}


@torch.no_grad()
def length_generalization(model: GPT, task: Task, variant, eval_lengths, device: str,
                          n_test: int = 2000, seed: int = 0) -> Dict[int, float]:
    """Exact-match on unseen (larger) lengths without retraining."""
    out = {}
    for L in eval_lengths:
        _, te = task.make_splits(L, n_train=1, n_test=n_test, seed=seed, variant=variant)
        X = torch.from_numpy(np.stack([e[0] for e in te])).long()
        a_start, a_len = task.answer_span(L)
        out[L] = exact_match_accuracy(model, X, a_start, a_len, device)
    return out


# --------------------------------------------------------------------------
# Run the sweep
# --------------------------------------------------------------------------
def run_sweep(config_path: str, limit: Optional[int] = None, verbose: bool = True) -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    specs = build_run_specs(cfg)
    sweep = cfg["sweep"]
    task = get_task(_infer_task(cfg))
    diff_col = task.difficulty_col
    by_fields = _by_diff_fields(diff_col)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    summary_path = os.path.join(RESULTS_DIR, f"{sweep}_summary.csv")
    by_diff_path = os.path.join(RESULTS_DIR, f"{sweep}_{task.by_suffix}.csv")
    lengthgen_path = os.path.join(RESULTS_DIR, f"{sweep}_lengthgen.csv")
    ev = cfg.get("eval", {})
    per_bin = ev.get("chain_per_bin", ev.get("inv_per_bin", 300))

    done = _done_tags(summary_path)
    todo = [s for s in specs if spec_tag(s) not in done]
    if limit is not None:
        todo = todo[:limit]

    print(f"sweep={sweep} task={task.name}: {len(specs)} total runs, {len(done)} done, "
          f"{len(todo)} to run now (device={pick_device()})")

    for i, s in enumerate(todo):
        tag = spec_tag(s)
        tr = s["tr"]
        cfg_run = TrainConfig(
            task=s["task"], variant=s["variant"], length=s["length"],
            n_layers=s["n_layers"], d_model=s["d_model"], n_heads=s["n_heads"],
            looped=s["looped"], n_loops=s["n_loops"],
            lr=tr.get("lr", 3e-4), weight_decay=tr.get("weight_decay", 0.1),
            warmup_steps=tr.get("warmup_steps", 500), max_steps=tr["max_steps"],
            batch_size=tr.get("batch_size", 256), grad_clip=tr.get("grad_clip", 1.0),
            dropout=tr.get("dropout", 0.0), n_train=tr.get("n_train", 100_000),
            n_test=tr.get("n_test", 5_000), eval_every=tr.get("eval_every", 1000),
            train_eval_n=tr.get("train_eval_n", 1000),
            eval_subset=tr.get("eval_subset", 512),
            eval_mode=tr.get("eval_mode", "teacher_forced"),
            min_steps=tr.get("min_steps", 4000), patience=tr.get("patience", 0),
            early_stop_acc=tr.get("early_stop_acc", 0.999), seed=s["seed"], tag=tag,
            max_eval_length=s.get("max_eval_length"),
        )
        print(f"\n[{i+1}/{len(todo)}] {tag}")
        res = train_run(cfg_run, verbose=verbose)

        # reload checkpoint; compute authoritative (autoregressive) metrics
        device = pick_device()
        ck = torch.load(res["ckpt_path"], map_location=device, weights_only=False)
        model = GPT(n_layers=s["n_layers"], d_model=s["d_model"], vocab_size=task.vocab_size,
                    max_seq_len=task.seq_len(s.get("max_eval_length") or s["length"]),
                    n_heads=s["n_heads"], looped=s["looped"], n_loops=s["n_loops"]).to(device)
        model.load_state_dict(ck["model_state"])
        model.eval()

        variant = s["variant"] if s["variant"] is not None else task.default_variant()
        _, test_ex = task.make_splits(s["length"], cfg_run.n_train, cfg_run.n_test,
                                      seed=cfg_run.seed, variant=variant)
        Xte = torch.from_numpy(np.stack([e[0] for e in test_ex])).long()
        a_start, a_len = task.answer_span(s["length"])
        id_test_em_ag = exact_match_accuracy(model, Xte, a_start, a_len, device)
        abl = accuracy_by_difficulty(model, task, s["length"], variant, device, per_bin)

        _append_rows(summary_path, SUMMARY_FIELDS, [{
            "sweep": sweep, "tag": tag, "sweep_kind": s["sweep_kind"],
            "task": task.name, "variant": s["variant"], "N": s["length"],
            "budget": s["budget"], "target_params": s["target_params"],
            "n_layers": s["n_layers"], "d_model": s["d_model"], "n_heads": s["n_heads"],
            "looped": int(s["looped"]), "n_loops": s["n_loops"], "seed": s["seed"],
            "params": model.num_params(), "steps": res["steps"],
            "best_id_test_acc": round(res["best_id_test_acc"], 5),
            "final_id_test_acc": round(res["final_id_test_acc"], 5),
            "stopped_early": int(res["stopped_early"]),
            "id_test_em_ag": round(id_test_em_ag, 5),
        }])

        _append_rows(by_diff_path, by_fields, [{
            "sweep": sweep, "tag": tag, "sweep_kind": s["sweep_kind"],
            "task": task.name, "variant": s["variant"], "N": s["length"],
            "budget": s["budget"], "n_layers": s["n_layers"], "d_model": s["d_model"],
            "n_heads": s["n_heads"], "looped": int(s["looped"]), "n_loops": s["n_loops"],
            "seed": s["seed"], diff_col: d, "exact_match": round(a, 5), "n_examples": ne,
        } for d, (a, ne) in abl.items()])

        if s["sweep_kind"] == "length_gen":
            lg = length_generalization(model, task, variant, s["eval_lengths"], device)
            _append_rows(lengthgen_path, LENGTHGEN_FIELDS, [{
                "sweep": sweep, "tag": tag, "task": task.name, "variant": s["variant"],
                "train_N": s["length"], "n_layers": s["n_layers"], "d_model": s["d_model"],
                "seed": s["seed"], "eval_N": L,
                "exact_match": (round(acc, 5) if acc is not None else ""),
            } for L, acc in lg.items()])

        del model
        if device == "mps":
            torch.mps.empty_cache()

    print(f"\nsweep complete. summary -> {summary_path}\n{task.by_suffix} -> {by_diff_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config", help="path to sweep YAML (e.g. configs/pilot.yaml)")
    ap.add_argument("--limit", type=int, default=None, help="max runs this invocation")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    run_sweep(args.config, limit=args.limit, verbose=not args.quiet)


if __name__ == "__main__":
    main()
