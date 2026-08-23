"""Unified, task-agnostic training loop.

One training loop for every task (addition, permutation, sorting, ...). All the
task-specific bits — vocabulary, sequence length, where the answer lives, how
data is generated — come from the shared ``Task`` interface (see datasets/),
selected by name in ``TrainConfig.task``.

Conventions: AdamW, lr 3e-4, 500-step warmup + cosine decay, weight_decay 0.1,
grad_clip 1.0, loss on the answer tokens only. Exact-match is measured
autoregressively; a cheap teacher-forced proxy is used for the in-loop
early-stop check.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from datasets import get_task
from model import GPT

RESULTS_DIR = "results"
CKPT_DIR = "checkpoints"


@dataclass
class TrainConfig:
    task: str = "permutation"          # task name in the datasets registry
    variant: Optional[str] = None      # SEQ/INDEP, S5/C5, or None (uses default)
    length: int = 16                   # length parameter: N digits / K perms / n items

    n_layers: int = 2
    d_model: int = 128
    n_heads: Optional[int] = None
    d_ff: Optional[int] = None
    dropout: float = 0.0
    looped: bool = False
    n_loops: int = 1

    # size positional embeddings for the largest EVAL length (length-generalization);
    # None -> sized to `length`.
    max_eval_length: Optional[int] = None

    lr: float = 3e-4
    weight_decay: float = 0.1
    warmup_steps: int = 500
    max_steps: int = 20_000
    batch_size: int = 256
    grad_clip: float = 1.0

    n_train: int = 100_000
    n_test: int = 5_000

    seed: int = 0
    eval_every: int = 1_000
    train_eval_n: int = 512
    eval_subset: int = 512
    eval_mode: str = "teacher_forced"  # "teacher_forced" (fast) | "autoregressive"
    early_stop_acc: float = 0.999
    min_steps: int = 4_000
    patience: int = 0
    tag: str = "run"
    device: Optional[str] = None


def pick_device(requested: Optional[str] = None) -> str:
    if requested:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def lr_at(step: int, base_lr: float, warmup: int, max_steps: int) -> float:
    if step < warmup:
        return base_lr * (step + 1) / warmup
    if step >= max_steps:
        return 0.0
    progress = (step - warmup) / max(1, (max_steps - warmup))
    return 0.5 * base_lr * (1.0 + math.cos(math.pi * progress))


@torch.no_grad()
def exact_match_accuracy(model, inputs, answer_start, answer_len, device,
                         batch_size: int = 512) -> float:
    """Autoregressive greedy decoding of the answer; fraction fully correct."""
    model.eval()
    n, correct = inputs.size(0), 0
    for i in range(0, n, batch_size):
        batch = inputs[i:i + batch_size].to(device)
        gen = batch[:, :answer_start].clone()
        for _ in range(answer_len):
            logits, _ = model(gen)
            gen = torch.cat([gen, logits[:, -1, :].argmax(dim=-1, keepdim=True)], dim=1)
        pred = gen[:, answer_start:answer_start + answer_len]
        true = batch[:, answer_start:answer_start + answer_len]
        correct += (pred == true).all(dim=1).sum().item()
    model.train()
    return correct / n


@torch.no_grad()
def teacher_forced_accuracy(model, inputs, targets, device, ignore_index,
                            batch_size: int = 512) -> float:
    """Cheap single-forward-pass proxy: all answer-token argmaxes correct."""
    model.eval()
    n, correct = inputs.size(0), 0
    for i in range(0, n, batch_size):
        xb = inputs[i:i + batch_size].to(device)
        yb = targets[i:i + batch_size].to(device)
        logits, _ = model(xb)
        pred = logits.argmax(dim=-1)
        mask = yb != ignore_index
        correct += ((pred == yb) | ~mask).all(dim=1).sum().item()
    model.train()
    return correct / n


def _stack(examples) -> Tuple[torch.Tensor, torch.Tensor]:
    X = torch.from_numpy(np.stack([e[0] for e in examples])).long()
    Y = torch.from_numpy(np.stack([e[1] for e in examples])).long()
    return X, Y


def train_run(cfg: TrainConfig, verbose: bool = True) -> Dict:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = pick_device(cfg.device)

    task = get_task(cfg.task)
    variant = cfg.variant if cfg.variant is not None else task.default_variant()
    answer_start, answer_len = task.answer_span(cfg.length)
    ignore_index = task.ignore_index
    model_max_len = task.seq_len(cfg.max_eval_length or cfg.length)

    if verbose:
        print(f"[{cfg.tag}] device={device} task={cfg.task} variant={variant} "
              f"len={cfg.length} L={cfg.n_layers} d={cfg.d_model} "
              f"looped={cfg.looped} T={cfg.n_loops}")

    train_ex, test_ex = task.make_splits(cfg.length, cfg.n_train, cfg.n_test,
                                         seed=cfg.seed, variant=variant)
    Xtr, Ytr = _stack(train_ex)
    Xte, Yte = _stack(test_ex)
    Xtr_dev, Ytr_dev = Xtr.to(device), Ytr.to(device)
    n_train = Xtr.size(0)
    sub = min(cfg.eval_subset, Xte.size(0))
    Xte_eval, Yte_eval = Xte[:sub], Yte[:sub]
    idx = torch.randperm(n_train)[: min(cfg.train_eval_n, n_train)]
    Xtr_eval, Ytr_eval = Xtr[idx], Ytr[idx]

    def _acc(X, Y):
        if cfg.eval_mode == "autoregressive":
            return exact_match_accuracy(model, X, answer_start, answer_len, device)
        return teacher_forced_accuracy(model, X, Y, device, ignore_index)

    model = GPT(n_layers=cfg.n_layers, d_model=cfg.d_model, vocab_size=task.vocab_size,
                max_seq_len=model_max_len, n_heads=cfg.n_heads, d_ff=cfg.d_ff,
                dropout=cfg.dropout, looped=cfg.looped, n_loops=cfg.n_loops).to(device)
    if verbose:
        print(f"[{cfg.tag}] train={n_train} test={Xte.size(0)} seq_len={Xtr.size(1)} "
              f"params={model.num_params():,}")

    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                              weight_decay=cfg.weight_decay, betas=(0.9, 0.95))
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(CKPT_DIR, exist_ok=True)
    log_rows: List[Dict] = []
    gen = torch.Generator().manual_seed(cfg.seed)
    best, since_improve, t0 = 0.0, 0, time.time()
    stopped_early = False

    for step in range(cfg.max_steps + 1):
        if step % cfg.eval_every == 0:
            tr_acc = _acc(Xtr_eval, Ytr_eval)
            te_acc = _acc(Xte_eval, Yte_eval)
            if te_acc > best + 1e-6:
                best, since_improve = te_acc, 0
            else:
                since_improve += 1
            loss_v = locals().get("last_loss", float("nan"))
            if verbose:
                print(f"[{cfg.tag}] step {step:>6} | loss {loss_v:.4f} | "
                      f"train {tr_acc:.4f} | test {te_acc:.4f} | {time.time()-t0:.0f}s")
            log_rows.append({
                "tag": cfg.tag, "task": cfg.task, "variant": variant,
                "length": cfg.length, "n_layers": cfg.n_layers, "d_model": cfg.d_model,
                "looped": int(cfg.looped), "n_loops": cfg.n_loops, "seed": cfg.seed,
                "step": step, "train_loss": round(float(loss_v), 5),
                "train_acc": round(tr_acc, 5), "id_test_acc": round(te_acc, 5)})
            if te_acc >= cfg.early_stop_acc:
                if verbose:
                    print(f"[{cfg.tag}] early stop at {step} (test {te_acc:.4f})")
                stopped_early = True
                break
            if cfg.patience > 0 and step >= cfg.min_steps and since_improve >= cfg.patience:
                if verbose:
                    print(f"[{cfg.tag}] patience stop at {step} (best {best:.4f})")
                stopped_early = True
                break
        if step == cfg.max_steps:
            break

        b = torch.randint(0, n_train, (cfg.batch_size,), generator=gen)
        lr = lr_at(step, cfg.lr, cfg.warmup_steps, cfg.max_steps)
        for g in optim.param_groups:
            g["lr"] = lr
        _, loss = model(Xtr_dev[b], targets=Ytr_dev[b],
                        n_loops=cfg.n_loops if cfg.looped else None,
                        ignore_index=ignore_index)
        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optim.step()
        last_loss = loss.item()

    csv_path = os.path.join(RESULTS_DIR, f"{cfg.tag}.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(log_rows[0].keys()))
        w.writeheader()
        w.writerows(log_rows)
    ckpt_path = os.path.join(CKPT_DIR, f"{cfg.tag}.pt")
    torch.save({"model_state": model.state_dict(), "config": asdict(cfg)}, ckpt_path)
    if verbose:
        print(f"[{cfg.tag}] done best={best:.4f} -> {ckpt_path}")
    return {"tag": cfg.tag, "best_id_test_acc": best,
            "final_id_test_acc": log_rows[-1]["id_test_acc"],
            "stopped_early": stopped_early, "csv_path": csv_path,
            "ckpt_path": ckpt_path, "steps": log_rows[-1]["step"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="permutation")
    ap.add_argument("--variant", default=None)
    ap.add_argument("--length", type=int, default=16)
    ap.add_argument("--n_layers", type=int, default=2)
    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--max_steps", type=int, default=20000)
    ap.add_argument("--n_train", type=int, default=100000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="run")
    a = ap.parse_args()
    cfg = TrainConfig(task=a.task, variant=a.variant, length=a.length,
                      n_layers=a.n_layers, d_model=a.d_model, max_steps=a.max_steps,
                      n_train=a.n_train, seed=a.seed, tag=a.tag)
    res = train_run(cfg)
    print("\nbest id-test exact-match:", round(res["best_id_test_acc"], 4))


if __name__ == "__main__":
    main()
