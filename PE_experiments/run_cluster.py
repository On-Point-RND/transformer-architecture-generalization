#!/usr/bin/env python3
"""Explicit cluster launcher for positional-encoding experiments.

Nothing is swept implicitly: at least one PE and one task must be supplied.
Multiple values intentionally request the Cartesian product PE x task x seed.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path


REPO_DIR = Path(__file__).resolve().parents[1]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from generator.positional_lab import POSITIONAL_TASK_DEFAULTS


CONFIG_DIR = REPO_DIR / "PE_experiments" / "configs"
PE_CHOICES = tuple(sorted(path.stem for path in CONFIG_DIR.glob("*.py")))
TASK_CHOICES = tuple(POSITIONAL_TASK_DEFAULTS)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pe", nargs="+", choices=PE_CHOICES,
                        help="one or more PE config names")
    parser.add_argument("--task", nargs="+", choices=TASK_CHOICES,
                        help="one or more registered positional tasks")
    parser.add_argument("--seed", nargs="+", type=int, default=[1337])
    parser.add_argument("--config", default="config/small.py",
                        help="base train config, relative to repo root")
    parser.add_argument("--out-root", default="out/pe_experiments")
    parser.add_argument("--max-iters", type=int)
    parser.add_argument("--device")
    parser.add_argument("--nproc-per-node", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--list", action="store_true",
                        help="print available PE/task names and exit")
    parser.add_argument("--extra", action="append", default=[],
                        help="extra train.py override, e.g. --extra=--batch_size=128")
    args = parser.parse_args()
    if args.list:
        return args
    if not args.pe or not args.task:
        parser.error("--pe and --task are required unless --list is used")
    if args.nproc_per_node < 1:
        parser.error("--nproc-per-node must be positive")
    return args


def train_prefix(nproc_per_node):
    if nproc_per_node == 1:
        return [sys.executable, "train.py"]
    return [
        sys.executable, "-m", "torch.distributed.run", "--standalone",
        f"--nproc_per_node={nproc_per_node}", "train.py",
    ]


def command_for(args, pe, task, seed):
    run_name = f"{task}__{pe}__seed{seed}"
    out_dir = Path(args.out_root) / task / pe / f"seed_{seed}"
    command = train_prefix(args.nproc_per_node)
    command += [
        args.config,
        f"PE_experiments/configs/{pe}.py",
        f"--dataset={task}",
        "--gen_params={}",
        f"--seed={seed}",
        f"--data_seed={seed}",
        f"--out_dir={out_dir.as_posix()}",
        f"--wandb_run_name={run_name}",
        f"--log_file={(out_dir / 'train_log.jsonl').as_posix()}",
        f"--init_from={'resume' if args.resume else 'scratch'}",
    ]
    if args.max_iters is not None:
        command += [f"--max_iters={args.max_iters}", f"--lr_decay_iters={args.max_iters}"]
    if args.device is not None and args.nproc_per_node == 1:
        command.append(f"--device={args.device}")
    command.extend(args.extra)
    return command


def main():
    args = parse_args()
    if args.list:
        print("PE:")
        print("\n".join(f"  {name}" for name in PE_CHOICES))
        print("Tasks:")
        print("\n".join(f"  {name}" for name in TASK_CHOICES))
        return 0

    os.chdir(REPO_DIR)
    failures = []
    jobs = [(pe, task, seed) for task in args.task for pe in args.pe for seed in args.seed]
    print(f"planned jobs: {len(jobs)}")
    for index, (pe, task, seed) in enumerate(jobs, start=1):
        command = command_for(args, pe, task, seed)
        print(f"\n[{index}/{len(jobs)}] PE={pe} task={task} seed={seed}")
        print(shlex.join(command), flush=True)
        if args.dry_run:
            continue
        result = subprocess.run(command, cwd=REPO_DIR)
        if result.returncode:
            failures.append((pe, task, seed, result.returncode))
            if not args.continue_on_error:
                return result.returncode
    if failures:
        print("failed jobs:", failures, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
