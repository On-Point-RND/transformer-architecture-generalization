#!/usr/bin/env python3
"""Run the checked-in GDN-negative and Mamba2 experiment matrix."""

import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "configs/architecture_comparison/experiments.yaml"
DEFAULT_RUN_ROOT = PROJECT_ROOT / "runs/architecture-comparison"
DEFAULT_OUTPUT = PROJECT_ROOT / "architecture-comparison-results.csv"
COMPLETION_MARKER = ".complete"

MODEL_CONFIGS = {
    "gdn_negative_eigenvalues": (
        PROJECT_ROOT / "configs/architecture_comparison/gdn_negative_eigenvalues.yaml"
    ),
    "mamba2": PROJECT_ROOT / "configs/architecture_comparison/mamba2.yaml",
}

RESULT_FIELDS = (
    "model",
    "task",
    "n_layer",
    "params",
    "block_size",
    "iter",
    "val_acc",
    "val_token_acc",
    "val_loss",
    "best_val_loss",
    "train_acc",
    "train_token_acc",
    "train_loss",
    "run_dir",
)


@dataclass(frozen=True)
class Experiment:
    task: str
    n_layer: int
    params: dict
    block_size: int


@dataclass(frozen=True)
class Job:
    model: str
    index: int
    total: int
    run_dir: Path
    command: tuple[str, ...]

    @property
    def label(self):
        return f"{self.model}:{self.index:02d}"


class ActiveProcesses:
    """Thread-safe registry used to stop child processes after Ctrl-C."""

    def __init__(self):
        self._lock = threading.Lock()
        self._processes = set()
        self._stopping = False

    def add(self, process):
        with self._lock:
            self._processes.add(process)
            stopping = self._stopping
        if stopping:
            process.terminate()

    def discard(self, process):
        with self._lock:
            self._processes.discard(process)

    def terminate_all(self, timeout=5):
        with self._lock:
            self._stopping = True
            processes = list(self._processes)
        for process in processes:
            if process.poll() is None:
                process.terminate()
        deadline = time.monotonic() + timeout
        for process in processes:
            if process.poll() is not None:
                continue
            try:
                process.wait(timeout=max(0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                process.kill()


OUTPUT_LOCK = threading.Lock()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--model",
        action="append",
        choices=tuple(MODEL_CONFIGS),
        dest="models",
        help="architecture to run; repeat to select both (default: both)",
    )
    parser.add_argument("--device", choices=("gpu", "mps", "cpu"))
    parser.add_argument("--gpu", type=int, help="CUDA device index (default: config value)")
    parser.add_argument("--dtype", choices=("float32", "bfloat16", "float16"))
    parser.add_argument(
        "--jobs", type=int, default=1, metavar="N",
        help="maximum experiments to run concurrently on the selected device (default: 1)",
    )
    parser.add_argument("--limit", type=int, help="run only the first N source rows")
    parser.add_argument("--dry-run", action="store_true", help="print commands only")
    parser.add_argument(
        "--collect-only", action="store_true", help="only rebuild the output CSV"
    )
    parser.add_argument(
        "--keep-going", action="store_true", help="continue after a failed child run"
    )
    return parser.parse_args()


def load_experiments(manifest):
    raw = yaml.safe_load(manifest.read_text(encoding="utf-8")) or {}
    default_block_size = int(raw.get("default_block_size", 128))
    experiments = []
    for index, row in enumerate(raw.get("experiments", ()), start=1):
        missing = {"task", "n_layer", "params"} - set(row)
        if missing:
            raise ValueError(f"manifest experiment {index} lacks: {sorted(missing)}")
        experiments.append(
            Experiment(
                task=row["task"],
                n_layer=int(row["n_layer"]),
                params=dict(row["params"]),
                block_size=int(row.get("block_size", default_block_size)),
            )
        )
    if not experiments:
        raise ValueError("experiment manifest is empty")
    keys = [(e.task, e.n_layer, json.dumps(e.params, sort_keys=True)) for e in experiments]
    if len(keys) != len(set(keys)):
        raise ValueError("source CSV contains duplicate task/layer/params experiment points")
    return experiments


def run_dir_for(run_root, model, index, experiment):
    return run_root / model / f"{index:02d}-{experiment.task}-layers-{experiment.n_layer}"


def _value(value):
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, separators=(",", ":"))
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def command_for(args, model, index, experiment):
    run_dir = run_dir_for(args.run_root, model, index, experiment)
    overrides = {
        "model.n_layer": experiment.n_layer,
        "model.block_size": experiment.block_size,
        "task.name": experiment.task,
        "task.params": experiment.params,
        "paths.run_dir": str(run_dir),
        "train.init": "auto",
    }
    if args.device:
        overrides["hardware.device"] = args.device
    if args.gpu is not None:
        overrides["hardware.gpu"] = args.gpu
    if args.dtype:
        overrides["hardware.dtype"] = args.dtype
    command = [
        sys.executable,
        str(PROJECT_ROOT / "main.py"),
        "--config",
        str(MODEL_CONFIGS[model]),
    ]
    for key, value in overrides.items():
        command += ["--set", f"{key}={_value(value)}"]
    # Incomplete runs already have a summary from their latest evaluation. This
    # flag lets main.py enter the run, while train.init=auto resumes its checkpoint.
    command.append("--rerun")
    return command


def read_summary(run_dir):
    path = run_dir / "summary.csv"
    if not path.is_file():
        return None
    with path.open(newline="", encoding="utf-8") as handle:
        return next(csv.DictReader(handle), None)


def completed(run_dir, max_iters):
    if (run_dir / COMPLETION_MARKER).is_file():
        return True
    summary = read_summary(run_dir)
    return summary is not None and int(summary["iter"]) >= max_iters


def collect(args, experiments, models, max_iters):
    rows = []
    complete = 0
    for model in models:
        for index, experiment in enumerate(experiments, start=1):
            run_dir = run_dir_for(args.run_root, model, index, experiment)
            summary = read_summary(run_dir)
            if summary is None:
                continue
            complete += int(completed(run_dir, max_iters))
            try:
                reported_run_dir = run_dir.relative_to(PROJECT_ROOT)
            except ValueError:
                reported_run_dir = run_dir
            rows.append(
                {
                    "model": model,
                    "task": experiment.task,
                    "n_layer": experiment.n_layer,
                    "params": json.dumps(experiment.params, sort_keys=True),
                    "block_size": experiment.block_size,
                    "iter": summary.get("iter", ""),
                    "val_acc": summary.get("val_acc", ""),
                    "val_token_acc": summary.get("val_token_acc", ""),
                    "val_loss": summary.get("val_loss", ""),
                    "best_val_loss": summary.get("best_val_loss", ""),
                    "train_acc": summary.get("train_acc", ""),
                    "train_token_acc": summary.get("train_token_acc", ""),
                    "train_loss": summary.get("train_loss", ""),
                    "run_dir": str(reported_run_dir),
                }
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"collected {len(rows)} summaries ({complete} complete) into {args.output}")


def build_jobs(args, experiments, models, max_iters):
    jobs = []
    for model in models:
        for index, experiment in enumerate(experiments, start=1):
            run_dir = run_dir_for(args.run_root, model, index, experiment)
            if completed(run_dir, max_iters):
                print(f"SKIP complete: {run_dir}")
                continue
            command = tuple(command_for(args, model, index, experiment))
            if args.dry_run:
                print(shlex.join(command))
                continue
            jobs.append(Job(model, index, len(experiments), run_dir, command))
    return jobs


def run_job(job, active_processes):
    """Run one experiment, streaming labeled output and retaining a local log."""
    job.run_dir.mkdir(parents=True, exist_ok=True)
    log_path = job.run_dir / "process.log"
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n$ {shlex.join(job.command)}\n")
        log.flush()
        process = subprocess.Popen(
            job.command,
            cwd=PROJECT_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        active_processes.add(process)
        try:
            with process.stdout:
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    with OUTPUT_LOCK:
                        print(f"[{job.label}] {line}", end="", flush=True)
            return process.wait()
        finally:
            active_processes.discard(process)


def run_jobs(args, jobs, experiments, models, max_iters):
    """Run a bounded queue and return the first nonzero child exit status."""
    if not jobs:
        return 0
    active_processes = ActiveProcesses()
    failures = []
    next_job = iter(jobs)

    def submit_available(executor, futures):
        while len(futures) < args.jobs:
            try:
                job = next(next_job)
            except StopIteration:
                break
            print(
                f"START {job.label} ({job.index}/{job.total}): {job.run_dir}",
                flush=True,
            )
            futures[executor.submit(run_job, job, active_processes)] = job

    executor = ThreadPoolExecutor(max_workers=args.jobs)
    futures = {}
    try:
        submit_available(executor, futures)
        while futures:
            done, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in done:
                job = futures.pop(future)
                try:
                    returncode = future.result()
                except Exception as error:
                    returncode = 1
                    print(f"FAILED {job.label}: {type(error).__name__}: {error}", flush=True)
                if returncode == 0:
                    job.run_dir.mkdir(parents=True, exist_ok=True)
                    (job.run_dir / COMPLETION_MARKER).write_text(
                        "complete\n", encoding="utf-8"
                    )
                    print(f"DONE {job.label}: {job.run_dir}", flush=True)
                else:
                    failures.append(returncode)
                    print(
                        f"FAILED {job.label} (exit {returncode}): {job.run_dir}",
                        flush=True,
                    )
                collect(args, experiments, models, max_iters)
            if not failures or args.keep_going:
                submit_available(executor, futures)
    except KeyboardInterrupt:
        print("\nInterrupted; terminating active experiments...", file=sys.stderr)
        active_processes.terminate_all()
        for future in futures:
            future.cancel()
        return 130
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
    return failures[0] if failures else 0


def main():
    args = parse_args()
    if args.jobs < 1:
        raise ValueError("--jobs must be positive")
    if args.gpu is not None and args.gpu < 0:
        raise ValueError("--gpu must be nonnegative")
    args.manifest = (
        args.manifest if args.manifest.is_absolute() else PROJECT_ROOT / args.manifest
    )
    args.run_root = (
        args.run_root if args.run_root.is_absolute() else PROJECT_ROOT / args.run_root
    )
    args.output = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output
    max_iters_by_model = {
        model: int(
            (yaml.safe_load(path.read_text(encoding="utf-8")) or {})
            .get("train", {})
            .get("max_iters", 80000)
        )
        for model, path in MODEL_CONFIGS.items()
    }
    if len(set(max_iters_by_model.values())) != 1:
        raise ValueError(f"model configs disagree on train.max_iters: {max_iters_by_model}")
    max_iters = next(iter(max_iters_by_model.values()))
    experiments = load_experiments(args.manifest)
    if args.limit is not None:
        if args.limit < 0:
            raise ValueError("--limit must be nonnegative")
        experiments = experiments[: args.limit]
    models = args.models or list(MODEL_CONFIGS)

    if not args.collect_only:
        jobs = build_jobs(args, experiments, models, max_iters)
        if not args.dry_run:
            returncode = run_jobs(args, jobs, experiments, models, max_iters)
            if returncode:
                return returncode

    if not args.dry_run:
        collect(args, experiments, models, max_iters)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
