#!/usr/bin/env python3
"""Train the run — or the grid of runs — described by a config.

    python main.py                                      # reads configs/main.yaml
    python main.py --set model.pos_encoding=alibi
    python main.py --config configs/my_experiment.yaml

A config file pulls in others with `include:`; several --config files merge left
to right, and --set overrides everything. Any field given a list of values
becomes a sweep axis, so

    model:
      pos_encoding: [rope, alibi, cope]
    train:
      seed: [1, 2]

is six runs, each in its own run_dir named after the axis values, with one
collected summary.csv next to them. Finished runs are skipped unless --rerun,
so an interrupted grid continues by running the same command again. Runs go one
after another on the card named in hardware.gpu.
"""

import argparse
from pathlib import Path

import yaml

from core.config import expand_configs, planned_run_dirs, run_paths, to_dict
from core.logs import collect_summaries

DEFAULT_CONFIG = "configs/main.yaml"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", action="append", metavar="FILE",
                        help=f"config file; repeatable, later files win "
                             f"(default: {DEFAULT_CONFIG})")
    parser.add_argument("--set", action="append", default=[], dest="overrides",
                        metavar="section.key=value", help="single value override")
    parser.add_argument("--rerun", action="store_true",
                        help="retrain runs that already have a summary.csv")
    parser.add_argument("--dry-run", action="store_true",
                        help="list the runs a config expands to and exit")
    return parser.parse_args()


def write_resolved(config):
    logs = run_paths(config.paths).logs
    logs.mkdir(parents=True, exist_ok=True)
    (logs / "config.resolved.yaml").write_text(
        yaml.safe_dump(to_dict(config), sort_keys=False), encoding="utf-8")


def train_one(config, position, rerun):
    run_dir = config.paths.run_dir
    if not rerun and (run_paths(config.paths).results / "summary.csv").is_file():
        print(f"{position} {run_dir} — already finished, skipping")
        return None
    print(f"{position} {run_dir}", flush=True)
    write_resolved(config)
    from core import train as trainer  # local: --dry-run and --help need no torch
    return trainer.run(config)


def train_guarded(config, position, rerun, guard):
    """Train one config; in a grid a failure is reported instead of raised."""
    try:
        return train_one(config, position, rerun)
    except Exception as error:  # noqa: BLE001 - one bad cell must not kill the grid
        if not guard:
            raise
        print(f"{position} FAILED: {type(error).__name__}: {error}", flush=True)
        return error


def report(configs, results):
    for config, result in zip(configs, results):
        outcome = (f"{result:.4f}" if isinstance(result, float)
                   else "skipped" if result is None else f"FAILED ({result})")
        print(f"  {outcome:>12}  {config.paths.run_dir}")
    if len(configs) > 1:
        results_dirs = [run_paths(c.paths).results for c in configs]
        table = collect_summaries(results_dirs, results_dirs[0].parent / "summary.csv")
        print(f"collected summary: {table}")
    return 1 if any(isinstance(r, Exception) for r in results) else 0


def main():
    args = parse_args()
    paths = args.config or [DEFAULT_CONFIG]
    if args.dry_run:
        # deliberately the torch-free path: listing a grid needs no model code
        planned = planned_run_dirs(paths, args.overrides)
        print(f"{len(planned)} run(s):")
        print(*[f"  {run_dir}" for run_dir in planned], sep="\n")
        return 0

    configs = expand_configs(paths, args.overrides)
    print(f"{len(configs)} run(s):")
    for config in configs:
        print(f"  {config.paths.run_dir}")
    grid = len(configs) > 1
    results = [train_guarded(config, f"[{index}/{len(configs)}]", args.rerun, guard=grid)
               for index, config in enumerate(configs, start=1)]
    return report(configs, results)


if __name__ == "__main__":
    raise SystemExit(main())
