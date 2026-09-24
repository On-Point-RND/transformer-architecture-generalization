import argparse
from pathlib import Path

import yaml

from core.config import expand_configs, planned_run_dirs, run_paths, to_dict
from core.logs import collect_summaries

DEFAULT_CONFIG = "configs/main.yaml"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=DEFAULT_CONFIG, metavar="FILE",
                        help=f"the experiment's config file (default: {DEFAULT_CONFIG})")
    parser.add_argument("--set", action="append", default=[], dest="overrides",
                        metavar="key=value",
                        help="set a value, e.g. task.params.n_pairs=[2,7]; pins an axis")
    parser.add_argument("--grid", action="append", default=[], dest="grids",
                        metavar="key=[...]",
                        help="add an axis, e.g. train.seed=[0,1,2]")
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
    from core import train as trainer  
    return trainer.run(config)


def train_guarded(config, position, rerun, guard):
    """Train one config; in a grid a failure is reported instead of raised."""
    try:
        return train_one(config, position, rerun)
    except Exception as error:
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
    if args.dry_run:
        planned = planned_run_dirs(args.config, args.overrides, args.grids)
        print(f"{len(planned)} run(s):")
        print(*[f"  {run_dir}" for run_dir in planned], sep="\n")
        return 0

    configs = expand_configs(args.config, args.overrides, args.grids)
    print(f"{len(configs)} run(s):")
    for config in configs:
        print(f"  {config.paths.run_dir}")
    grid = len(configs) > 1
    results = [train_guarded(config, f"[{index}/{len(configs)}]", args.rerun, guard=grid)
               for index, config in enumerate(configs, start=1)]
    return report(configs, results)


if __name__ == "__main__":
    raise SystemExit(main())
