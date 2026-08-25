#!/usr/bin/env python3
"""Redraw a run's curves, or compare several runs.

    python plot.py runs/small                    # refresh curves.png
    python plot.py --compare runs/pe_lab/*/*     # one figure, one line per run
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from core.logs import read_metrics, series, write_curves  # noqa: E402

COMPARE_PANELS = (("val_loss", "validation loss"), ("val_acc", "validation exact-match"))


def run_label(records):
    labelled = [r for r in records if r.get("positional_encoding")]
    return labelled[-1]["positional_encoding"] if labelled else None


def compare(run_dirs, out_path):
    figure, axes = plt.subplots(1, len(COMPARE_PANELS), figsize=(6 * len(COMPARE_PANELS), 4.5))
    for run_dir in run_dirs:
        records = read_metrics(Path(run_dir) / "metrics.jsonl")
        label = run_label(records) or Path(run_dir).name
        for axis, (key, ylabel) in zip(axes, COMPARE_PANELS):
            axis.plot(*series(records, "eval", "iter", key), marker=".", label=label)
            axis.set_xlabel("iteration")
            axis.set_ylabel(ylabel)
            axis.grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(out_path, dpi=120)
    print(out_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--compare", action="store_true", help="one figure for all runs")
    parser.add_argument("-o", "--out", type=Path, default=Path("compare.png"))
    args = parser.parse_args()
    if args.compare:
        return compare(args.runs, args.out)
    for run_dir in args.runs:
        print(write_curves(run_dir / "metrics.jsonl", run_dir / "curves.png"))


if __name__ == "__main__":
    main()
