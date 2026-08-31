#!/usr/bin/env python3
"""Plot accuracy-vs-length curves from ``run_pose_self_training.py``.

The default input/output directory and filenames intentionally match the
original experiment so existing result files remain usable::

    python scripts/plot_length_curves.py
    python scripts/plot_length_curves.py --root /path/to/results
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_ROOT = Path(__file__).resolve().parents[1] / "saved" / "pose_fixmatch"

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE_AXIS = "#c3c2b7"
SERIES = ["#2a78d6", "#008300", "#e87ba4", "#eda100"]


def load_curve(root: Path, name: str):
    path = root / f"{name}.json"
    if not path.is_file():
        return None
    result = json.loads(path.read_text(encoding="utf-8"))
    accuracy = {int(length): value for length, value in result["in_domain"].items()}
    accuracy.update({int(length): value for length, value in result["ood"].items()})
    return dict(sorted(accuracy.items()))


def plot_family(root: Path, specs, train_max: int, title: str, out_name: str) -> bool:
    curves = [(label, load_curve(root, name)) for label, name in specs]
    curves = [(label, curve) for label, curve in curves if curve]
    if not curves:
        return False

    figure, axis = plt.subplots(figsize=(8.2, 4.6), dpi=200)
    figure.patch.set_facecolor(SURFACE)
    axis.set_facecolor(SURFACE)

    max_length = max(max(curve) for _, curve in curves)
    axis.axvspan(train_max + 0.5, max_length + 0.5, color=GRID, alpha=0.35, zorder=0)
    axis.text(train_max + 0.62, 1.06, "OOD \u2192", color=MUTED, fontsize=9, va="bottom")

    for index, (label, accuracy) in enumerate(curves):
        xs, ys = list(accuracy), list(accuracy.values())
        color = SERIES[index % len(SERIES)]
        axis.plot(
            xs,
            ys,
            color=color,
            linewidth=2,
            marker="o",
            markersize=5.5,
            markerfacecolor=color,
            markeredgecolor=SURFACE,
            markeredgewidth=1,
            zorder=3,
            label=label,
        )

    used_y = []
    for label, accuracy in curves:
        visible = [length for length, value in accuracy.items() if value >= 0.02]
        label_x = (
            visible[-1]
            if visible
            else min(accuracy, key=lambda length: abs(length - train_max))
        )
        point_y = accuracy[label_x]
        label_y = point_y
        while any(abs(label_y - occupied) < 0.07 for occupied in used_y):
            label_y += 0.075
        used_y.append(label_y)
        axis.annotate(
            label,
            (label_x, point_y),
            xytext=(8, (label_y - point_y) * 100 + 4),
            textcoords="offset points",
            color=SECONDARY,
            fontsize=9,
            fontweight="medium",
        )

    axis.set_xlim(0.5, max_length + 2.2)
    axis.set_ylim(-0.03, 1.12)
    axis.set_xticks(range(1, max_length + 1))
    axis.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    axis.grid(axis="y", color=GRID, linewidth=0.8, zorder=1)
    for spine in ("top", "right", "left"):
        axis.spines[spine].set_visible(False)
    axis.spines["bottom"].set_color(BASELINE_AXIS)
    axis.tick_params(colors=MUTED, labelsize=9)
    axis.set_xlabel("operand length (digits)", color=SECONDARY, fontsize=10)
    axis.set_ylabel("exact match", color=SECONDARY, fontsize=10)
    axis.set_title(title, color=INK, fontsize=12, loc="left", pad=14)
    axis.legend(loc="upper right", frameon=False, fontsize=9, labelcolor=SECONDARY)

    output = root / out_name
    figure.tight_layout()
    figure.savefig(output, facecolor=SURFACE, bbox_inches="tight")
    plt.close(figure)
    print(f"saved {output}")
    return True


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help=f"result JSON and output image directory (default: {DEFAULT_ROOT})",
    )
    parser.add_argument(
        "--family",
        choices=("all", "L5", "L10"),
        default="all",
        help="render both experiment families or only one",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    families = {
        "L10": (
            [
                ("baseline (abs PE)", "addition_baseline_L10"),
                ("PoSE", "addition_pose_L10"),
                ("PoSE + self-improve", "addition_pose_selfimprove_L10"),
            ],
            10,
            "ADDITION - exact match vs operand length (train 1-10, OOD 11-15)",
            "accuracy_vs_length_L10.png",
        ),
        "L5": (
            [
                ("baseline (abs PE)", "addition_baseline"),
                ("PoSE", "addition_pose"),
                ("PoSE + FixMatch", "addition_pose_fixmatch"),
                ("PoSE + FixMatch curr.", "addition_pose_fixmatch_curr"),
            ],
            5,
            "ADDITION - exact match vs operand length (train 1-5, OOD 6-15)",
            "accuracy_vs_length_L5.png",
        ),
    }
    selected = families if args.family == "all" else {args.family: families[args.family]}
    rendered = [plot_family(args.root, *specification) for specification in selected.values()]
    if any(rendered):
        return 0
    print(f"no matching result JSON files found under {args.root}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
