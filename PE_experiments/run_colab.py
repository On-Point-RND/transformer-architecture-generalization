#!/usr/bin/env python3
"""Colab wrapper around PE_experiments/run_cluster.py.

When executed inside an uploaded/cloned repository it uses that checkout.
Otherwise it clones the configured GitHub branch into /content and forwards
all remaining arguments to the cluster launcher.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


DEFAULT_REPO_URL = "https://github.com/averageX56/SMILES-nanoGPT-sandbox.git"
DEFAULT_BRANCH = "AVE-branch"
DEFAULT_REPO_DIR = Path("/content/SMILES-nanoGPT-sandbox")


def is_repo(path):
    return (path / "train.py").is_file() and (path / "PE_experiments" / "run_cluster.py").is_file()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--repo-url", default=DEFAULT_REPO_URL)
    parser.add_argument("--branch", default=DEFAULT_BRANCH)
    parser.add_argument("--repo-dir", type=Path, default=DEFAULT_REPO_DIR)
    parser.add_argument("--pack-results", action="store_true")
    parser.add_argument("--help", action="store_true")
    return parser.parse_known_args()


def resolve_repo(args):
    local_checkout = Path(__file__).resolve().parents[1]
    if is_repo(local_checkout):
        return local_checkout
    target = args.repo_dir.expanduser().resolve()
    if is_repo(target):
        return target
    if target.exists() and any(target.iterdir()):
        raise RuntimeError(f"{target} exists but is not the expected repository")
    target.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        "git", "clone", "--depth", "1", "--branch", args.branch,
        args.repo_url, str(target),
    ], check=True)
    if not is_repo(target):
        raise RuntimeError("cloned branch does not contain PE_experiments/run_cluster.py")
    return target


def main():
    args, forwarded = parse_args()
    if args.help:
        print(__doc__)
        print("Colab options: --repo-url --branch --repo-dir --pack-results")
        print("Training options:")
        repo = resolve_repo(args)
        return subprocess.run([
            sys.executable, str(repo / "PE_experiments" / "run_cluster.py"), "--help"
        ]).returncode

    repo = resolve_repo(args)
    command = [sys.executable, str(repo / "PE_experiments" / "run_cluster.py"), *forwarded]
    result = subprocess.run(command, cwd=repo)
    if result.returncode:
        return result.returncode
    if args.pack_results:
        out_dir = repo / "out" / "pe_experiments"
        if out_dir.exists():
            archive = shutil.make_archive("/content/pe_experiments_results", "zip", out_dir)
            print(f"results archive: {archive}")
        else:
            print(f"no results directory to pack: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
