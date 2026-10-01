"""Small, dependency-free git provenance capture for experiment artifacts."""

import json
import subprocess
from pathlib import Path


def _git(*args):
    result = subprocess.run(
        ["git", *args], text=True, capture_output=True, check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def git_info():
    status = _git("status", "--porcelain")
    return {
        "git_commit": _git("rev-parse", "HEAD"),
        "git_branch": _git("branch", "--show-current"),
        "git_dirty": bool(status),
    }


def write_git_artifacts(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    info_path = directory / "git-info.json"
    info_path.write_text(json.dumps(git_info(), indent=2) + "\n", encoding="utf-8")
    diff_path = directory / "git.diff"
    diff_path.write_text(_git("diff", "--binary") + "\n", encoding="utf-8")
    return info_path, diff_path
