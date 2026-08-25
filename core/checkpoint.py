import os
import random
from pathlib import Path

import numpy as np
import torch

LAST = "last.pt"
BEST = "best.pt"


def rng_state():
    return {
        "torch": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }


def restore_rng(state):
    torch.set_rng_state(state["torch"])
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])
    if state["torch_cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def save(run_dir, name, payload):
    path = Path(run_dir) / name
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)
    return path


def load(run_dir, name, device):
    return torch.load(Path(run_dir) / name, map_location=device, weights_only=False)


def exists(run_dir, name=LAST):
    return (Path(run_dir) / name).is_file()


def strip_compile_prefix(state_dict):
    prefix = "_orig_mod."
    return {k.removeprefix(prefix): v for k, v in state_dict.items()}


def model_fields(checkpoint):
    config = checkpoint.get("config", {})
    model = config.get("model")
    if isinstance(model, dict):
        return dict(model)
    legacy = dict(checkpoint.get("model_args", {}))
    legacy["name"] = "positional" if model in (None, "positional") else model
    return legacy


def check_architecture(run_dir, checkpoint, model_config, fields):
    stored = model_fields(checkpoint)
    current = {f: getattr(model_config, f, None) for f in fields}
    differing = {f: (stored[f], current[f])
                 for f in fields if f in stored and stored[f] != current[f]}
    if not differing:
        return
    detail = ", ".join(f"{f}: checkpoint {a!r} vs config {b!r}"
                       for f, (a, b) in sorted(differing.items()))
    raise ValueError(
        f"cannot resume {run_dir}: the checkpoint was trained with a different "
        f"architecture ({detail}). Fix the config, or train into a new run_dir."
    )
