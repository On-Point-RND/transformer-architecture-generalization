import itertools
from ast import literal_eval
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


@dataclass
class ModelConfig:
    name: str = "positional"  # file name in models/
    block_size: int = 1024
    vocab_size: Optional[int] = None  # filled in from the task's vocabulary
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True  # bias in Linears and LayerNorms


@dataclass
class TaskConfig:
    name: str = "kv_retrieval"
    params: Dict[str, Any] = field(default_factory=dict)
    n_val: int = 1000  # size of the fixed held-out validation set
    n_train: int = 0  # 0 = endless stream; >0 = train from that many examples


@dataclass
class HardwareConfig:
    """Where and how the run executes — nothing about the experiment itself."""
    device: str = "gpu"  # 'gpu' | 'cpu'
    gpu: int = 0  # which card, when device is 'gpu'
    dtype: str = "bfloat16"  # 'float32' | 'bfloat16' | 'float16'
    compile: bool = False


@dataclass
class PathsConfig:
    """Where a run writes. Empty roots keep everything inside run_dir.

    A root that is set gets run_dir appended (``checkpoints: /mnt/big`` ->
    ``/mnt/big/runs/main/pos_encoding=rope/last.pt``), so grid points and
    separate experiments cannot overwrite each other's files.
    """
    run_dir: str = "runs/run"
    logs: str = ""         # metrics.jsonl, config.resolved.yaml
    checkpoints: str = ""  # last.pt, best.pt
    results: str = ""      # summary.csv, curves.png


@dataclass
class OptimizerConfig:
    """How the parameters are updated. Which algorithm, and on what schedule."""
    name: str = "adamw"  # file/entry in optimizers/
    learning_rate: float = 6e-4
    # two scalars rather than a betas pair: a list written directly under a
    # section is a sweep axis, so `betas: [0.9, 0.95]` would mean two runs
    beta1: float = 0.9
    beta2: float = 0.95
    weight_decay: float = 1e-1
    decay: str = "matrices"  # 'matrices' (2D tensors only) | 'all'
    grad_clip: float = 1.0  # 0.0 disables
    schedule: str = "cosine"  # 'cosine' | 'constant'
    warmup_iters: int = 2000
    lr_decay_iters: int = 600000
    min_lr: float = 6e-5


@dataclass
class TrainConfig:
    init: str = "scratch"  # 'scratch' | 'resume' | 'auto' (resume if last.pt exists)
    seed: int = 1337  # initialisation/shuffling seed
    data_seed: Optional[int] = None  # None = follow train.seed

    batch_size: int = 12
    gradient_accumulation_steps: int = 40
    max_iters: int = 600000

    eval_interval: int = 2000
    eval_iters: int = 200
    eval_only: bool = False
    eval_accuracy: bool = True  # also compute the task's metrics at eval
    log_interval: int = 100
    diag_interval: int = 0  # weight/grad norms every N evals; 0 disables
    always_save_checkpoint: bool = True  # write last.pt at every eval
    reproducible_val: bool = False  # seeded val batches; off = legacy behaviour

    # Early stopping. Both triggers are off by default, so a config that does not
    # mention them trains for the full max_iters exactly as before.
    early_stop_metric: str = "val"  # 'val' | 'train' | a task metric, e.g. 'val_acc'
    early_stop_patience: int = 0  # evals without improvement before stopping; 0 disables
    early_stop_min_delta: float = 0.0  # a gain smaller than this does not count
    early_stop_target: Optional[float] = None  # stop once the metric is this good

    def __post_init__(self):
        # resolved here rather than in the training loop, so the number that was
        # actually used lands in config.resolved.yaml and in the checkpoint
        if self.data_seed is None:
            self.data_seed = self.seed


SECTIONS = ("model", "task", "train", "optimizer", "hardware", "paths")


@dataclass
class Config:
    model: ModelConfig
    task: TaskConfig
    train: TrainConfig
    optimizer: OptimizerConfig
    hardware: HardwareConfig
    paths: PathsConfig


@dataclass
class RunPaths:
    logs: Path
    checkpoints: Path
    results: Path


def run_paths(paths: PathsConfig) -> RunPaths:
    run_dir = Path(paths.run_dir)
    return RunPaths(*[Path(root) / run_dir if root else run_dir
                      for root in (paths.logs, paths.checkpoints, paths.results)])


LEAF_KEYS = {"params"}


def _merge(base: dict, extra: dict) -> dict:
    out = dict(base)
    for key, value in extra.items():
        mergeable = (isinstance(value, dict) and isinstance(out.get(key), dict)
                     and key not in LEAF_KEYS)
        out[key] = _merge(out[key], value) if mergeable else value
    return out


def _tuplify(value):
    if isinstance(value, list):
        return tuple(_tuplify(v) for v in value)
    if isinstance(value, dict):
        return {k: _tuplify(v) for k, v in value.items()}
    return value


def _build(cls, values: dict, where: str):
    known = {f.name for f in fields(cls)}
    unknown = sorted(set(values) - known)
    if unknown:
        raise ValueError(f"unknown key(s) {unknown} under '{where}:'; known: {sorted(known)}")
    return cls(**values)


def _set_override(sections: dict, assignment: str):
    key, _, raw = assignment.partition("=")
    section, _, name = key.strip().partition(".")
    if not name:
        raise ValueError(f"--set expects 'section.key=value', got {assignment!r}")
    if section not in sections:
        raise ValueError(f"--set {assignment}: no section {section!r}")
    try:
        value = literal_eval(raw)
    except (SyntaxError, ValueError):
        value = raw  # plain strings need no quoting
    sections[section][name] = value


def read_yaml(path) -> dict:
    """One config file plus whatever it ``include:``s, as a single dict.

    Includes are resolved relative to the including file and merged first, so a
    file always overrides what it pulls in.
    """
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    includes = raw.pop("include", [])
    includes = [includes] if isinstance(includes, str) else includes
    merged: dict = {}
    for name in includes:
        merged = _merge(merged, read_yaml(path.parent / name))
    return _merge(merged, raw)


def merge_sections(paths, overrides=()) -> dict:
    """Read the YAML files, apply --set, return {'model': ..., 'task': ..., 'train': ...}.

    Kept separate from ``load_config`` so config files can be inspected and
    diffed without importing torch.
    """
    merged: dict = {}
    for path in paths:
        merged = _merge(merged, read_yaml(path))
    sections = {name: dict(merged.get(name, {})) for name in SECTIONS}
    stray = sorted(set(merged) - set(sections))
    if stray:
        raise ValueError(f"unknown top-level section(s) {stray}; "
                         f"expected {'/'.join(SECTIONS)}")
    for assignment in overrides:
        _set_override(sections, assignment)
    return sections


def _build_config(sections) -> Config:
    from models import get_model  # local import: models import this module
    model_cls = get_model(sections["model"].get("name", ModelConfig.name))[0]
    sections["task"]["params"] = _tuplify(sections["task"].get("params", {}))
    return Config(
        model=_build(model_cls, sections["model"], "model"),
        task=_build(TaskConfig, sections["task"], "task"),
        train=_build(TrainConfig, sections["train"], "train"),
        optimizer=_build(OptimizerConfig, sections["optimizer"], "optimizer"),
        hardware=_build(HardwareConfig, sections["hardware"], "hardware"),
        paths=_build(PathsConfig, sections["paths"], "paths"),
    )


def planned_run_dirs(paths, overrides=()):
    """Where each run would write, without importing the model code."""
    expanded = expand_sections(merge_sections(paths, overrides))
    return [sections["paths"]["run_dir"] for _, sections in expanded]


def expand_configs(paths, overrides=()):
    expanded = expand_sections(merge_sections(paths, overrides))
    return [_build_config(sections) for _, sections in expanded]


def load_config(paths, overrides=()) -> Config:
    """The single Config this input describes; errors if it is a grid."""
    configs = expand_configs(paths, overrides)
    if len(configs) != 1:
        raise ValueError(f"this config expands to {len(configs)} runs; "
                         f"use expand_configs() or drop the list-valued fields")
    return configs[0]


def _slug(value):
    return "".join(c if c.isalnum() or c in "+-._" else "-" for c in str(value))


def _label(axes, combination):
    parts = []
    for (_, key, values), value in zip(axes, combination):
        scalar = isinstance(value, (str, int, float, bool))
        parts.append(f"{key}={_slug(value) if scalar else values.index(value) + 1}")
    return "__".join(parts)


def _with_values(sections, axes, combination, label):
    picked = {name: dict(values) for name, values in sections.items()}
    for (section, key, _), value in zip(axes, combination):
        picked[section][key] = value
    base = picked["paths"].get("run_dir", PathsConfig.run_dir)
    picked["paths"]["run_dir"] = f"{base}/{label}" if label else base
    return picked


def expand_sections(sections):
    """[(label, sections)] — one entry per combination of list-valued fields.

    Only fields written directly under model/task/train are axes; anything
    nested inside a value stays data.
    """
    axes = [(name, key, values) for name, section in sections.items()
            for key, values in section.items() if isinstance(values, list)]
    expanded = []
    # with no axes the product is one empty tuple, i.e. the config as written
    for combination in itertools.product(*[values for _, _, values in axes]):
        label = _label(axes, combination)
        expanded.append((label, _with_values(sections, axes, combination, label)))
    return expanded


def to_dict(config) -> dict:
    if is_dataclass(config):
        return {f.name: to_dict(getattr(config, f.name)) for f in fields(config)}
    if isinstance(config, dict):
        return {k: to_dict(v) for k, v in config.items()}
    if isinstance(config, tuple):
        return list(config)
    return config
