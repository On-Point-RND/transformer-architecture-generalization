"""Experiment config: one YAML file, no includes.

A field comes from the file, else from the dataclass default below -- one
level, in one place. Lists under ``grid:`` are axes of a sweep; a list anywhere
else is data, such as a task's ``n_pairs: [2, 25]`` range.

    model:  {name: positional, n_layer: 4}
    task:   {name: kv_retrieval, params: {k_card: 80, v_card: 80, n_pairs: [2, 25]}}
    grid:
      model.pos_encoding: [nope, rope]
      task.params.n_pairs: [[2, 7], [2, 25]]
"""

import copy
import itertools
from ast import literal_eval
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


@dataclass
class ModelConfig:
    name: str = "vanilla"  # entry in models.MODELS
    block_size: int = 256
    vocab_size: Optional[int] = None  # filled in from the task's vocabulary
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.0
    bias: bool = False  # bias in Linears and LayerNorms


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
    ``/mnt/big/runs/EVE-PE/pe_lab/pos_encoding=rope/last.pt``), so grid points and
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
    learning_rate: float = 1e-4
    beta1: float = 0.9
    beta2: float = 0.95
    weight_decay: float = 1e-1
    decay: str = "matrices"  # 'matrices' (2D tensors only) | 'all'
    grad_clip: float = 1.0  # 0.0 disables
    schedule: str = "cosine"  # 'cosine' | 'constant'
    warmup_iters: int = 100
    lr_decay_iters: int = 80000
    min_lr: float = 1e-5


@dataclass
class TrainConfig:
    init: str = "scratch"  # 'scratch' | 'resume' | 'auto' (resume if last.pt exists)
    seed: int = 1337  # initialisation/shuffling seed
    data_seed: Optional[int] = None  # None = follow train.seed

    batch_size: int = 256
    gradient_accumulation_steps: int = 1
    max_iters: int = 80000

    eval_interval: int = 250
    eval_iters: int = 100
    eval_only: bool = False
    eval_accuracy: bool = True  # also compute the task's metrics at eval
    log_interval: int = 100
    diag_interval: int = 0  # weight/grad norms every N evals; 0 disables
    always_save_checkpoint: bool = True  # write last.pt at every eval
    reproducible_val: bool = False  # seeded val batches; off = legacy behaviour

    early_stop_metric: str = "val"  # 'val' | 'train' | a task metric, e.g. 'val_acc'
    early_stop_patience: int = 0  # evals without improvement before stopping; 0 disables
    early_stop_min_delta: float = 0.0  # a gain smaller than this does not count
    early_stop_target: Optional[float] = None  # stop once the metric is this good

    def __post_init__(self):
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


def set_path(sections: dict, key: str, value):
    """``set_path(s, "task.params.n_pairs", v)`` is ``s["task"]["params"]["n_pairs"] = v``."""
    section, *names = key.split(".")
    if section not in sections:
        raise ValueError(f"{key!r}: no section {section!r}; expected {'/'.join(SECTIONS)}")
    if not names:
        raise ValueError(f"{key!r}: name a field, e.g. {section}.<field>")
    node = sections[section]
    for name in names[:-1]:
        node = node.setdefault(name, {})
    node[names[-1]] = value


def parse_value(raw: str):
    """A --set/--grid value: a Python literal (1e-3, [2, 7], True), else YAML
    (so ``[nope, rope]`` needs no quotes), else the plain string."""
    try:
        return literal_eval(raw)
    except (SyntaxError, ValueError):
        pass
    try:
        return yaml.safe_load(raw)
    except yaml.YAMLError:
        return raw


def _assignment(text: str, flag: str):
    key, sep, raw = text.partition("=")
    if not sep or "." not in key:
        raise ValueError(f"{flag} expects 'section.field=value', got {text!r}")
    return key.strip(), parse_value(raw)


def read_config(path, overrides=(), grids=()):
    """The file plus CLI edits, as (sections, grid).

    ``--set key=value`` writes data and, if key was an axis, pins it. ``--grid
    key=[...]`` adds an axis. Both take the dotted keys the grid uses.
    """
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if "include" in raw:
        raise ValueError(f"{path}: 'include:' is gone; each config is self-contained "
                         f"and unset fields take the dataclass defaults in core/config.py")
    grid = dict(raw.pop("grid", None) or {})
    stray = sorted(set(raw) - set(SECTIONS))
    if stray:
        raise ValueError(f"{path}: unknown top-level section(s) {stray}; "
                         f"expected {'/'.join(SECTIONS)} or grid")
    sections = {name: dict(raw.get(name) or {}) for name in SECTIONS}
    for text in overrides:
        key, value = _assignment(text, "--set")
        set_path(sections, key, value)
        grid.pop(key, None)
    for text in grids:
        key, values = _assignment(text, "--grid")
        grid[key] = values
    for key, values in grid.items():
        if not isinstance(values, list):
            raise ValueError(f"grid {key}: expected a list of values, got {values!r}")
    return sections, grid


def _slug(value):
    return "".join(c if c.isalnum() or c in "+-._" else "-" for c in str(value))


def _label_value(value):
    if isinstance(value, (list, tuple)):
        return "-".join(_slug(v) for v in value)
    return _slug(value)


def expand(sections, grid):
    """One sections dict per grid point; run_dir gets ``name=value__...`` appended.

    An axis is named by the last segment of its key, or by the whole key when two
    axes share that segment (model.name and task.name).
    """
    keys = list(grid)
    short = [key.rsplit(".", 1)[-1] for key in keys]
    names = [key if short.count(name) > 1 else name for key, name in zip(keys, short)]
    points = []
    for values in itertools.product(*(grid[key] for key in keys)):
        point = copy.deepcopy(sections)
        for key, value in zip(keys, values):
            set_path(point, key, value)
        label = "__".join(f"{name}={_label_value(v)}" for name, v in zip(names, values))
        base = point["paths"].get("run_dir", PathsConfig.run_dir)
        point["paths"]["run_dir"] = f"{base}/{label}" if label else base
        points.append(point)
    return points


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


def _build_config(sections) -> Config:
    from models import get_model  # local import: models import this module
    model_config_cls = get_model(sections["model"].get("name", ModelConfig.name))[0]
    sections["task"]["params"] = _tuplify(sections["task"].get("params", {}))
    return Config(
        model=_build(model_config_cls, sections["model"], "model"),
        task=_build(TaskConfig, sections["task"], "task"),
        train=_build(TrainConfig, sections["train"], "train"),
        optimizer=_build(OptimizerConfig, sections["optimizer"], "optimizer"),
        hardware=_build(HardwareConfig, sections["hardware"], "hardware"),
        paths=_build(PathsConfig, sections["paths"], "paths"),
    )


def planned_run_dirs(path, overrides=(), grids=()):
    """Where each run would write, without importing the model code."""
    return [s["paths"]["run_dir"] for s in expand(*read_config(path, overrides, grids))]


def expand_configs(path, overrides=(), grids=()):
    return [_build_config(s) for s in expand(*read_config(path, overrides, grids))]


def to_dict(config) -> dict:
    if is_dataclass(config):
        return {f.name: to_dict(getattr(config, f.name)) for f in fields(config)}
    if isinstance(config, dict):
        return {k: to_dict(v) for k, v in config.items()}
    if isinstance(config, tuple):
        return list(config)
    return config
