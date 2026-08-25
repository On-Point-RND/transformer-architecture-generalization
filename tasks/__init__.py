"""Task registry.

Training/validation data is streamed live from a ``Task`` — there are no
on-disk bins and no ``meta.pkl``. Pick a task by name in the config, e.g.

    task:
      name: kv
      params: {k_card: 80, v_card: 80, n_pairs: [2, 25]}

Contract — each task is a ``Task`` subclass (see tasks/base.py) that implements

    _sample_one() -> DatasetItem(prompt, answer)   # one prompt->answer example
    vocab_size                                      # property: number of token ids
    PAD_ID                                          # class attr, default 0

The base class provides the rest for free: ``sample_train``/``generate_val``
(deduped by prompt hash), the answer-masked ``collate`` and the rng state used
for exact resume.

Adding a task = add one file here and one entry in TASKS below. The config name
need not equal the filename, and related tasks may share a single module.
"""

import importlib

from tasks.positional_lab import POSITIONAL_TASK_DEFAULTS

# name -> (module path, class name, default constructor params)
TASKS = {
    "kv_retrieval": ("tasks.kv_retrieval", "KVRetrievalTask", {}),
    "kv": ("tasks.kv_retrieval", "KVRetrievalTask", {}),  # alias used by configs
    "nested_kv_retrieval": ("tasks.nested_kv_retrieval", "NestedKVRetrievalTask", {}),
    "nested_kv": ("tasks.nested_kv_retrieval", "NestedKVRetrievalTask", {}),  # alias
    "addition": ("tasks.addition", "AdditionTask", {}),
    "sorting": ("tasks.sorting", "SortingTask", {}),
    "indexing": ("tasks.indexing", "IndexingTask", {}),
    "dyck": ("tasks.dyck", "DyckTask", {}),
    "function_composition": ("tasks.function_composition", "FunctionCompositionTask", {}),
    "positional_lab": ("tasks.positional_lab", "PositionalLabTask", {}),
}

# Positional laboratory variants are first-class datasets. For example,
# ``task.name: relative_offset_copy`` selects the task without an extra
# ``params: {task: ...}`` layer.
for _task_name, _task_defaults in POSITIONAL_TASK_DEFAULTS.items():
    TASKS[_task_name] = ("tasks.positional_lab", "PositionalLabTask", dict(_task_defaults))


def get_task(name, params=None):
    """Instantiate the task registered under ``name``.

    ``params`` (a dict) is merged over the registry defaults and passed to the
    constructor. Raises a helpful ValueError listing known names on a miss.
    """
    if name not in TASKS:
        known = ", ".join(sorted(TASKS))
        raise ValueError(
            f"Unknown task '{name}'. Known tasks: {known}. "
            f"Register a new one in tasks/__init__.py (see the contract there)."
        )
    module_path, class_name, defaults = TASKS[name]
    cls = getattr(importlib.import_module(module_path), class_name)
    return cls(**{**defaults, **(params or {})})
