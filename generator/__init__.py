"""
Generator registry for the sandbox.

Training/validation data is streamed live from a ``Generator`` — there are no
on-disk bins and no ``meta.pkl``. You pick a task at run time with the existing
``--dataset`` flag (reused as the generator name), e.g.:

    python train.py config/small.py --dataset=kv

Contract — each generator is a ``Generator`` subclass (see generator/base.py)
that implements:

    _sample_one() -> DatasetItem(prompt, answer)   # one prompt->answer example
    vocab_size                                      # property: number of token ids
    PAD_ID                                          # class attr, default 0

The base class provides the rest for free: ``sample_train``/``generate_val``
(deduped by prompt hash) and a generic answer-masked ``collate``. As long as a
generator honours that contract, train.py works with it unchanged.

Adding a task = add one file under generator/ and one entry in GENERATORS
below. The config name need not equal the filename, and related tasks may share
a single module.
"""

import importlib

from generator.positional_lab import POSITIONAL_TASK_DEFAULTS

# name -> (module path, class name, default constructor params)
GENERATORS = {
    "kv_retrieval": ("generator.kv_retrieval", "KVRetrievalGenerator", {}),
    "kv": ("generator.kv_retrieval", "KVRetrievalGenerator", {}),  # alias used by config/*.py
    "nested_kv_retrieval": (
        "generator.nested_kv_retrieval",
        "NestedKVRetrievalGenerator",
        {},
    ),
    "nested_kv": (
        "generator.nested_kv_retrieval",
        "NestedKVRetrievalGenerator",
        {},
    ),  # alias
    "addition": ("generator.addition", "AdditionGenerator", {}),
    "sorting": ("generator.sorting", "SortingGenerator", {}),
    "indexing": ("generator.indexing", "IndexingGenerator", {}),
    "dyck": ("generator.dyck", "DyckGenerator", {}),
    "function_composition": (
        "generator.function_composition",
        "FunctionCompositionGenerator",
        {},
    ),
    "positional_lab": (
        "generator.positional_lab",
        "PositionalLabGenerator",
        {},
    ),
}

# Positional laboratory variants are first-class datasets.  For example,
# ``--dataset=relative_offset_copy`` selects the task without an extra
# ``gen_params={'task': ...}`` layer.
for _task_name, _task_defaults in POSITIONAL_TASK_DEFAULTS.items():
    GENERATORS[_task_name] = (
        "generator.positional_lab",
        "PositionalLabGenerator",
        dict(_task_defaults),
    )


def get_generator(name, params=None):
    """Instantiate the generator registered under ``name``.

    ``params`` (a dict) is merged over the registry defaults and passed to the
    constructor. Raises a helpful ValueError listing known names on a miss.
    """
    if name not in GENERATORS:
        known = ", ".join(sorted(GENERATORS))
        raise ValueError(
            f"Unknown generator '{name}'. Known generators: {known}. "
            f"Register a new one in generator/__init__.py (see the contract there)."
        )
    module_path, class_name, defaults = GENERATORS[name]
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    merged = {**defaults, **(params or {})}
    return cls(**merged)
