import importlib

from tasks.positional_lab import POSITIONAL_TASK_DEFAULTS

TASKS = {
    "kv_retrieval": ("tasks.kv_retrieval", "KVRetrievalTask", {}),
    "kv": ("tasks.kv_retrieval", "KVRetrievalTask", {}),
    "nested_kv_retrieval": ("tasks.nested_kv_retrieval", "NestedKVRetrievalTask", {}),
    "nested_kv": ("tasks.nested_kv_retrieval", "NestedKVRetrievalTask", {}),
    "addition": ("tasks.addition", "AdditionTask", {}),
    "sorting": ("tasks.sorting", "SortingTask", {}),
    "string_sorting": ("tasks.sorting", "StringSortingTask", {}),
    "permutation": ("tasks.permutation", "PermutationTask", {}),
    "s5": ("tasks.permutation", "PermutationTask", {"variant": "S5"}),
    "c5": ("tasks.permutation", "PermutationTask", {"variant": "C5"}),
    "add_seq": ("tasks.addition", "AdditionTask", {"carry": True, "bos_eos": True}),
    "add_indep": ("tasks.addition", "AdditionTask", {"carry": False, "bos_eos": True}),
    "indexing": ("tasks.indexing", "IndexingTask", {}),
    "dyck": ("tasks.dyck", "DyckTask", {}),
    "maze": ("tasks.maze", "MazeTask", {}),
    "ltl": ("tasks.ltl", "LTLTask", {}),
    "ltl_local": ("tasks.ltl", "LTLTask", {"mode": "local"}),
    "ltl_global": ("tasks.ltl", "LTLTask", {"mode": "global"}),
    "function_composition": ("tasks.function_composition", "FunctionCompositionTask", {}),
    "positional_lab": ("tasks.positional_lab", "PositionalLabTask", {}),
}


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
