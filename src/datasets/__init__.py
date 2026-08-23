"""Task registry.

Pick a task by name at run time:

    from datasets import get_task
    task = get_task("permutation")          # -> PermutationTask()

To add a task: create a generation module in this package and register its
``Task`` subclass (see datasets/base.py and datasets/tasks.py) in ``TASKS``.
The raw generation modules (addition, permutation, sorting, wordsort) are also
importable directly, e.g. ``from datasets import addition``.
"""

from datasets.base import Task
from datasets.tasks import AdditionTask, PermutationTask, SortingTask

TASKS = {
    "addition": AdditionTask,
    "permutation": PermutationTask,
    "perm": PermutationTask,      # alias
    "sorting": SortingTask,
    "sort": SortingTask,          # alias
}


def get_task(name: str) -> Task:
    """Instantiate the task registered under ``name``."""
    if name not in TASKS:
        known = ", ".join(sorted(TASKS))
        raise ValueError(f"Unknown task '{name}'. Known: {known}.")
    return TASKS[name]()
