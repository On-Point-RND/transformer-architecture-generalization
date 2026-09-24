"""The complete task registry.

There are deliberately no legacy aliases: a config names one of the eight
tasks below, and its params are passed directly to that task's constructor.
"""

from .kv_retrieval import KVRetrievalTask
from .maze import MazeTask
from .permutation import PermutationTask
from .ruler import RulerTask
from .selective_count import SelectiveCountTask
from .sorting import StringSortingTask
from .state_based_recall import StateBasedRecallTask

TASKS = {
    "kv_retrieval": (KVRetrievalTask, {}),
    "selective_count": (SelectiveCountTask, {}),
    "sorting": (StringSortingTask, {}),
    "c5": (PermutationTask, {"group": "c5"}),
    "ruler": (RulerTask, {}),
    "s5": (PermutationTask, {"group": "s5"}),
    "state_based_recall": (StateBasedRecallTask, {}),
    "maze": (MazeTask, {}),
}


def get_task(name, params=None):
    """Build a registered task. Registry presets such as C5/S5 are fixed."""
    try:
        task_class, preset = TASKS[name]
    except KeyError:
        raise ValueError(
            f"unknown task {name!r}; choose one of: {', '.join(TASKS)}"
        ) from None
    return task_class(**{**(params or {}), **preset})
