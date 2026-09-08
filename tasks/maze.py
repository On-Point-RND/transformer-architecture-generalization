import numpy as np

from .base import DatasetItem, Task


class MazeTask(Task):

    PAD_ID = 0
    WALL, OPEN, ORIGIN, TARGET, PATH_START, PATH_END = range(1, 7)
    MOVE_U, MOVE_D, MOVE_L, MOVE_R = range(7, 11)
    N_SPECIAL = 11

    def __init__(self, path: str, seed: int | None = 42):
        data = np.load(path)
        self.prompt_flat, self.prompt_off = data["prompt_flat"], data["prompt_off"]
        self.answer_flat, self.answer_off = data["answer_flat"], data["answer_off"]
        self.grid_n = data["grid_n"]
        self.path_length = data["path_length"]
        self.start_end_manhattan = data["start_end_manhattan"]
        self.vocab = int(data["vocab_size"])
        self.n_items = len(self.grid_n)
        self.rng = np.random.default_rng(seed)

    @property
    def vocab_size(self) -> int:
        return self.vocab

    @property
    def min_block_size(self) -> int:
        """len(prompt) + len(answer) <= block_size + 1, over the whole file."""
        spans = np.diff(self.prompt_off) + np.diff(self.answer_off)
        return int(spans.max()) - 1

    def metrics(self, predicted, targets):
        """Exact match plus per-move accuracy: 15 of 16 moves right is not 0."""
        scores = super().metrics(predicted, targets)
        answer = targets != -1
        scores["step_acc"] = float((predicted == targets)[answer].astype(np.float32).mean())
        return scores

    def _sample_one(self) -> DatasetItem:
        i = int(self.rng.integers(self.n_items))
        return DatasetItem(
            prompt=self.prompt_flat[self.prompt_off[i]:self.prompt_off[i + 1]],
            answer=self.answer_flat[self.answer_off[i]:self.answer_off[i + 1]],
            metadata={
                "grid_n": int(self.grid_n[i]),
                "path_length": int(self.path_length[i]),
                "start_end_manhattan": int(self.start_end_manhattan[i]),
            },
        )
