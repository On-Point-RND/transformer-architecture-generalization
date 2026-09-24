"""Perfect-maze solving with data generated on the fly.

The prompt contains two edge tokens per cell (right, then down), followed by
the start and target coordinates. A randomized DFS creates a spanning tree, so
there is exactly one valid path and exact-match evaluation is well defined.
"""

from collections import deque

import numpy as np

from .base import DatasetItem, Task, max_int, sample_int, validate_int_spec


class MazeTask(Task):
    PAD_ID = 0
    WALL_ID = 1
    OPEN_ID = 2
    ORIGIN_ID = 3
    TARGET_ID = 4
    PATH_START_ID = 5
    PATH_END_ID = 6
    MOVE_UP_ID = 7
    MOVE_DOWN_ID = 8
    MOVE_LEFT_ID = 9
    MOVE_RIGHT_ID = 10
    N_SPECIAL = 11

    DIRECTIONS = (
        (-1, 0, MOVE_UP_ID),
        (1, 0, MOVE_DOWN_ID),
        (0, -1, MOVE_LEFT_ID),
        (0, 1, MOVE_RIGHT_ID),
    )

    def __init__(
        self,
        grid_size: int | tuple[int, int] = 5,
        max_grid_size: int | None = None,
        min_path_length: int = 3,
        seed: int | None = 42,
    ):
        super().__init__(seed)
        self.grid_size = validate_int_spec(grid_size, "grid_size", 2)
        self.max_grid_size = (
            max_int(self.grid_size)
            if max_grid_size is None
            else validate_int_spec(max_grid_size, "max_grid_size", 2)
        )
        if max_int(self.grid_size) > self.max_grid_size:
            raise ValueError("max_grid_size must cover every sampled grid_size")
        self.min_path_length = validate_int_spec(min_path_length, "min_path_length", 1)
        smallest_grid = self.grid_size if isinstance(self.grid_size, int) else self.grid_size[0]
        if self.min_path_length >= smallest_grid ** 2:
            raise ValueError(
                "min_path_length must be smaller than the number of cells in every grid"
            )

    @property
    def vocab_size(self) -> int:
        return self.N_SPECIAL + self.max_grid_size ** 2

    def _coordinate_token(self, row: int, col: int) -> int:
        return self.N_SPECIAL + row * self.max_grid_size + col

    def _generate_tree(self, n: int):
        adjacency = [set() for _ in range(n * n)]
        start = int(self.rng.integers(n * n))
        visited = {start}
        stack = [start]
        while stack:
            cell = stack[-1]
            row, col = divmod(cell, n)
            candidates = []
            for dr, dc, _ in self.DIRECTIONS:
                nr, nc = row + dr, col + dc
                neighbour = nr * n + nc
                if 0 <= nr < n and 0 <= nc < n and neighbour not in visited:
                    candidates.append(neighbour)
            if not candidates:
                stack.pop()
                continue
            neighbour = candidates[int(self.rng.integers(len(candidates)))]
            adjacency[cell].add(neighbour)
            adjacency[neighbour].add(cell)
            visited.add(neighbour)
            stack.append(neighbour)
        return adjacency

    @staticmethod
    def _path(adjacency, start: int, target: int):
        parent = {start: None}
        queue = deque([start])
        while queue:
            cell = queue.popleft()
            if cell == target:
                break
            for neighbour in adjacency[cell]:
                if neighbour not in parent:
                    parent[neighbour] = cell
                    queue.append(neighbour)
        path = [target]
        while path[-1] != start:
            path.append(parent[path[-1]])
        return path[::-1]

    def _draw_endpoints(self, adjacency, n):
        for _ in range(1_000):
            start, target = self.rng.choice(n * n, size=2, replace=False).tolist()
            path = self._path(adjacency, start, target)
            if len(path) - 1 >= self.min_path_length:
                return start, target, path
        raise RuntimeError("could not draw maze endpoints satisfying min_path_length")

    def _sample_one(self) -> DatasetItem:
        n = sample_int(self.rng, self.grid_size)
        adjacency = self._generate_tree(n)
        start, target, path = self._draw_endpoints(adjacency, n)

        edges = []
        for cell in range(n * n):
            row, col = divmod(cell, n)
            right = cell + 1 if col + 1 < n else None
            down = cell + n if row + 1 < n else None
            edges += [
                self.OPEN_ID if right in adjacency[cell] else self.WALL_ID,
                self.OPEN_ID if down in adjacency[cell] else self.WALL_ID,
            ]

        moves = []
        move_by_delta = {(dr, dc): token for dr, dc, token in self.DIRECTIONS}
        for first, second in zip(path, path[1:]):
            r1, c1 = divmod(first, n)
            r2, c2 = divmod(second, n)
            moves.append(move_by_delta[(r2 - r1, c2 - c1)])

        start_row, start_col = divmod(start, n)
        target_row, target_col = divmod(target, n)
        prompt = np.asarray(
            edges
            + [
                self.ORIGIN_ID,
                self._coordinate_token(start_row, start_col),
                self.TARGET_ID,
                self._coordinate_token(target_row, target_col),
                self.PATH_START_ID,
            ],
            dtype=np.int64,
        )
        answer = np.asarray([*moves, self.PATH_END_ID], dtype=np.int64)
        return DatasetItem(
            prompt,
            answer,
            metadata={
                "grid_size": n,
                "path_length": len(moves),
                "start": (start_row, start_col),
                "target": (target_row, target_col),
                "start_end_manhattan": abs(start_row - target_row) + abs(start_col - target_col),
            },
        )
