"""Solve DFS mazes in the maze-dataset format (Ivanitskiy et al., 2023).

Input::

    <ADJLIST_START> (r,c) <--> (r,c) ; ... <ADJLIST_END>
    <ORIGIN_START> (r,c) <ORIGIN_END>
    <TARGET_START> (r,c) <TARGET_END> <PATH_START>

Output::

    -> (r,c) ... (r,c) <PATH_END>

The input contains the shuffled edges of an ``n x n`` DFS maze tree; edge
directions are shuffled too. The target is its unique origin-to-target path,
including both endpoints. Each coordinate is one token.

Generation matches maze-dataset's default ``gen_dfs`` and random endpoints
(distinct, with no minimum distance). Tokenization matches
``MazeTokenizerModular()`` / legacy ``AOTP_UT_uniform``. Unlike the library,
only the answer contributes to the loss and coordinate ids are row-major.

Lengths: prompt = ``4n^2 + 5``; answer <= ``n^2 + 1``; therefore
``block_size >= 5n^2 + 5`` (130 for a 5x5 maze).
"""

from collections import deque

import numpy as np

from .base import DatasetItem, Task, max_int, sample_int, validate_int_spec


class MazeTask(Task):
    PAD_ID = 0
    CONNECTOR_ID = 1  # <-->
    ENDLINE_ID = 2  # ;
    ADJLIST_START_ID = 3
    ADJLIST_END_ID = 4
    ORIGIN_START_ID = 5
    ORIGIN_END_ID = 6
    TARGET_START_ID = 7
    TARGET_END_ID = 8
    PATH_START_ID = 9
    PATH_END_ID = 10
    N_SPECIAL = 11

    DIRECTIONS = ((-1, 0), (1, 0), (0, -1), (0, 1))

    def __init__(
        self,
        grid_size: int | tuple[int, int] = 5,
        max_grid_size: int | None = None,
        min_path_length: int = 1,
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
        # as maze-dataset's _random_start_coord: randint(0, n - 1) excludes its
        # upper bound, so the search never starts in the last row or column
        row, col = self.rng.integers(0, max(n - 1, 1), size=2)
        start = int(row) * n + int(col)
        visited = {start}
        stack = [start]
        while stack:
            cell = stack[-1]
            row, col = divmod(cell, n)
            candidates = []
            for dr, dc in self.DIRECTIONS:
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

        def cell_token(cell):
            return self._coordinate_token(*divmod(cell, n))

        # each passage once, then a random order and a random orientation per pair
        edges = sorted((a, b) for a in range(n * n) for b in adjacency[a] if a < b)
        order = self.rng.permutation(len(edges))
        swap = self.rng.random(len(edges)) < 0.5
        adjlist = [self.ADJLIST_START_ID]
        for index, flipped in zip(order, swap):
            a, b = edges[index][::-1] if flipped else edges[index]
            adjlist += [cell_token(a), self.CONNECTOR_ID, cell_token(b), self.ENDLINE_ID]
        adjlist.append(self.ADJLIST_END_ID)

        prompt = np.asarray(
            adjlist
            + [
                self.ORIGIN_START_ID, cell_token(start), self.ORIGIN_END_ID,
                self.TARGET_START_ID, cell_token(target), self.TARGET_END_ID,
                self.PATH_START_ID,
            ],
            dtype=np.int64,
        )
        answer = np.asarray([cell_token(c) for c in path] + [self.PATH_END_ID], dtype=np.int64)
        start_row, start_col = divmod(start, n)
        target_row, target_col = divmod(target, n)
        return DatasetItem(
            prompt,
            answer,
            metadata={
                "grid_size": n,
                "path_length": len(path) - 1,
                "start": (start_row, start_col),
                "target": (target_row, target_col),
                "start_end_manhattan": abs(start_row - target_row) + abs(start_col - target_col),
            },
        )
