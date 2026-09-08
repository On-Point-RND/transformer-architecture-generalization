import argparse
from pathlib import Path

import numpy as np

PAD, WALL, OPEN, ORIGIN, TARGET, PATH_START, PATH_END = range(7)
MOVE_U, MOVE_D, MOVE_L, MOVE_R = range(7, 11)
N_SPECIAL = 11

MOVES = {(-1, 0): MOVE_U, (1, 0): MOVE_D, (0, -1): MOVE_L, (0, 1): MOVE_R}


def coord_token(row, col, max_grid_n):
    return N_SPECIAL + row * max_grid_n + col


def encode(maze, max_grid_n):
    """One SolvedMaze -> (prompt, answer); layout as in the module docstring."""
    cells = np.where(maze.connection_list.reshape(-1), OPEN, WALL)
    head = [ORIGIN, coord_token(*maze.start_pos, max_grid_n),
            TARGET, coord_token(*maze.end_pos, max_grid_n),
            PATH_START]
    steps = np.diff(maze.solution, axis=0)
    moves = [MOVES[(int(dr), int(dc))] for dr, dc in steps]
    prompt = np.concatenate([cells, head]).astype(np.int64)
    answer = np.array(moves + [PATH_END], dtype=np.int64)
    return prompt, answer


def offsets(sequences):
    return np.cumsum([0] + [len(s) for s in sequences]).astype(np.int64)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--grid-n", type=int, default=5)
    parser.add_argument("--n-mazes", type=int, default=200_000)
    parser.add_argument("--max-grid-n", type=int, default=16,
                        help="sizes the coordinate vocabulary; keep it identical "
                             "across every artifact one model is evaluated on, or "
                             "check_architecture will refuse the checkpoint")
    parser.add_argument("--generator", default="gen_dfs",
                        choices=["gen_dfs", "gen_wilson"],
                        help="spanning-tree generators only: percolation admits "
                             "several shortest paths, so exact match is ill-defined")
    parser.add_argument("--min-path-length", type=int, default=3)
    parser.add_argument("-o", "--out", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.grid_n > args.max_grid_n:
        raise ValueError(f"grid_n {args.grid_n} exceeds max_grid_n {args.max_grid_n}")

    from maze_dataset import MazeDataset, MazeDatasetConfig
    from maze_dataset.generation import LatticeMazeGenerators

    config = MazeDatasetConfig(
        name=Path(args.out).stem,
        grid_n=args.grid_n,
        n_mazes=args.n_mazes,
        maze_ctor=getattr(LatticeMazeGenerators, args.generator),
    )
    dataset = MazeDataset.from_config(config)
    dataset = dataset.filter_by.path_length(min_length=args.min_path_length)

    prompts, answers, meta = [], [], []
    for maze in dataset:
        prompt, answer = encode(maze, args.max_grid_n)
        prompts.append(prompt)
        answers.append(answer)
        meta.append((maze.connection_list.shape[1], len(answer) - 1,
                     int(np.abs(np.asarray(maze.start_pos) - maze.end_pos).sum())))

    grid_n, path_length, manhattan = np.array(meta, dtype=np.int64).T
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        prompt_flat=np.concatenate(prompts),
        prompt_off=offsets(prompts),
        answer_flat=np.concatenate(answers),
        answer_off=offsets(answers),
        grid_n=grid_n,
        path_length=path_length,
        start_end_manhattan=manhattan,
        vocab_size=N_SPECIAL + args.max_grid_n ** 2,
        config=repr(config.serialize()),
    )
    longest = int((np.diff(offsets(prompts)) + np.diff(offsets(answers))).max())
    print(f"wrote {len(prompts)} mazes to {args.out}")
    print(f"vocab_size = {N_SPECIAL + args.max_grid_n ** 2}, "
          f"needs block_size >= {longest - 1}")


if __name__ == "__main__":
    main()
