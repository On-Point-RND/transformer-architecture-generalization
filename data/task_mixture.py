import random
from collections import Counter

import torch
from torch.utils.data import IterableDataset


class MultitaskIterableDataset(IterableDataset):
    """Mixes several Task generators into one infinite stream of
    (x, y) tensors for teacher-forced training.

    x is input_ids[:-1], y is input_ids[1:] with -100 everywhere except
    the answer span, so nn.CrossEntropyLoss(ignore_index=-100) only
    scores the model on the tokens it is actually supposed to predict.
    """

    def __init__(self, tasks, split, tokenizer, max_seq_len=2048, seed=42):
        self.tasks = tasks
        self.split = split
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.seed = seed

        self.attempted_counts = Counter()
        self.yielded_counts = Counter()

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        effective_seed = self.seed + worker_id * 1_000_003

        rng = random.Random(effective_seed)
        streams = {
            task.name: task.infinite_stream(self.split, seed=rng.randint(0, 100_000))
            for task in self.tasks
        }
        task_names = list(streams.keys())

        while True:
            name = rng.choice(task_names)
            example = next(streams[name])
            self.attempted_counts[name] += 1

            full_text = f"{example.prompt} {example.target}"
            token_ids = [self.tokenizer.BOS_ID] + self.tokenizer.encode(full_text) + [self.tokenizer.EOS_ID]

            if len(token_ids) > self.max_seq_len:
                continue

            x = torch.tensor(token_ids[:-1], dtype=torch.long)
            y = torch.tensor(token_ids[1:], dtype=torch.long)

            prompt_len = len(self.tokenizer.encode(example.prompt)) + 1
            y[:prompt_len] = -100

            self.yielded_counts[name] += 1
            yield x, y

    def skip_report(self) -> str:
        if not self.attempted_counts:
            return "No examples attempted yet."
        lines = []
        for name in sorted(self.attempted_counts):
            attempted = self.attempted_counts[name]
            yielded = self.yielded_counts[name]
            skip_rate = 1 - yielded / attempted if attempted else 0.0
            lines.append(f"  {name:12s} attempted={attempted:6d}  yielded={yielded:6d}  skip_rate={skip_rate:5.1%}")
        return "\n".join(lines)


def collate_fn(batch, pad_id: int):
    xs, ys = zip(*batch)
    max_len = max(x.size(0) for x in xs)

    x_padded = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    y_padded = torch.full((len(batch), max_len), -100, dtype=torch.long)
    for i, (x, y) in enumerate(batch):
        x_padded[i, : x.size(0)] = x
        y_padded[i, : y.size(0)] = y
    return x_padded, y_padded