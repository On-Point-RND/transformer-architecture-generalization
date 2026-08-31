import math
import unittest

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from experiments import (
    evaluate_lengths,
    pseudo_label_greedy,
    pseudo_label_vote,
    train_fixmatch,
    train_self_improve,
    train_supervised,
)
from tasks.base import DatasetItem, Task


def item(answer, prompt=(1,)):
    return DatasetItem(
        prompt=np.asarray(prompt, dtype=np.int64),
        answer=np.asarray(answer, dtype=np.int64),
    )


class TinyTask(Task):
    PAD_ID = 0

    def __init__(self):
        self.rng = np.random.default_rng(0)

    @property
    def vocab_size(self):
        return 8

    def _sample_one(self):
        return item([4])


class TinyModel(nn.Module):
    """Predicts fixed tokens, with explicit PoSE views controlled by a cycle."""

    def __init__(self, *, scale=8.0, plain_tokens=(4,), pose_tokens=(4,)):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(8))
        self.scale = float(scale)
        self.plain_tokens = tuple(plain_tokens)
        self.pose_tokens = tuple(pose_tokens)
        self.pose_calls = 0

    def sample_pose_positions(self, batch_size, length, device):
        token = self.pose_tokens[self.pose_calls % len(self.pose_tokens)]
        self.pose_calls += 1
        # The fake model treats column zero as a view id. The production model
        # supplies strictly increasing in-range positions instead.
        return torch.full((batch_size, length), token, dtype=torch.long, device=device)

    def forward(self, idx, targets=None, positions=None):
        batch, width = idx.shape
        if targets is not None:
            logits = self.bias.view(1, 1, -1).expand(batch, width, -1)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-1,
            )
            return logits, loss

        if positions is None:
            step = min(width - 1, len(self.plain_tokens) - 1)
            tokens = torch.full(
                (batch,), self.plain_tokens[step], dtype=torch.long, device=idx.device
            )
        else:
            tokens = positions[:, 0].long()
        logits = self.bias.view(1, -1).expand(batch, -1)
        logits = logits + F.one_hot(tokens, logits.size(-1)) * self.scale
        return logits[:, None, :], None


class CountingSGD(torch.optim.SGD):
    def __init__(self, params, **kwargs):
        super().__init__(params, **kwargs)
        self.step_calls = 0

    def step(self, closure=None):
        self.step_calls += 1
        return super().step(closure)


class SelfTrainingTests(unittest.TestCase):
    def test_greedy_accepts_and_rejects_by_minimum_confidence(self):
        candidate = item([7])
        accepted = pseudo_label_greedy(
            TinyModel(scale=8.0), [candidate], confidence_threshold=0.99
        )
        rejected = pseudo_label_greedy(
            TinyModel(scale=1.0), [candidate], confidence_threshold=0.9
        )

        self.assertEqual(accepted.accepted, 1)
        self.assertEqual(accepted.items[0].answer.tolist(), [4])
        self.assertGreaterEqual(accepted.accepted_scores[0], 0.99)
        self.assertEqual(rejected.accepted, 0)
        self.assertEqual(rejected.rejected_score, 1)

    def test_pseudo_labels_ignore_ground_truth_values_but_keep_its_length(self):
        model = TinyModel(plain_tokens=(4, 5))
        first = pseudo_label_greedy(
            model, [item([1, 2])], confidence_threshold=None
        )
        second = pseudo_label_greedy(
            model, [item([6, 7])], confidence_threshold=None
        )

        self.assertEqual(first.items[0].answer.tolist(), [4, 5])
        self.assertEqual(second.items[0].answer.tolist(), [4, 5])

    def test_terminal_filter_rejects_missing_or_early_terminal(self):
        missing = pseudo_label_greedy(
            TinyModel(plain_tokens=(4,)),
            [item([0])],
            confidence_threshold=None,
            terminal_token_id=5,
        )
        early = pseudo_label_greedy(
            TinyModel(plain_tokens=(5, 4)),
            [item([0, 0])],
            confidence_threshold=None,
            terminal_token_id=5,
        )

        self.assertEqual((missing.accepted, missing.rejected_terminal), (0, 1))
        self.assertEqual((early.accepted, early.rejected_terminal), (0, 1))

    def test_vote_uses_plain_plus_explicit_pose_views(self):
        # Votes: plain=4, then PoSE=5,5,5,4. Token 5 wins three of five.
        model = TinyModel(plain_tokens=(4,), pose_tokens=(5, 5, 5, 4))
        result = pseudo_label_vote(model, [item([0])], n_votes=5, vote_min=3)

        self.assertEqual(result.accepted, 1)
        self.assertEqual(result.items[0].answer.tolist(), [5])
        self.assertAlmostEqual(result.accepted_scores[0], 3 / 5)
        self.assertEqual(model.pose_calls, 4)

        rejected = pseudo_label_vote(
            TinyModel(plain_tokens=(4,), pose_tokens=(5, 5, 5, 4)),
            [item([0])],
            n_votes=5,
            vote_min=4,
        )
        self.assertEqual(rejected.accepted, 0)

    def test_evaluate_lengths_reports_exact_match(self):
        def sampler(length, n):
            answer = [4] if length == 1 else [7]
            return [item(answer) for _ in range(n)]

        scores = evaluate_lengths(TinyModel(), sampler, [1, 2], n=3)
        self.assertEqual(scores, {1: 1.0, 2: 0.0})

    def test_one_step_supervised_and_fixmatch_smoke(self):
        task = TinyTask()
        supervised = lambda n: [item([4]) for _ in range(n)]

        baseline = TinyModel()
        baseline_opt = CountingSGD(baseline.parameters(), lr=0.01)
        baseline_info = train_supervised(
            baseline,
            task,
            baseline_opt,
            steps=1,
            batch_size=2,
            block_size=2,
            sampler=supervised,
        )
        self.assertTrue(math.isfinite(baseline_info["final_loss"]))
        self.assertEqual(baseline_opt.step_calls, 1)

        fixmatch = TinyModel()
        fixmatch_opt = CountingSGD(fixmatch.parameters(), lr=0.01)
        fixmatch_info = train_fixmatch(
            fixmatch,
            task,
            fixmatch_opt,
            steps=1,
            batch_size=2,
            block_size=2,
            supervised_sampler=supervised,
            unlabeled_sampler=lambda length, n: [item([7]) for _ in range(n)],
            unlabeled_min_length=2,
            unlabeled_max_length=2,
            unlabeled_batch_size=2,
            confidence_threshold=0.99,
        )
        self.assertTrue(math.isfinite(fixmatch_info["final_loss"]))
        self.assertEqual(fixmatch_info["accept_rate"], 1.0)
        self.assertEqual(fixmatch_opt.step_calls, 1)

    def test_self_improve_uses_exact_budget_and_advances_frontier(self):
        task = TinyTask()
        model = TinyModel()
        optimizer = CountingSGD(model.parameters(), lr=0.01)
        result = train_self_improve(
            model,
            task,
            optimizer,
            steps=3,
            supervised_steps=1,
            rounds=2,
            initial_frontier=2,
            batch_size=2,
            block_size=2,
            supervised_sampler=lambda n: [item([4]) for _ in range(n)],
            unlabeled_sampler=lambda length, n: [item([7]) for _ in range(n)],
            pseudo_per_round=2,
            pseudo_batch_size=2,
            n_votes=3,
            vote_min=2,
            answer_length_bounds=lambda length: (1, 1),
        )

        self.assertEqual(result["steps"], 3)
        self.assertEqual(optimizer.step_calls, 3)
        self.assertEqual(result["final_frontier"], 4)
        self.assertEqual([row["frontier"] for row in result["round_stats"]], [3, 4])
        self.assertEqual(result["pseudo_pool_size"], 4)


if __name__ == "__main__":
    unittest.main()
