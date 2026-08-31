"""PoSE-aware pseudo-label training for the repository's prompt/answer tasks.

There is deliberately no EOS assumption here.  An unlabeled ``DatasetItem``
contributes its prompt and only the *length* of its answer; its answer token
values are never inspected while a pseudo-label is made.  The model is greedily
decoded for exactly that many steps.
"""

from __future__ import annotations

import copy
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_

from tasks.base import DatasetItem, Task

LengthSampler = Callable[[int, int], Sequence[DatasetItem]]
BatchSampler = Callable[[int], Sequence[DatasetItem]]
AnswerLengthBounds = Callable[[int], tuple[int, int]]


@dataclass
class PseudoLabelResult:
    """Accepted pseudo-items and aggregate rejection accounting."""

    items: list[DatasetItem] = field(default_factory=list)
    attempted: int = 0
    rejected_length: int = 0
    rejected_terminal: int = 0
    rejected_score: int = 0
    accepted_scores: list[float] = field(default_factory=list)

    @property
    def accepted(self) -> int:
        return len(self.items)

    @property
    def acceptance_rate(self) -> float:
        return self.accepted / self.attempted if self.attempted else 0.0


def _check_bounds(bounds: tuple[int, int] | None) -> tuple[int, int] | None:
    if bounds is None:
        return None
    low, high = map(int, bounds)
    if low < 1 or high < low:
        raise ValueError(f"answer-length bounds must satisfy 1 <= low <= high, got {bounds}")
    return low, high


def _shape(item: DatasetItem) -> tuple[np.ndarray, int]:
    prompt = np.asarray(item.prompt, dtype=np.int64)
    if prompt.ndim != 1 or not len(prompt):
        raise ValueError("each pseudo-label prompt must be a non-empty 1-D token array")
    # This is the only access to the unlabeled answer during pseudo-labeling.
    answer_length = len(item.answer)
    if answer_length < 1:
        raise ValueError("fixed-duration decoding requires a non-empty answer")
    return prompt, answer_length


def _logits(output):
    value = output[0] if isinstance(output, tuple) else output
    if not isinstance(value, torch.Tensor) or value.ndim != 3:
        raise TypeError("model forward must return logits [B, T, V], optionally in a tuple")
    return value


def _decode_group(model, prompts, answer_length, device, pose_view=False):
    prompt_ids = torch.as_tensor(np.stack(prompts), dtype=torch.long, device=device)
    batch, prompt_length = prompt_ids.shape
    max_input_length = prompt_length + answer_length - 1
    positions = None
    if pose_view:
        positions = model.sample_pose_positions(batch, max_input_length, device)

    sequence = prompt_ids
    answer = []
    minimum_confidence = torch.ones(batch, device=device)
    for _ in range(answer_length):
        current_positions = None if positions is None else positions[:, : sequence.size(1)]
        step_logits = _logits(model(sequence, positions=current_positions))[:, -1, :]
        probabilities = torch.softmax(step_logits, dim=-1)
        confidence, token = probabilities.max(dim=-1)
        minimum_confidence = torch.minimum(minimum_confidence, confidence)
        answer.append(token)
        sequence = torch.cat((sequence, token[:, None]), dim=1)
    return torch.stack(answer, dim=1).cpu().numpy(), minimum_confidence.cpu().tolist()


def _predictions(model, items, device, pose_view=False, batch_size=None):
    if batch_size is not None and batch_size < 1:
        raise ValueError("batch_size must be positive or None")
    groups = defaultdict(list)
    for index, item in enumerate(items):
        prompt, answer_length = _shape(item)
        groups[(len(prompt), answer_length)].append((index, prompt))

    answers, scores = [None] * len(items), [None] * len(items)
    for (_, answer_length), group in groups.items():
        chunk_size = batch_size or len(group)
        for start in range(0, len(group), chunk_size):
            chunk = group[start:start + chunk_size]
            decoded, confidence = _decode_group(
                model, [prompt for _, prompt in chunk], answer_length, device, pose_view
            )
            for row, (index, _) in enumerate(chunk):
                answers[index] = decoded[row]
                scores[index] = confidence[row]
    return answers, scores


def _pseudo_item(source, answer, method, score):
    metadata = dict(source.metadata)
    metadata.update(pseudo_label=True, pseudo_method=method, pseudo_score=float(score))
    return DatasetItem(
        prompt=np.asarray(source.prompt, dtype=np.int64).copy(),
        answer=np.asarray(answer, dtype=np.int64).copy(),
        metadata=metadata,
    )


def _has_valid_terminal(answer, terminal_token_id):
    if terminal_token_id is None:
        return True
    answer = np.asarray(answer)
    ends_correctly = answer[-1] == terminal_token_id
    has_early_terminal = np.any(answer[:-1] == terminal_token_id)
    return bool(ends_correctly and not has_early_terminal)


@torch.no_grad()
def pseudo_label_greedy(
    model,
    items: Sequence[DatasetItem],
    *,
    device: str = "cpu",
    confidence_threshold: float | None = 0.95,
    length_bounds: tuple[int, int] | None = None,
    terminal_token_id: int | None = None,
) -> PseudoLabelResult:
    """Weak-view greedy labels accepted by minimum per-token confidence.

    ``length_bounds`` counts the complete task-specific answer, including a
    task-specific EOS when present.
    """
    items = list(items)
    bounds = _check_bounds(length_bounds)
    if confidence_threshold is not None and not 0 <= confidence_threshold <= 1:
        raise ValueError("confidence_threshold must be in [0, 1] or None")
    result = PseudoLabelResult(attempted=len(items))
    eligible = []
    for item in items:
        _, answer_length = _shape(item)
        if bounds is not None and not bounds[0] <= answer_length <= bounds[1]:
            result.rejected_length += 1
        else:
            eligible.append(item)

    was_training = model.training
    model.eval()
    try:
        answers, confidences = _predictions(model, eligible, device)
    finally:
        model.train(was_training)
    for item, answer, confidence in zip(eligible, answers, confidences):
        if not _has_valid_terminal(answer, terminal_token_id):
            result.rejected_terminal += 1
            continue
        if confidence_threshold is not None and confidence < confidence_threshold:
            result.rejected_score += 1
            continue
        result.items.append(_pseudo_item(item, answer, "greedy", confidence))
        result.accepted_scores.append(float(confidence))
    return result


@torch.no_grad()
def pseudo_label_vote(
    model,
    items: Sequence[DatasetItem],
    *,
    device: str = "cpu",
    n_votes: int = 5,
    vote_min: int = 3,
    length_bounds: tuple[int, int] | None = None,
    terminal_token_id: int | None = None,
    batch_size: int = 128,
) -> PseudoLabelResult:
    """Frozen-teacher labels voted across one plain and PoSE position views."""
    if n_votes < 1 or not 1 <= vote_min <= n_votes:
        raise ValueError("require n_votes >= 1 and 1 <= vote_min <= n_votes")
    items = list(items)
    bounds = _check_bounds(length_bounds)
    result = PseudoLabelResult(attempted=len(items))
    eligible = []
    for item in items:
        _, answer_length = _shape(item)
        if bounds is not None and not bounds[0] <= answer_length <= bounds[1]:
            result.rejected_length += 1
        else:
            eligible.append(item)

    votes = [[] for _ in eligible]
    was_training = model.training
    model.eval()
    try:
        for vote in range(n_votes):
            answers, _ = _predictions(
                model, eligible, device, pose_view=vote > 0, batch_size=batch_size
            )
            for row, answer in enumerate(answers):
                if _has_valid_terminal(answer, terminal_token_id):
                    votes[row].append(tuple(int(token) for token in answer))
    finally:
        model.train(was_training)

    for item, choices in zip(eligible, votes):
        if not choices:
            result.rejected_terminal += 1
            continue
        answer, count = Counter(choices).most_common(1)[0]
        score = count / n_votes
        if count < vote_min:
            result.rejected_score += 1
            continue
        result.items.append(_pseudo_item(item, answer, "vote", score))
        result.accepted_scores.append(score)
    return result


def _draw(sampler, count, name):
    items = list(sampler(count))
    if not items:
        raise ValueError(f"{name} returned no items")
    return items


def _loss_on_items(model, task, items, block_size, device, explicit_pose=False):
    x_np, y_np = task.collate(items, block_size)
    x = torch.as_tensor(x_np, dtype=torch.long, device=device)
    y = torch.as_tensor(y_np, dtype=torch.long, device=device)
    positions = None
    if explicit_pose:
        positions = torch.zeros_like(x)
        rows_by_length = defaultdict(list)
        for row, item in enumerate(items):
            valid = len(item.prompt) + len(item.answer) - 1
            rows_by_length[valid].append(row)
        for valid, rows in rows_by_length.items():
            sampled = model.sample_pose_positions(len(rows), valid, device)
            positions[rows, :valid] = sampled
    output = model(x, y, positions=positions) if explicit_pose else model(x, y)
    loss = output[1] if isinstance(output, tuple) else None
    if loss is None:
        raise TypeError("training forward must return (logits, loss)")
    return loss


def _optimise(model, optimizer, loss, scheduler, grad_clip):
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    if grad_clip:
        clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
    if scheduler is not None:
        scheduler.step()


def _final_loss(losses):
    return float(np.mean(losses[-min(250, len(losses)) :]))


def train_supervised(
    model,
    task: Task,
    optimizer,
    *,
    steps: int,
    batch_size: int,
    block_size: int,
    device: str = "cpu",
    sampler: BatchSampler | None = None,
    scheduler=None,
    grad_clip: float = 1.0,
) -> dict:
    """Minimal supervised baseline using the current ``Task.collate`` contract."""
    if steps < 1:
        raise ValueError("steps must be positive")
    sampler = sampler or task.sample_train
    model.train()
    losses = []
    for _ in range(steps):
        loss = _loss_on_items(
            model, task, _draw(sampler, batch_size, "sampler"), block_size, device
        )
        _optimise(model, optimizer, loss, scheduler, grad_clip)
        losses.append(float(loss.detach()))
    return {"final_loss": _final_loss(losses), "steps": steps}


@torch.no_grad()
def evaluate_lengths(
    model,
    sampler: LengthSampler,
    lengths: Iterable[int],
    *,
    n: int,
    device: str = "cpu",
) -> dict[int, float]:
    """Greedy exact answer match for caller-controlled per-length samples."""
    if n < 1:
        raise ValueError("n must be positive")
    was_training = model.training
    model.eval()
    scores = {}
    try:
        for length in lengths:
            items = list(sampler(int(length), n))
            if not items:
                raise ValueError(f"evaluation sampler returned no items at length {length}")
            answers, _ = _predictions(model, items, device)
            correct = sum(
                np.array_equal(answer, np.asarray(item.answer, dtype=np.int64))
                for answer, item in zip(answers, items)
            )
            scores[int(length)] = correct / len(items)
    finally:
        model.train(was_training)
    return scores


def _eval_snapshot(model, sampler, lengths, n, device, step, frontier):
    if sampler is None or not lengths:
        return None
    return {
        "step": step,
        "frontier": frontier,
        "acc": evaluate_lengths(model, sampler, lengths, n=n, device=device),
    }


def train_fixmatch(
    model,
    task: Task,
    optimizer,
    *,
    steps: int,
    batch_size: int,
    block_size: int,
    unlabeled_sampler: LengthSampler,
    unlabeled_min_length: int,
    unlabeled_max_length: int,
    unlabeled_batch_size: int = 128,
    unlabeled_every: int = 1,
    confidence_threshold: float | None = 0.95,
    lambda_u: float = 1.0,
    answer_length_bounds: AnswerLengthBounds | None = None,
    terminal_token_id: int | None = None,
    curriculum: bool = False,
    advance_threshold: float = 0.7,
    accept_ema_beta: float = 0.98,
    acceptance_window: int = 250,
    device: str = "cpu",
    seed: int = 0,
    supervised_sampler: BatchSampler | None = None,
    scheduler=None,
    grad_clip: float = 1.0,
    eval_sampler: LengthSampler | None = None,
    eval_lengths: Iterable[int] = (),
    eval_every: int = 0,
    eval_n: int = 100,
) -> dict:
    """Supervised training plus confidence-filtered OOD pseudo-labels."""
    if steps < 1 or unlabeled_every < 1 or acceptance_window < 1:
        raise ValueError("steps, unlabeled_every, and acceptance_window must be positive")
    if unlabeled_min_length > unlabeled_max_length:
        raise ValueError("unlabeled_min_length must not exceed unlabeled_max_length")
    supervised_sampler = supervised_sampler or task.sample_train
    rng = np.random.default_rng(seed)
    frontier = unlabeled_min_length if curriculum else unlabeled_max_length
    frontier_ema = 0.0
    losses, acceptance_history, history = [], [], []
    accepted_total = attempted_total = 0
    model.train()

    for step in range(1, steps + 1):
        supervised = _draw(supervised_sampler, batch_size, "supervised_sampler")
        loss = _loss_on_items(model, task, supervised, block_size, device)
        if step % unlabeled_every == 0:
            length = int(rng.integers(unlabeled_min_length, frontier + 1))
            candidates = list(unlabeled_sampler(length, unlabeled_batch_size))
            bounds = answer_length_bounds(length) if answer_length_bounds else None
            pseudo = pseudo_label_greedy(
                model, candidates, device=device,
                confidence_threshold=confidence_threshold, length_bounds=bounds,
                terminal_token_id=terminal_token_id,
            )
            accepted_total += pseudo.accepted
            attempted_total += pseudo.attempted
            rate = pseudo.acceptance_rate
            acceptance_history.append(
                {"step": step, "length": length, "accepted": pseudo.accepted,
                 "attempted": pseudo.attempted, "rate": rate, "frontier": frontier}
            )
            if pseudo.items:
                loss = loss + lambda_u * _loss_on_items(
                    model, task, pseudo.items, block_size, device, explicit_pose=True
                )
            if curriculum and length == frontier:
                frontier_ema = accept_ema_beta * frontier_ema + (1 - accept_ema_beta) * rate
                if frontier_ema >= advance_threshold and frontier < unlabeled_max_length:
                    frontier += 1
                    frontier_ema = 0.0

        _optimise(model, optimizer, loss, scheduler, grad_clip)
        losses.append(float(loss.detach()))
        if eval_every and (step % eval_every == 0 or step == steps):
            snapshot = _eval_snapshot(
                model, eval_sampler, tuple(eval_lengths), eval_n, device, step, frontier
            )
            if snapshot is not None:
                history.append(snapshot)

    recent = [entry["rate"] for entry in acceptance_history[-acceptance_window:]]
    return {
        "final_loss": _final_loss(losses),
        "accept_rate": float(np.mean(recent)) if recent else 0.0,
        "overall_accept_rate": accepted_total / attempted_total if attempted_total else 0.0,
        "acceptance_history": acceptance_history,
        "history": history,
        "final_frontier": frontier,
        "steps": steps,
    }


def train_self_improve(
    model,
    task: Task,
    optimizer,
    *,
    steps: int,
    supervised_steps: int,
    rounds: int,
    initial_frontier: int,
    batch_size: int,
    block_size: int,
    unlabeled_sampler: LengthSampler,
    pseudo_per_round: int = 512,
    pseudo_batch_size: int = 128,
    n_votes: int = 5,
    vote_min: int = 3,
    lambda_u: float = 1.0,
    answer_length_bounds: AnswerLengthBounds | None = None,
    terminal_token_id: int | None = None,
    label_batch_size: int = 128,
    device: str = "cpu",
    seed: int = 0,
    supervised_sampler: BatchSampler | None = None,
    scheduler=None,
    grad_clip: float = 1.0,
    eval_sampler: LengthSampler | None = None,
    eval_lengths: Iterable[int] = (),
    eval_every: int = 0,
    eval_n: int = 100,
) -> dict:
    """Warm up, then self-train from frozen-teacher voted labels per frontier."""
    if steps < 1 or rounds < 1 or not 0 <= supervised_steps <= steps:
        raise ValueError("require steps >= 1, rounds >= 1, and 0 <= supervised_steps <= steps")
    supervised_sampler = supervised_sampler or task.sample_train
    rng = np.random.default_rng(seed)
    remainder = steps - supervised_steps
    quotient, extra = divmod(remainder, rounds)
    budgets = [quotient + int(index < extra) for index in range(rounds)]
    pool, round_stats, history, losses = [], [], [], []
    frontier = initial_frontier
    step = 0

    def train_steps(count):
        nonlocal step
        model.train()
        for _ in range(count):
            supervised = _draw(supervised_sampler, batch_size, "supervised_sampler")
            loss = _loss_on_items(model, task, supervised, block_size, device)
            if pool:
                indices = rng.integers(len(pool), size=min(pseudo_batch_size, len(pool)))
                pseudo_items = [pool[int(index)] for index in indices]
                loss = loss + lambda_u * _loss_on_items(
                    model, task, pseudo_items, block_size, device, explicit_pose=True
                )
            _optimise(model, optimizer, loss, scheduler, grad_clip)
            losses.append(float(loss.detach()))
            step += 1
            if eval_every and (step % eval_every == 0 or step == steps):
                snapshot = _eval_snapshot(
                    model, eval_sampler, tuple(eval_lengths), eval_n, device, step, frontier
                )
                if snapshot is not None:
                    history.append(snapshot)

    train_steps(supervised_steps)
    for budget in budgets:
        frontier += 1
        teacher = copy.deepcopy(model).to(device)
        teacher.requires_grad_(False)
        candidates = list(unlabeled_sampler(frontier, pseudo_per_round))
        bounds = answer_length_bounds(frontier) if answer_length_bounds else None
        pseudo = pseudo_label_vote(
            teacher, candidates, device=device, n_votes=n_votes,
            vote_min=vote_min, length_bounds=bounds,
            terminal_token_id=terminal_token_id,
            batch_size=label_batch_size,
        )
        pool.extend(pseudo.items)
        round_stats.append(
            {"frontier": frontier, "kept": pseudo.accepted,
             "of": pseudo.attempted, "pool": len(pool)}
        )
        del teacher
        train_steps(budget)

    return {
        "final_loss": _final_loss(losses),
        "history": history,
        "round_stats": round_stats,
        "final_frontier": frontier,
        "pseudo_pool_size": len(pool),
        "steps": step,
    }
