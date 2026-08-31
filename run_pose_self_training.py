#!/usr/bin/env python3
"""Run the PoSE/FixMatch/self-improvement length-generalization study.

This is the current-repository counterpart of ``run_pose_fixmatch.py`` from
``transformer-all-tasks``.  It deliberately keeps the old variant names and
JSON/checkpoint filenames while using the current config, model, and task APIs.

Examples::

    python run_pose_self_training.py --variant baseline
    python run_pose_self_training.py --variant abs_shift
    python run_pose_self_training.py --variant pose
    python run_pose_self_training.py --variant pose_fixmatch_curr
    python run_pose_self_training.py --variant pose_selfimprove --train-max-len 10
    python run_pose_self_training.py --variant baseline --set hardware.device=cpu
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from statistics import fmean

import torch

from core.config import load_config, to_dict
from core.train import resolve_device
from models import get_model
from tasks import get_task


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "configs" / "pose_self_training.yaml"
VARIANTS = (
    "baseline",
    "abs_shift",
    "pose",
    "pose_fixmatch",
    "pose_fixmatch_curr",
    "pose_selfimprove",
)

LENGTH_PARAMETERS = {
    "addition": "n_digits",
    "add_seq": "n_digits",
    "add_indep": "n_digits",
    "sorting": "n_items",
    "indexing": "n_items",
    "dyck": "prefix_len",
}

# Keep the previous experiment's filenames even though the current registry
# calls its BOS/EOS sequential-addition preset ``add_seq``.
ARTIFACT_TASK_NAMES = {"add_seq": "addition"}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument(
        "--config",
        action="append",
        metavar="FILE",
        help=f"standard config file; repeatable, later files win (default: {DEFAULT_CONFIG})",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        dest="overrides",
        metavar="section.key=value",
        help="standard config override",
    )
    parser.add_argument(
        "--task",
        help="task registry name; provide compatible task.params via --config/--set",
    )
    parser.add_argument(
        "--length-param",
        help="task parameter controlled by train/OOD length (inferred for built-in tasks)",
    )
    parser.add_argument("--train-min-len", type=int, help="default: config task lower bound")
    parser.add_argument("--train-max-len", type=int, help="default: config task upper bound")
    parser.add_argument("--ood-max-len", type=int, default=15)
    parser.add_argument("--steps", type=int, help="default: config train.max_iters")
    parser.add_argument(
        "--sup-steps",
        type=int,
        help="self-improve supervised warmup (default: half of total steps)",
    )
    parser.add_argument("--batch-size", type=int, help="default: config train.batch_size")
    parser.add_argument("--unlabeled-batch-size", type=int, default=128)
    parser.add_argument("--unlabeled-every", type=int, default=1)
    parser.add_argument("--pseudo-per-round", type=int, default=512)
    parser.add_argument("--n-votes", type=int, default=5)
    parser.add_argument("--vote-min", type=int, default=3)
    parser.add_argument("--tau", type=float, default=0.95)
    parser.add_argument("--lambda-u", type=float, default=1.0)
    parser.add_argument("--advance-threshold", type=float, default=0.7)
    parser.add_argument("--lr", type=float, help="default: config optimizer.learning_rate")
    parser.add_argument("--train-seed", type=int, help="default: config train.seed")
    parser.add_argument("--eval-seed", type=int, default=0)
    parser.add_argument("--n-eval", type=int, default=300)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument(
        "--out",
        type=Path,
        help="result JSON path (default: <config paths.run_dir>/<task>_<variant>.json)",
    )
    return parser.parse_args(argv)


def _configured_length_range(params, name):
    if name not in params:
        raise ValueError(
            f"task.params has no {name!r}; add it to the config or pass --length-param"
        )
    value = params[name]
    if isinstance(value, int):
        return value, value
    if isinstance(value, tuple) and len(value) == 2:
        low, exclusive_high = map(int, value)
        return low, exclusive_high - 1
    raise ValueError(
        f"task.params.{name} must be an int or a two-item half-open range, got {value!r}"
    )


def _resolve_length_parameter(task_name, params, requested):
    if requested:
        return requested
    if task_name in LENGTH_PARAMETERS:
        return LENGTH_PARAMETERS[task_name]
    candidates = [name for name in set(LENGTH_PARAMETERS.values()) if name in params]
    if len(candidates) == 1:
        return candidates[0]
    raise ValueError(
        f"cannot infer the length parameter for task {task_name!r}; pass --length-param"
    )


def _fixed_task(task_name, params, length_param, length, seed):
    fixed = {**params, length_param: int(length), "seed": int(seed)}
    return get_task(task_name, fixed)


def _validate_task_shape(task, expected_vocab_size, block_size, length):
    if task.vocab_size != expected_vocab_size:
        raise ValueError(
            f"task vocabulary changes at length {length}: trained model has "
            f"{expected_vocab_size} tokens but the evaluation task has {task.vocab_size}; "
            "set a fixed vocabulary-capacity parameter (for example indexing.max_items)"
        )
    minimum = getattr(task, "min_block_size", None)
    if minimum is not None and minimum > block_size:
        raise ValueError(
            f"length {length} needs block_size >= {minimum}, got {block_size}"
        )


def _build_samplers(
    task_name,
    params,
    length_param,
    *,
    train_seed,
    eval_seed,
    vocab_size,
    block_size,
):
    unlabeled_tasks = {}

    def unlabeled_sampler(length, n):
        if length not in unlabeled_tasks:
            task = _fixed_task(
                task_name,
                params,
                length_param,
                length,
                train_seed + 100_000 + int(length),
            )
            _validate_task_shape(task, vocab_size, block_size, length)
            unlabeled_tasks[length] = task
        return unlabeled_tasks[length].sample(n)

    def eval_sampler(length, n):
        # Recreate the task on every call: mid-training snapshots and the final
        # sweep then see the same examples instead of advancing an eval RNG.
        task = _fixed_task(task_name, params, length_param, length, eval_seed)
        _validate_task_shape(task, vocab_size, block_size, length)
        return task.sample(n)

    return unlabeled_sampler, eval_sampler


def _answer_length_bounds(task):
    """Current fixed-width addition answer size, including its optional EOS."""
    if type(task).__name__ != "AdditionTask":
        return None
    extra = 2 if task.bos_eos else 1  # carry digit, plus EOS when enabled
    return lambda length: (int(length) + extra, int(length) + extra)


def _build_model_and_optimizer(config, task, device, steps, learning_rate):
    if config.hardware.compile:
        raise ValueError("this runner does not support hardware.compile; set it to false")
    if config.train.gradient_accumulation_steps != 1:
        raise ValueError(
            "this runner currently requires train.gradient_accumulation_steps=1"
        )
    if config.hardware.dtype != "float32":
        raise ValueError(
            "this runner reproduces the original float32 recipe; "
            "set hardware.dtype=float32"
        )

    _, model_cls = get_model(config.model.name)
    config.model.vocab_size = task.vocab_size
    model = model_cls(config.model).to(device)
    config.optimizer.learning_rate = learning_rate
    device_type = "cuda" if "cuda" in device else "cpu"
    optimizer = model.configure_optimizers(config.optimizer, device_type)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=learning_rate,
        total_steps=steps,
        pct_start=0.05,
        anneal_strategy="cos",
        cycle_momentum=False,
    )
    return model, optimizer, scheduler


def _normalise_training_info(value):
    if isinstance(value, dict):
        if "final_loss" not in value:
            raise ValueError("training result has no 'final_loss'")
        return value
    return {"final_loss": float(value)}


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _load_run_config(args):
    position = {
        "baseline": "wpe",
        "abs_shift": "abs_shift",
    }.get(args.variant, "pose")
    overrides = list(args.overrides)
    if args.task:
        overrides.append(f"task.name={args.task}")
    # The variant owns this field; append last so a generic --set cannot make a
    # result labelled 'pose' silently train another positional mechanism.
    overrides.append(f"model.pos_encoding={position}")
    return load_config(args.config or [DEFAULT_CONFIG], overrides)


def _artifact_path(args, config, task_name, train_max):
    if args.out:
        return args.out
    label = ARTIFACT_TASK_NAMES.get(task_name, task_name)
    suffix = f"_L{train_max}" if train_max != 5 else ""
    return Path(config.paths.run_dir) / f"{label}_{args.variant}{suffix}.json"


def run(args):
    from experiments.self_training import (
        evaluate_lengths,
        train_fixmatch,
        train_self_improve,
        train_supervised,
    )

    config = _load_run_config(args)
    task_name = config.task.name
    base_params = dict(config.task.params)
    length_param = _resolve_length_parameter(task_name, base_params, args.length_param)
    configured_min, configured_max = _configured_length_range(base_params, length_param)
    train_min = configured_min if args.train_min_len is None else args.train_min_len
    train_max = configured_max if args.train_max_len is None else args.train_max_len
    if not 1 <= train_min <= train_max:
        raise ValueError(
            f"expected 1 <= train_min_len <= train_max_len, got {train_min}..{train_max}"
        )
    if args.ood_max_len <= train_max:
        raise ValueError("--ood-max-len must be greater than the training maximum")

    steps = config.train.max_iters if args.steps is None else args.steps
    batch_size = config.train.batch_size if args.batch_size is None else args.batch_size
    learning_rate = (
        config.optimizer.learning_rate if args.lr is None else args.lr
    )
    train_seed = config.train.seed if args.train_seed is None else args.train_seed
    if steps < 1 or batch_size < 1:
        raise ValueError("steps and batch size must be positive")
    config.train.max_iters = steps
    config.train.batch_size = batch_size
    config.train.seed = train_seed
    config.train.data_seed = train_seed
    config.task.params = {
        **base_params,
        length_param: (int(train_min), int(train_max) + 1),
    }

    train_params = {
        **config.task.params,
        "seed": int(train_seed),
    }
    task = get_task(task_name, train_params)
    task.generate_val(config.task.n_val)
    if config.task.n_train:
        task.build_train_pool(config.task.n_train)

    device = resolve_device(config.hardware)
    torch.manual_seed(train_seed)
    model, optimizer, scheduler = _build_model_and_optimizer(
        config, task, device, steps, learning_rate
    )
    block_size = config.model.block_size
    _validate_task_shape(task, task.vocab_size, block_size, train_max)
    unlabeled_sampler, eval_sampler = _build_samplers(
        task_name,
        base_params,
        length_param,
        train_seed=train_seed,
        eval_seed=args.eval_seed,
        vocab_size=task.vocab_size,
        block_size=block_size,
    )

    ood_min = train_max + 1
    ood_max = args.ood_max_len
    snapshot_lengths = tuple(range(ood_min, min(ood_min + 5, ood_max + 1)))
    bounds = _answer_length_bounds(task)
    terminal_token_id = getattr(task, "EOS_ID", None) if getattr(task, "bos_eos", False) else None
    common = dict(
        steps=steps,
        batch_size=batch_size,
        block_size=block_size,
        device=device,
        scheduler=scheduler,
        grad_clip=config.optimizer.grad_clip,
    )

    print(
        f"== {args.variant} on {task_name}: "
        f"{sum(parameter.numel() for parameter in model.parameters()):,} params, "
        f"device={device}"
    )
    started = time.time()
    if args.variant in ("baseline", "abs_shift", "pose"):
        info = _normalise_training_info(
            train_supervised(model, task, optimizer, **common)
        )
    elif args.variant in ("pose_fixmatch", "pose_fixmatch_curr"):
        curriculum = args.variant == "pose_fixmatch_curr"
        if bounds is None:
            raise ValueError(
                "FixMatch needs a task-specific answer-length rule without reading "
                "hidden target lengths; the current runner defines one for AdditionTask"
            )
        info = _normalise_training_info(
            train_fixmatch(
                model,
                task,
                optimizer,
                unlabeled_sampler=unlabeled_sampler,
                unlabeled_batch_size=args.unlabeled_batch_size,
                unlabeled_min_length=ood_min,
                unlabeled_max_length=ood_max,
                unlabeled_every=args.unlabeled_every,
                confidence_threshold=args.tau,
                lambda_u=args.lambda_u,
                answer_length_bounds=bounds,
                terminal_token_id=terminal_token_id,
                curriculum=curriculum,
                advance_threshold=args.advance_threshold,
                acceptance_window=max(
                    1, config.train.log_interval // args.unlabeled_every
                ),
                eval_sampler=eval_sampler,
                eval_lengths=snapshot_lengths,
                eval_every=args.eval_every,
                eval_n=min(100, args.n_eval),
                seed=train_seed,
                **common,
            )
        )
    else:
        if bounds is None:
            raise ValueError(
                "self-improvement needs a task-specific answer-length rule without "
                "reading hidden target lengths; the current runner defines one for "
                "AdditionTask"
            )
        rounds = min(5, ood_max - train_max)
        supervised_steps = steps // 2 if args.sup_steps is None else args.sup_steps
        info = _normalise_training_info(
            train_self_improve(
                model,
                task,
                optimizer,
                unlabeled_sampler=unlabeled_sampler,
                supervised_steps=supervised_steps,
                rounds=rounds,
                initial_frontier=train_max,
                pseudo_per_round=args.pseudo_per_round,
                pseudo_batch_size=args.unlabeled_batch_size,
                n_votes=args.n_votes,
                vote_min=args.vote_min,
                lambda_u=args.lambda_u,
                answer_length_bounds=bounds,
                terminal_token_id=terminal_token_id,
                label_batch_size=args.unlabeled_batch_size,
                eval_sampler=eval_sampler,
                eval_lengths=snapshot_lengths,
                eval_every=args.eval_every,
                eval_n=min(100, args.n_eval),
                seed=train_seed,
                **common,
            )
        )
    train_seconds = time.time() - started

    id_lengths = tuple(range(train_min, train_max + 1))
    ood_lengths = tuple(range(ood_min, ood_max + 1))
    id_accuracy = evaluate_lengths(
        model, eval_sampler, id_lengths, n=args.n_eval, device=device
    )
    ood_accuracy = evaluate_lengths(
        model, eval_sampler, ood_lengths, n=args.n_eval, device=device
    )

    artifact_task = ARTIFACT_TASK_NAMES.get(task_name, task_name)
    result = {
        "variant": args.variant,
        "task": artifact_task,
        "config": {
            "resolved": to_dict(config),
            "runner": vars(args),
            "task_registry_name": task_name,
            "length_parameter": length_param,
            "train_min_len": train_min,
            "train_max_len": train_max,
            "ood_max_len": ood_max,
        },
        "device": device,
        "final_train_loss": info["final_loss"],
        "final_accept_rate": info.get("accept_rate"),
        "final_frontier": info.get("final_frontier"),
        "training_history": info.get("history"),
        "round_stats": info.get("round_stats"),
        "train_seconds": round(train_seconds, 1),
        "in_domain": {str(length): value for length, value in id_accuracy.items()},
        "in_domain_mean": float(fmean(id_accuracy.values())),
        "ood": {str(length): value for length, value in ood_accuracy.items()},
        "ood_mean": float(fmean(ood_accuracy.values())),
    }

    output = _artifact_path(args, config, task_name, train_max)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, default=_json_default) + "\n", encoding="utf-8"
    )
    checkpoint = output.with_suffix(".pth")
    torch.save(model.state_dict(), checkpoint)
    print(
        f"== {args.variant}: in-domain mean={result['in_domain_mean']:.3f} "
        f"OOD mean={result['ood_mean']:.3f} ({train_seconds:.0f}s train)"
    )
    print(f"Saved results to {output}")
    print(f"Checkpoint saved to {checkpoint}")
    return result


def main(argv=None):
    run(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
