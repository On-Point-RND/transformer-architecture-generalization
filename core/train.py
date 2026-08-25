import math
import os
import random
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import torch

from core import checkpoint
from core.config import run_paths, to_dict
from core.logs import RunLogger
from tasks import get_task
from models import get_model

ARCHITECTURE_FIELDS = ("name", "n_layer", "n_head", "n_embd", "block_size",
                       "bias", "vocab_size", "pos_encoding")
DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}


def resolve_device(hardware):
    if hardware.device == "cpu":
        return "cpu"
    if hardware.device != "gpu":
        raise ValueError(f"hardware.device must be 'gpu' or 'cpu', got {hardware.device!r}")
    visible = torch.cuda.device_count()
    if visible == 0:
        raise RuntimeError("hardware.device is 'gpu' but torch sees no CUDA device; "
                           "set hardware.device: cpu")
    if not 0 <= hardware.gpu < visible:
        raise ValueError(f"hardware.gpu={hardware.gpu}, but the visible cards are "
                         f"0..{visible - 1}")
    return f"cuda:{hardware.gpu}"


def get_lr(it, train):
    """Linear warmup, cosine decay to min_lr, then flat."""
    if it < train.warmup_iters:
        return train.learning_rate * (it + 1) / (train.warmup_iters + 1)
    if it > train.lr_decay_iters:
        return train.min_lr
    decay_ratio = (it - train.warmup_iters) / (train.lr_decay_iters - train.warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # 1 -> 0
    return train.min_lr + coeff * (train.learning_rate - train.min_lr)


def make_get_batch(task, val_items, config, device, val_rng):
    batch_size, block_size = config.train.batch_size, config.model.block_size
    on_cuda = "cuda" in device

    def get_batch(split):
        items = (task.sample_train(batch_size) if split == "train"
                 else val_rng.sample(val_items, min(batch_size, len(val_items))))
        x_np, y_np = task.collate(items, block_size)
        x, y = torch.from_numpy(x_np), torch.from_numpy(y_np)
        if not on_cuda:
            return x.to(device), y.to(device)
        # pinning lets the copies overlap with compute
        return (x.pin_memory().to(device, non_blocking=True),
                y.pin_memory().to(device, non_blocking=True))

    return get_batch


def _eval_batch(model, get_batch, ctx, split, task, want_scores):
    """(loss, scores) for one batch; scoring is the task's business, loss is not."""
    x, y = get_batch(split)
    with ctx:
        logits, loss = model(x, y)
    if not want_scores:
        return loss.item(), {}
    return loss.item(), task.metrics(logits.argmax(dim=-1).cpu().numpy(), y.cpu().numpy())


def average_scores(split, batches):
    keys = batches[0] if batches else {}
    return {f"{split}_{key}": sum(b[key] for b in batches) / len(batches) for key in keys}


@torch.no_grad()
def estimate_loss(model, get_batch, ctx, train, task):
    """Mean loss and mean of the task's metrics over eval_iters fresh batches."""
    out = {}
    model.eval()
    for split in ("train", "val"):
        drawn = [_eval_batch(model, get_batch, ctx, split, task, train.eval_accuracy)
                 for _ in range(train.eval_iters)]
        out[split] = sum(loss for loss, _ in drawn) / len(drawn)
        out.update(average_scores(split, [scores for _, scores in drawn]))
    model.train()
    return out


def accumulate_gradients(model, x, y, get_batch, scaler, ctx, steps):
    """Run `steps` micro-batches; returns the last loss and the prefetched batch."""
    for _ in range(steps):
        with ctx:
            _, loss = model(x, y)
            loss = loss / steps  # accumulate the mean, not the sum
        x, y = get_batch("train")  # prefetch while the GPU is busy
        scaler.scale(loss).backward()
    return loss, x, y


def optimizer_step(model, optimizer, scaler, grad_clip):
    """Clip, step, and return the pre-clip global grad norm (or None)."""
    grad_norm = None
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)  # free the gradients early
    return grad_norm


def format_eval(iter_num, losses):
    parts = [f"train loss {losses['train']:.4f}", f"val loss {losses['val']:.4f}"]
    parts += [f"{key} {value:.4f}" for key, value in losses.items()
              if key not in ("train", "val")]
    return f"step {iter_num}: " + ", ".join(parts)


def update_mfu(model, running_mfu, fwdbwd_per_iter, dt, settled):
    """EMA of model flops utilisation; ignored until the loop settles."""
    if not settled:
        return running_mfu
    mfu = model.estimate_mfu(fwdbwd_per_iter, dt)
    return mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu


def run_metadata(config, model):
    """Fields recorded on every eval line, in checkpoints and in summary.csv."""
    report = model.param_report()
    encoding = report.get("positional_encoding", config.model.name)
    params = config.task.params
    return {
        "positional_encoding": encoding,
        "combination": encoding if "+" in encoding else "",
        "model": config.model.name,
        "task": config.task.name,
        "task_variant": params.get("task", params.get("variant", "default")),
        "seed": config.train.seed,
        "train_distribution": to_dict(params),
        "number_of_parameters": sum(p.numel() for p in model.parameters()),
        "positional_parameters": report.get("positional_parameters", {}),
    }


def build_model(config, task, device, resumed, ckpt_dir):
    _, model_cls = get_model(config.model.name)
    config.model.vocab_size = task.vocab_size
    model = model_cls(config.model)
    if resumed is not None:
        checkpoint.check_architecture(ckpt_dir, resumed, config.model, ARCHITECTURE_FIELDS)
        model.load_state_dict(checkpoint.strip_compile_prefix(resumed["model"]))
    return model.to(device)


def load_resume_state(config, resumed, optimizer, scaler, task):
    """Optimizer/scaler/RNG/task position; returns (iter_num, best_val_loss)."""
    optimizer.load_state_dict(resumed["optimizer"])
    if resumed.get("scaler") is not None:
        scaler.load_state_dict(resumed["scaler"])
    if resumed.get("rng") is not None:
        checkpoint.restore_rng(resumed["rng"])
    # after generate_val, so the val set is identical and the train stream continues
    if resumed.get("task") is not None:
        task.load_state_dict(resumed["task"])
    return resumed["iter_num"], resumed["best_val_loss"]


def pick_resume(train, ckpt_dir, device):
    """None for a fresh run, otherwise the checkpoint to continue from."""
    if train.init == "scratch":
        return None
    if train.init == "auto" and not checkpoint.exists(ckpt_dir):
        return None
    if train.init not in ("resume", "auto"):
        raise ValueError(f"train.init must be scratch/resume/auto, got {train.init!r}")
    return checkpoint.load(ckpt_dir, checkpoint.LAST, device)


def save_checkpoints(config, ckpt_dir, model, optimizer, scaler, task, metadata,
                     iter_num, val_loss, best_val_loss):
    """Write last.pt, and best.pt when val loss improved. Returns the new best."""
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "iter_num": iter_num,
        "best_val_loss": min(val_loss, best_val_loss),
        "config": to_dict(config),
        "run_metadata": metadata,
        "rng": checkpoint.rng_state(),
        "task": task.state_dict(),
    }
    if iter_num == 0:  # nothing trained yet (and eval_only must not touch disk)
        return min(val_loss, best_val_loss)
    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
    if config.train.always_save_checkpoint:
        checkpoint.save(ckpt_dir, checkpoint.LAST, payload)
    if val_loss >= best_val_loss:
        return best_val_loss
    checkpoint.save(ckpt_dir, checkpoint.BEST, payload)
    return val_loss


def run(config):
    """Train one model. Returns the best val loss reached."""
    train_cfg, hardware = config.train, config.hardware
    paths = run_paths(config.paths)
    device = resolve_device(hardware)
    steps = train_cfg.gradient_accumulation_steps

    torch.manual_seed(train_cfg.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device_type = "cuda" if "cuda" in device else "cpu"
    ctx = (nullcontext() if device_type == "cpu" else
           torch.amp.autocast(device_type=device_type, dtype=DTYPES[hardware.dtype]))

    task = get_task(config.task.name, {**config.task.params, "seed": train_cfg.data_seed})
    val_items = task.generate_val(config.task.n_val)

    resumed = pick_resume(train_cfg, paths.checkpoints, device)
    model = build_model(config, task, device, resumed, paths.checkpoints)
    scaler = torch.amp.GradScaler(device_type, enabled=hardware.dtype == "float16")
    optimizer = model.configure_optimizers(train_cfg.weight_decay, train_cfg.learning_rate,
                                           (train_cfg.beta1, train_cfg.beta2), device_type)
    iter_num, best_val_loss = 0, float("inf")
    if resumed is not None:
        iter_num, best_val_loss = load_resume_state(config, resumed, optimizer, scaler, task)

    metadata = run_metadata(config, model)
    raw_model = model
    if hardware.compile:
        model = torch.compile(model)

    val_rng = random.Random(train_cfg.seed) if train_cfg.reproducible_val else random
    get_batch = make_get_batch(task, val_items, config, device, val_rng)
    logger = RunLogger(paths, metadata, resume=resumed is not None)
    tokens_per_iter = steps * train_cfg.batch_size * config.model.block_size
    print(f"task '{config.task.name}' vocab_size = {task.vocab_size}, "
          f"held-out val size = {len(val_items)}")
    print(f"positional encoding: {metadata['positional_encoding']}, "
          f"parameters: {metadata['number_of_parameters'] / 1e6:.2f}M, "
          f"positional: {metadata['positional_parameters']}")
    print(f"tokens per iteration will be: {tokens_per_iter:,}")

    x, y = get_batch("train")  # the very first batch
    t0 = time.time()
    local_iter_num, running_mfu, evals = 0, -1.0, 0

    while True:
        lr = get_lr(iter_num, train_cfg) if train_cfg.decay_lr else train_cfg.learning_rate
        for group in optimizer.param_groups:
            group["lr"] = lr

        if iter_num % train_cfg.eval_interval == 0:
            losses = estimate_loss(model, get_batch, ctx, train_cfg, task)
            scores = {k: v for k, v in losses.items() if k not in ("train", "val")}
            print(format_eval(iter_num, losses))
            logger.log_eval(iter=iter_num, lr=lr, mfu=running_mfu,
                            train_loss=losses["train"], val_loss=losses["val"], **scores)
            best_val_loss = save_checkpoints(config, paths.checkpoints, raw_model, optimizer,
                                             scaler, task, metadata, iter_num,
                                             losses["val"], best_val_loss)
            logger.write_summary(iter=iter_num, train_loss=losses["train"],
                                 val_loss=losses["val"], best_val_loss=best_val_loss,
                                 tokens=tokens_per_iter * iter_num, **scores)
            logger.write_curves()
            evals += 1
            logger.log_diagnostics(raw_model, iter_num, train_cfg.diag_interval, evals)

        if iter_num == 0 and train_cfg.eval_only:
            break

        loss, x, y = accumulate_gradients(model, x, y, get_batch, scaler, ctx, steps)
        grad_norm = optimizer_step(model, optimizer, scaler, train_cfg.grad_clip)

        dt = time.time() - t0
        t0 = time.time()
        if iter_num % train_cfg.log_interval == 0:
            # loss.item() syncs; scale back up to approximate the full-batch loss
            lossf = loss.item() * steps
            running_mfu = update_mfu(raw_model, running_mfu, train_cfg.batch_size * steps,
                                     dt, settled=local_iter_num >= 5)
            print(f"iter {iter_num}: loss {lossf:.4f}, time {dt * 1000:.2f}ms, "
                  f"mfu {running_mfu * 100:.2f}%")
            logger.log("train", iter=iter_num, loss=lossf, lr=lr, dt=dt, mfu=running_mfu,
                       grad_norm=None if grad_norm is None else float(grad_norm))

        iter_num += 1
        local_iter_num += 1
        # NOTE(frozen): `>` runs one iteration past max_iters, as the original did
        if iter_num > train_cfg.max_iters:
            break

    return best_val_loss

