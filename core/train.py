import math
import json
import random
import sys
import time
from contextlib import nullcontext
from dataclasses import fields
from pathlib import Path

import torch
import yaml

from core import checkpoint
from core.config import run_paths, to_dict
from core.logs import RunLogger
from core.provenance import git_info, write_git_artifacts
from tasks import get_task
from models import get_model

DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}


def architecture_fields(model_config):
    return tuple(f.name for f in fields(model_config) if f.name != "dropout")


def resolve_device(hardware):
    if hardware.device == "cpu":
        return "cpu"
    if hardware.device == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("hardware.device is 'mps' but PyTorch MPS is unavailable; "
                               "set hardware.device: cpu")
        return "mps"
    if hardware.device != "gpu":
        raise ValueError(f"hardware.device must be 'gpu', 'mps', or 'cpu', "
                         f"got {hardware.device!r}")
    visible = torch.cuda.device_count()
    if visible == 0:
        raise RuntimeError("hardware.device is 'gpu' but torch sees no CUDA device; "
                           "set hardware.device: cpu")
    if not 0 <= hardware.gpu < visible:
        raise ValueError(f"hardware.gpu={hardware.gpu}, but the visible cards are "
                         f"0..{visible - 1}")
    return f"cuda:{hardware.gpu}"


def get_lr(it, opt):
    if opt.schedule == "constant":
        return opt.learning_rate
    if it < opt.warmup_iters:
        return opt.learning_rate * (it + 1) / (opt.warmup_iters + 1)
    if it > opt.lr_decay_iters:
        return opt.min_lr
    decay_ratio = (it - opt.warmup_iters) / (opt.lr_decay_iters - opt.warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  
    return opt.min_lr + coeff * (opt.learning_rate - opt.min_lr)


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
        return (x.pin_memory().to(device, non_blocking=True),
                y.pin_memory().to(device, non_blocking=True))

    return get_batch


def _eval_batch(model, get_batch, ctx, split, task, want_scores):
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
    for _ in range(steps):
        with ctx:
            _, loss = model(x, y)
            loss = loss / steps  
        x, y = get_batch("train")  
        scaler.scale(loss).backward()
    return loss, x, y


def optimizer_step(model, optimizer, scaler, grad_clip):
    grad_norm = None
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)  
    return grad_norm


def watch_early_stop(train, losses, best, stale):
    if not train.early_stop_patience and train.early_stop_target is None:
        return False, best, stale
    value = losses.get(train.early_stop_metric)
    if value is None:
        raise ValueError(f"train.early_stop_metric={train.early_stop_metric!r} is not "
                         f"among the eval metrics {sorted(losses)}")
    lower_is_better = train.early_stop_metric in ("train", "val")
    target = train.early_stop_target
    if target is not None:
        reached = value <= target if lower_is_better else value >= target
        if reached:
            return True, value, 0
    if best is None:
        return False, value, 0
    delta = train.early_stop_min_delta
    improved = value < best - delta if lower_is_better else value > best + delta
    if improved:
        return False, value, 0
    stale += 1
    stop = train.early_stop_patience > 0 and stale >= train.early_stop_patience
    return stop, best, stale


def format_eval(iter_num, losses):
    """Eval line for the console. Colours only when stdout is a TTY."""
    paint = ((lambda text, code: f"\033[{code}m{text}\033[0m") if sys.stdout.isatty()
             else (lambda text, code: text))
    parts = [paint(f"train loss {losses['train']:.4f}", "1;31")] 
    if "train_acc" in losses:
        parts.append(paint(f"train_acc {losses['train_acc']:.4f}", "1;34"))  
    parts.append(f"val loss {losses['val']:.4f}")

    val_acc_key = next((k for k in ("val_acc", "val_token_acc") if k in losses), None)
    if val_acc_key is not None:
        parts.append(paint(f"{val_acc_key} {losses[val_acc_key]:.4f}", "1;38;5;28")) 
    shown = {"train", "val", "train_acc", val_acc_key}
    parts += [f"{key} {value:.4f}" for key, value in losses.items() if key not in shown]
    return f"step {iter_num}: " + ", ".join(parts)


def update_mfu(model, running_mfu, fwdbwd_per_iter, dt, settled):
    """EMA of model flops utilisation; ignored until the loop settles."""
    if not settled:
        return running_mfu
    mfu = model.estimate_mfu(fwdbwd_per_iter, dt)
    return mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu


def run_metadata(config, model):
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
        "data_seed": config.train.data_seed,
        "training_stage": config.train.stage,
        "regime": config.train.regime,
        "source_run_id": config.train.source_run_id,
        "source_checkpoint": config.train.source_checkpoint,
        "source_positional_encoding": config.train.source_positional_encoding,
        "source_global_step": config.train.source_global_step,
        "train_distribution": to_dict(params),
        "number_of_parameters": sum(p.numel() for p in model.parameters()),
        "number_of_layers": config.model.n_layer,
        "number_of_heads": config.model.n_head,
        "embedding_dimension": config.model.n_embd,
        "block_size": config.model.block_size,
        "optimizer": config.optimizer.name,
        "learning_rate": config.optimizer.learning_rate,
        "scheduler": config.optimizer.schedule,
        "warmup": config.optimizer.warmup_iters,
        "weight_decay": config.optimizer.weight_decay,
        "batch_size": config.train.batch_size,
        "gradient_accumulation": config.train.gradient_accumulation_steps,
        "max_iterations": config.train.max_iters,
        "positional_hyperparameters": {
            name: getattr(config.model, name)
            for name in checkpoint.POSITIONAL_FIELDS if hasattr(config.model, name)
        },
        "positional_parameters": report.get("positional_parameters", {}),
        **git_info(),
    }


def build_model(config, task, device, resumed, ckpt_dir, transfer=False):
    _, model_builder = get_model(config.model.name)
    config.model.vocab_size = task.vocab_size
    model = model_builder(config.model)
    if resumed is not None:
        if transfer:
            checkpoint.check_transfer_architecture(
                ckpt_dir, resumed, config.model, architecture_fields(config.model)
            )
            transfer_report = checkpoint.load_transfer_model_state(model, resumed)
        else:
            checkpoint.check_architecture(ckpt_dir, resumed, config.model,
                                          architecture_fields(config.model))
            model.load_state_dict(checkpoint.strip_compile_prefix(resumed["model"]))
            transfer_report = None
    else:
        transfer_report = None
    model.checkpoint_transfer_report = transfer_report
    return model.to(device)


def load_resume_state(config, resumed, optimizer, scaler, task, model, transfer=False):
    if transfer:
        named = resumed.get("optimizer_named")
        if named is not None:
            optimizer_report = checkpoint.restore_named_optimizer_state(model, optimizer, named)
        else:
            try:
                optimizer.load_state_dict(resumed["optimizer"])
                optimizer_report = {"restored": "legacy-full-state", "skipped": []}
            except (ValueError, KeyError):
                optimizer_report = {"restored": [], "skipped": "legacy-topology-mismatch"}
    else:
        optimizer.load_state_dict(resumed["optimizer"])
        optimizer_report = {"restored": "full-state", "skipped": []}
    if resumed.get("scaler") is not None:
        scaler.load_state_dict(resumed["scaler"])
    if resumed.get("rng") is not None:
        checkpoint.restore_rng(resumed["rng"])
    if resumed.get("task") is not None:
        task.load_state_dict(resumed["task"])
    return resumed["iter_num"], resumed["best_val_loss"], optimizer_report


def pick_resume(train, ckpt_dir, device):
    if train.init == "scratch":
        return None, False, ""
    if train.init == "checkpoint":
        if not train.checkpoint_path:
            raise ValueError("train.init=checkpoint requires train.checkpoint_path")
        source = Path(train.checkpoint_path)
        if not source.is_file():
            raise FileNotFoundError(f"continuation checkpoint does not exist: {source}")
        return checkpoint.load_path(source, device), True, str(source)
    if train.init == "auto" and not checkpoint.exists(ckpt_dir):
        return None, False, ""
    if train.init not in ("resume", "auto"):
        raise ValueError(
            f"train.init must be scratch/resume/auto/checkpoint, got {train.init!r}"
        )
    return checkpoint.load(ckpt_dir, checkpoint.LAST, device), False, str(
        Path(ckpt_dir) / checkpoint.LAST
    )


def save_checkpoints(config, ckpt_dir, model, optimizer, scaler, task, metadata,
                     iter_num, val_loss, best_val_loss):
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "optimizer_named": checkpoint.named_optimizer_state(model, optimizer),
        "scaler": scaler.state_dict(),
        "iter_num": iter_num,
        "best_val_loss": min(val_loss, best_val_loss),
        "config": to_dict(config),
        "run_metadata": metadata,
        "rng": checkpoint.rng_state(),
        "task": task.state_dict(),
    }
    if iter_num == 0:  
        return min(val_loss, best_val_loss)
    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
    if config.train.always_save_checkpoint:
        checkpoint.save(ckpt_dir, checkpoint.LAST, payload)
    if val_loss >= best_val_loss:
        return best_val_loss
    checkpoint.save(ckpt_dir, checkpoint.BEST, payload)
    return val_loss


def run(config):    
    train_cfg, opt_cfg, hardware = config.train, config.optimizer, config.hardware
    paths = run_paths(config.paths)
    device = resolve_device(hardware)
    steps = train_cfg.gradient_accumulation_steps

    torch.manual_seed(train_cfg.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device_type = torch.device(device).type
    ctx = (nullcontext() if device_type == "cpu" or hardware.dtype == "float32" else
           torch.amp.autocast(device_type=device_type, dtype=DTYPES[hardware.dtype]))

    task = get_task(config.task.name, {**config.task.params, "seed": train_cfg.data_seed})
    val_items = task.generate_val(config.task.n_val)
    if config.task.n_train:
        task.build_train_pool(config.task.n_train)

    resumed, transfer, source_path = pick_resume(train_cfg, paths.checkpoints, device)
    if transfer:
        source_sections = checkpoint.config_sections(resumed)
        current_task = to_dict(config.task)
        source_task = source_sections["task"]
        task_differences = {
            key: (source_task.get(key), current_task.get(key))
            for key in ("name", "params", "n_val", "n_train")
            if source_task.get(key) != current_task.get(key)
        }
        source_data_seed = source_sections["train"].get("data_seed")
        if source_data_seed != train_cfg.data_seed:
            task_differences["data_seed"] = (source_data_seed, train_cfg.data_seed)
        if task_differences:
            detail = ", ".join(
                f"{key}: checkpoint {old!r} vs config {new!r}"
                for key, (old, new) in sorted(task_differences.items())
            )
            raise ValueError(
                "continuation requires the same task distribution and validation set "
                f"({detail})"
            )
        source_metadata = resumed.get("run_metadata", {})
        source_model = checkpoint.model_fields(resumed)
        actual_source_encoding = (
            source_metadata.get("positional_encoding")
            or source_model.get("pos_encoding", "")
        )
        if (train_cfg.source_positional_encoding
                and train_cfg.source_positional_encoding != actual_source_encoding):
            raise ValueError(
                "train.source_positional_encoding does not match checkpoint: "
                f"declared {train_cfg.source_positional_encoding!r}, "
                f"checkpoint {actual_source_encoding!r}"
            )
        actual_source_step = resumed.get("iter_num")
        if (train_cfg.source_global_step is not None
                and train_cfg.source_global_step != actual_source_step):
            raise ValueError(
                "train.source_global_step does not match checkpoint: "
                f"declared {train_cfg.source_global_step}, checkpoint {actual_source_step}"
            )
        train_cfg.source_checkpoint = train_cfg.source_checkpoint or source_path
        train_cfg.source_positional_encoding = (
            train_cfg.source_positional_encoding
            or actual_source_encoding
        )
        train_cfg.source_global_step = (
            train_cfg.source_global_step
            if train_cfg.source_global_step is not None else actual_source_step
        )
        if not train_cfg.source_run_id:
            run_id_file = Path(source_path).parent / "mlflow-run-id"
            if run_id_file.is_file():
                train_cfg.source_run_id = run_id_file.read_text(encoding="utf-8").strip()
    model = build_model(
        config, task, device, resumed, source_path or paths.checkpoints, transfer=transfer
    )
    transfer_report = model.checkpoint_transfer_report
    scaler = torch.amp.GradScaler(
        device_type if device_type in ("cpu", "cuda") else "cpu",
        enabled=device_type == "cuda" and hardware.dtype == "float16",
    )
    optimizer = model.configure_optimizers(opt_cfg, device_type)
    iter_num, best_val_loss = 0, float("inf")
    optimizer_report = None
    if resumed is not None:
        iter_num, best_val_loss, optimizer_report = load_resume_state(
            config, resumed, optimizer, scaler, task, model, transfer=transfer
        )
        if transfer:
            best_val_loss = float("inf")

    metadata = run_metadata(config, model)
    def report_counts(report):
        if report is None:
            return None
        return {key: len(value) if isinstance(value, list) else value
                for key, value in report.items()}

    metadata["checkpoint_transfer"] = report_counts(transfer_report)
    metadata["optimizer_transfer"] = report_counts(optimizer_report)
    paths.logs.mkdir(parents=True, exist_ok=True)
    (paths.logs / "config.resolved.yaml").write_text(
        yaml.safe_dump(to_dict(config), sort_keys=False), encoding="utf-8"
    )
    (paths.logs / "checkpoint-metadata.json").write_text(
        json.dumps({**metadata, "checkpoint_transfer_details": transfer_report,
                    "optimizer_transfer_details": optimizer_report},
                   indent=2, default=str) + "\n", encoding="utf-8"
    )
    write_git_artifacts(paths.logs)
    raw_model = model
    if hardware.compile:
        model = torch.compile(model)

    val_rng = random.Random(train_cfg.seed) if train_cfg.reproducible_val else random
    get_batch = make_get_batch(task, val_items, config, device, val_rng)
    logger = RunLogger(paths, metadata, config, resume=resumed is not None and not transfer)
    tokens_per_iter = steps * train_cfg.batch_size * config.model.block_size
    print(f"task '{config.task.name}' vocab_size = {task.vocab_size}, "
          f"held-out val size = {len(val_items)}")
    print(f"positional encoding: {metadata['positional_encoding']}, "
          f"parameters: {metadata['number_of_parameters'] / 1e6:.2f}M, "
          f"positional: {metadata['positional_parameters']}")
    print(f"tokens per iteration will be: {tokens_per_iter:,}")

    try:
        x, y = get_batch("train")
        started_at = t0 = time.time()
        local_iter_num, running_mfu, evals = 0, -1.0, 0
        best_watched, stale = None, 0

        while True:
            lr = get_lr(iter_num, opt_cfg)
            for group in optimizer.param_groups:
                group["lr"] = lr

            if iter_num % train_cfg.eval_interval == 0:
                eval_started = time.time()
                losses = estimate_loss(model, get_batch, ctx, train_cfg, task)
                eval_time = time.time() - eval_started
                scores = {k: v for k, v in losses.items() if k not in ("train", "val")}
                print(format_eval(iter_num, losses))
                logger.log_eval(iter=iter_num, lr=lr, mfu=running_mfu,
                                train_loss=losses["train"], val_loss=losses["val"],
                                id_accuracy=losses.get("val_acc"),
                                id_token_accuracy=losses.get("val_token_acc"),
                                evaluation_time_seconds=eval_time, **scores)
                best_val_loss = save_checkpoints(
                    config, paths.checkpoints, raw_model, optimizer, scaler, task, metadata,
                    iter_num, losses["val"], best_val_loss
                )
                logger.write_summary(iter=iter_num, train_loss=losses["train"],
                                     val_loss=losses["val"], best_val_loss=best_val_loss,
                                     tokens=tokens_per_iter * iter_num,
                                     training_time_seconds=time.time() - started_at,
                                     evaluation_time_seconds=eval_time, **scores)
                logger.write_curves()
                evals += 1
                logger.log_diagnostics(raw_model, iter_num, train_cfg.diag_interval, evals)
                stop, best_watched, stale = watch_early_stop(
                    train_cfg, losses, best_watched, stale
                )
                if stop:
                    reason = "target reached" if stale == 0 else f"no gain for {stale} eval(s)"
                    print(f"early stop at {iter_num}: {train_cfg.early_stop_metric} "
                          f"{losses[train_cfg.early_stop_metric]:.4f} - {reason}")
                    break

            if train_cfg.eval_only:
                break
            if iter_num >= train_cfg.max_iters:
                break

            loss, x, y = accumulate_gradients(model, x, y, get_batch, scaler, ctx, steps)
            grad_norm = optimizer_step(model, optimizer, scaler, opt_cfg.grad_clip)

            dt = time.time() - t0
            t0 = time.time()
            if iter_num % train_cfg.log_interval == 0:
                lossf = loss.item() * steps
                running_mfu = update_mfu(raw_model, running_mfu, train_cfg.batch_size * steps,
                                         dt, settled=local_iter_num >= 5)
                print(f"iter {iter_num}: loss {lossf:.4f}, time {dt * 1000:.2f}ms, "
                      f"mfu {running_mfu * 100:.2f}%")
                logger.log("train", iter=iter_num, loss=lossf, lr=lr, dt=dt,
                           mfu=running_mfu,
                           grad_norm=None if grad_norm is None else float(grad_norm))

            iter_num += 1
            local_iter_num += 1
    except BaseException:
        logger.close(status="FAILED")
        raise
    logger.close(status="FINISHED")
    return best_val_loss
