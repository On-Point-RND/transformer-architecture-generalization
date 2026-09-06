"""Isolated, resumable PE benchmark. See README.md for the exact protocol."""
import sys
sys.dont_write_bytecode = True

import argparse
import contextlib
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
import traceback

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RUNTIME = HERE / "runtime"
sys.path.insert(0, str(RUNTIME))


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    os.replace(temp, path)


def atomic_torch(path, value):
    temp = path.with_suffix(".tmp")
    torch.save(value, temp)
    os.replace(temp, path)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(item):
    return hashlib.sha256(item.prompt.tobytes()).digest()


def task_params(name, cfg, stage=None, length=None):
    if name in ("kv_retrieval", "sorting"):
        # Repository generators use an EXCLUSIVE upper bound.
        size = length if length is not None else (2, 2 * stage + 1)
        if name == "kv_retrieval":
            return dict(k_card=cfg["vocab_card"], v_card=cfg["vocab_card"], n_pairs=size)
        return dict(v_card=cfg["vocab_card"], n_items=size)
    return {"length_range": tuple(cfg["lab_train_lengths"]) if length is None else (length, length)}


def points(name, cfg):
    if name == "kv_retrieval":
        return list(range(cfg["kv_test_lengths"][0], cfg["kv_test_lengths"][1] + 1))
    if name == "sorting":
        return list(range(cfg["sorting_test_lengths"][0], cfg["sorting_test_lengths"][1] + 1))
    return cfg["lab_test_lengths"]


def is_id(name, length, cfg):
    lo, hi = (2, 2 * cfg["cl_stages"]) if name in ("kv_retrieval", "sorting") else cfg["lab_train_lengths"]
    return lo <= length <= hi


def make_task(name, cfg, seed, stage=None, length=None):
    return get_task(name, {**task_params(name, cfg, stage, length), "seed": seed})


def dataset(name, cfg, seed, directory):
    path = directory / f"data_seed={seed}.pt"
    if path.exists():
        return torch.load(path, weights_only=False, map_location="cpu")
    print(f"Preparing held-out data: {name}, seed={seed}", flush=True)
    reserved = set()

    def draw(task, count):
        result = []
        rejects = 0
        while len(result) < count:
            item = task.sample(1)[0]
            key = digest(item)
            if key in reserved:
                rejects += 1
                if rejects >= 10000:
                    raise ValueError(f"Held-out space exhausted for {name}; increase vocabulary")
                continue
            rejects = 0
            reserved.add(key)
            result.append(item)
        return result

    stages = range(1, cfg["cl_stages"] + 1) if name in ("kv_retrieval", "sorting") else [0]
    vals = {stage: draw(make_task(name, cfg, seed + 100000 + stage, stage=stage), cfg["val_samples"])
            for stage in stages}
    tests = {}
    for length in points(name, cfg):
        tests[length] = draw(make_task(name, cfg, seed + 200000 + length, length=length), cfg["test_samples"])
    data = {"val": vals, "test": tests, "reserved": reserved}
    atomic_torch(path, data)
    return data


def batch(task, items, device):
    length = max(len(x.prompt) + len(x.answer) - 1 for x in items)
    x, y = task.collate(items, length)
    return torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)


def precision(device):
    if not device.startswith("cuda"):
        return torch.float32
    with torch.cuda.device(device):
        native_bf16 = torch.cuda.get_device_capability()[0] >= 8 and torch.cuda.is_bf16_supported()
    return torch.bfloat16 if native_bf16 else torch.float16


def autocast(device):
    dtype = precision(device)
    return (torch.autocast("cuda", dtype=dtype)
            if dtype != torch.float32 else contextlib.nullcontext())


@contextlib.contextmanager
def preserve_rng():
    state = checkpoint.rng_state()
    try:
        yield
    finally:
        checkpoint.restore_rng(state)


def evaluate(model, task, items, device, batch_size):
    with torch.no_grad():
        return _evaluate(model, task, items, device, batch_size)


def _evaluate(model, task, items, device, batch_size):
    model.eval()
    correct = 0
    total_loss = 0.0
    tokens = 0
    offset = 0
    while offset < len(items):
        chunk = items[offset:offset + batch_size]
        try:
            x, y = batch(task, chunk, device)
            with autocast(device):
                logits, loss = model(x, y)
            mask = y != -1
            batch_correct = int(((logits.argmax(-1) == y) | ~mask).all(-1).sum().item())
            count = int(mask.sum().item())
            correct += batch_correct
            total_loss += float(loss) * count
            tokens += count
            offset += len(chunk)
            del logits, loss, x, y, mask
        except torch.cuda.OutOfMemoryError:
            if batch_size == 1:
                raise
            # Exception traceback can keep tensors alive until leaving this block.
            batch_size = max(1, batch_size // 2)
        else:
            continue
        x = y = logits = loss = mask = None
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    if not math.isfinite(total_loss):
        raise FloatingPointError("Non-finite evaluation loss")
    return {"acc": correct / len(items), "correct": correct, "n": len(items), "loss": total_loss / tokens}


class Worker:
    def __init__(self, name, encoding, seed, cfg, directory, data, device):
        self.name, self.encoding, self.seed = name, encoding, seed
        self.cfg, self.data, self.device = cfg, data, device
        self.directory = directory / f"seed={seed}" / encoding
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / "last.pt"
        self.cl = name in ("kv_retrieval", "sorting")
        self.state = dict(epoch=0, step=0, stage=1 if self.cl else 0, stage_epoch=0,
                          best=-1.0, stale=0, discovered=False, stop_epoch=None,
                          stop_reason=None, aligned=False, micro_batch=cfg["micro_batch_size"])
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        vocab = make_task(name, cfg, seed, stage=self.state["stage"]).vocab_size
        self.model_config = ModelConfig(**cfg["model"], vocab_size=vocab, pos_encoding=encoding)
        self.model = Model(self.model_config).to(device)
        opt = OptimizerConfig(learning_rate=cfg["learning_rate"], schedule="constant",
                              weight_decay=cfg["weight_decay"], grad_clip=cfg["grad_clip"])
        self.optimizer = self.model.configure_optimizers(opt, "cuda" if device.startswith("cuda") else "cpu")
        self.scaler = torch.amp.GradScaler("cuda", enabled=precision(device) == torch.float16)
        loaded = torch.load(self.path, map_location="cpu", weights_only=False) if self.path.exists() else None
        if loaded:
            self.state = loaded["state"]
            self.model.load_state_dict(loaded["model"])
            self.optimizer.load_state_dict(loaded["optimizer"])
            if loaded.get("scaler"):
                self.scaler.load_state_dict(loaded["scaler"])
        self.reset_task()
        if loaded:
            self.task.load_state_dict(loaded["task"])
            checkpoint.restore_rng(loaded["rng"])
        else:
            self.save()

    def reset_task(self):
        self.task = make_task(self.name, self.cfg, self.seed + 300000 + self.state["stage"], stage=self.state["stage"])

    def save(self):
        atomic_torch(self.path, {"model": self.model.state_dict(), "optimizer": self.optimizer.state_dict(),
                                "scaler": self.scaler.state_dict(),
                                "state": self.state, "task": self.task.state_dict(), "rng": checkpoint.rng_state(),
                                "model_config": vars(self.model_config), "experiment": self.cfg})
        atomic_json(self.directory / "status.json", self.state)

    def log(self, event, **values):
        row = dict(event=event, epoch=self.state["epoch"], step=self.state["step"], stage=self.state["stage"], **values)
        with (self.directory / "metrics.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row) + "\n")
        print(f"{self.name}/{self.seed}/{self.encoding}: {row}", flush=True)

    def train_step(self):
        items = []
        rejects = 0
        while len(items) < self.cfg["batch_size"]:
            item = self.task.sample(1)[0]
            if digest(item) not in self.data["reserved"]:
                items.append(item)
                rejects = 0
            else:
                rejects += 1
                if rejects >= 10000:
                    raise ValueError("Training support exhausted by held-out data")
        # Token-weighted accumulation equals the loss of the full logical batch.
        total_tokens = sum(len(x.answer) for x in items)
        lr = self.cfg["learning_rate"] * min(1.0, (self.state["step"] + 1) / max(1, self.cfg["warmup_steps"]))
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        self.model.train()
        rng = checkpoint.rng_state()
        overflow_retries = 0
        while True:
            self.optimizer.zero_grad(set_to_none=True)
            oom = False
            try:
                for start in range(0, len(items), self.state["micro_batch"]):
                    chunk = items[start:start + self.state["micro_batch"]]
                    x, y = batch(self.task, chunk, self.device)
                    with autocast(self.device):
                        _, loss = self.model(x, y)
                        loss = loss * (sum(len(x.answer) for x in chunk) / total_tokens)
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Non-finite training loss")
                    self.scaler.scale(loss).backward()
                    del x, y, loss, _
            except torch.cuda.OutOfMemoryError:
                if self.state["micro_batch"] == 1:
                    raise
                oom = True
            if not oom:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.cfg["grad_clip"],
                    error_if_nonfinite=not self.scaler.is_enabled())
                old_scale = self.scaler.get_scale()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                if self.scaler.get_scale() >= old_scale:
                    break
                # An overflow must not consume an optimizer step or a new batch.
                overflow_retries += 1
                if overflow_retries >= 16:
                    raise FloatingPointError("FP16 gradients still overflow after 16 retries")
                checkpoint.restore_rng(rng)
                self.log("amp_retry", scale=self.scaler.get_scale())
                continue
            # Retry the SAME logical batch, before any optimizer update.
            self.optimizer.zero_grad(set_to_none=True)
            x = y = loss = _ = None
            self.state["micro_batch"] = max(1, self.state["micro_batch"] // 2)
            checkpoint.restore_rng(rng)
            torch.cuda.empty_cache()
            self.log("oom_retry", micro_batch=self.state["micro_batch"])
        self.state["step"] += 1

    def advance(self, target=None):
        s, cfg = self.state, self.cfg
        aligning = target is not None
        if (aligning and s["aligned"]) or (not aligning and s["discovered"]):
            return
        # One quantum per model makes the unbounded CL tail fair to every encoding.
        end = s["epoch"] + cfg["eval_every_epochs"]
        if aligning:
            end = min(end, target)
        while s["epoch"] < end:
            for _ in range(cfg["steps_per_epoch"]):
                self.train_step()
            s["epoch"] += 1
            s["stage_epoch"] += 1
            if s["epoch"] % 25 == 0:
                self.log("progress", stage_epoch=s["stage_epoch"])
            stage_end = self.cl and s["stage"] < cfg["cl_stages"] and s["stage_epoch"] == cfg["cl_stage_epochs"]
            should_eval = s["epoch"] % cfg["eval_every_epochs"] == 0 or stage_end or s["epoch"] == end
            if not should_eval:
                continue
            with preserve_rng():
                scores = evaluate(self.model, self.task, self.data["val"][s["stage"]], self.device, cfg["eval_batch_size"])
            improved = scores["acc"] > s["best"]
            s["best"] = max(s["best"], scores["acc"])
            s["stale"] = 0 if improved else s["stale"] + 1
            self.log("alignment_val" if aligning else "val", **scores, stale=s["stale"])
            if not aligning and (not self.cl or s["stage"] == cfg["cl_stages"]):
                reason = "val_acc_1" if scores["correct"] == scores["n"] else None
                if not self.cl and s["stale"] >= cfg["patience"]:
                    reason = reason or "patience"
                if reason:
                    s.update(discovered=True, stop_epoch=s["epoch"], stop_reason=reason)
            if stage_end:
                s.update(stage=s["stage"] + 1, stage_epoch=0, best=-1.0, stale=0)
                self.reset_task()
            self.save()
            if s["discovered"] and not aligning:
                break
        if aligning and s["epoch"] == target:
            s["aligned"] = True
            self.save()

    def test(self):
        path = self.directory / "acc_L.json"
        rows = read_json(path) if path.exists() else []
        done = {row["L"] for row in rows}
        for length, items in self.data["test"].items():
            if length in done:
                continue
            task = make_task(self.name, self.cfg, self.seed, length=length)
            scores = evaluate(self.model, task, items, self.device, self.cfg["eval_batch_size"])
            rows.append(dict(task=self.name, encoding=self.encoding, seed=self.seed, L=length,
                             split="ID" if is_id(self.name, length, self.cfg) else "OOD",
                             epoch=self.state["epoch"], **scores))
            atomic_json(path, rows)
            self.log("test", L=length, **scores)


def status(directory, seed, encoding):
    # Checkpoint is authoritative: process may have died before status.json replacement.
    path = directory / f"seed={seed}" / encoding / "last.pt"
    return torch.load(path, map_location="cpu", weights_only=False)["state"] if path.exists() else {}


def run_task(name, cfg, output, device):
    directory = output / name
    directory.mkdir(exist_ok=True)
    data = {seed: dataset(name, cfg, seed, directory) for seed in cfg["seeds"]}
    runs = [(seed, encoding) for seed in cfg["seeds"] for encoding in cfg["encodings"]]
    while True:
        pending = [(seed, enc) for seed, enc in runs if not status(directory, seed, enc).get("discovered")]
        if not pending:
            break
        for seed, enc in pending:
            worker = Worker(name, enc, seed, cfg, directory, data[seed], device)
            worker.advance()
            del worker
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
        yield  # Allow the other CL task to advance even if this one never hits 1.0.
    target = max(status(directory, seed, enc)["stop_epoch"] for seed, enc in runs)
    atomic_json(directory / "alignment.json", {"target_epochs": target, "scope": "all encodings and seeds"})
    for seed, enc in runs:
        worker = Worker(name, enc, seed, cfg, directory, data[seed], device)
        while not worker.state["aligned"]:
            worker.advance(target)
        worker.test()
        del worker
        if device.startswith("cuda"):
            torch.cuda.empty_cache()


def aggregate(output):
    rows = []
    for path in sorted(output.glob("*/seed=*/*/acc_L.json")):
        rows.extend(read_json(path))
    if not rows:
        return
    path = output / "acc_L.csv"
    temp = path.with_suffix(".tmp")
    with temp.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temp, path)
    os.environ["MPLCONFIGDIR"] = str(output / "cache" / "matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for name in sorted({row["task"] for row in rows}):
        fig, ax = plt.subplots(figsize=(10, 6))
        for enc in sorted({row["encoding"] for row in rows if row["task"] == name}):
            selected = [r for r in rows if r["task"] == name and r["encoding"] == enc]
            lengths = sorted({r["L"] for r in selected})
            means = [np.mean([r["acc"] for r in selected if r["L"] == length]) for length in lengths]
            ax.plot(lengths, means, label=enc, marker=".", markersize=3)
        id_end = max(r["L"] for r in rows if r["task"] == name and r["split"] == "ID")
        ax.axvline(id_end, color="gray", linestyle="--", label="ID boundary")
        ax.set(xlabel="L (pairs / items / prompt tokens)", ylabel="Exact-match accuracy", ylim=(-0.02, 1.02), title=name)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(output / name / "acc_L.png", dpi=160)
        plt.close(fig)


@contextlib.contextmanager
def keep_awake():
    """Prevent Windows idle sleep while running; restore the previous setting."""
    previous = None
    if os.name == "nt":
        import ctypes
        previous = ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
        if not previous:
            print("Could not inhibit idle sleep; check Windows power settings", flush=True)
    try:
        yield
    finally:
        if previous:
            ctypes.windll.kernel32.SetThreadExecutionState(previous)


@contextlib.contextmanager
def lock_output(output):
    stream = (output / "run.lock").open("a+b")
    stream.seek(0)
    if not stream.read(1):
        stream.write(b"0")
        stream.flush()
    stream.seek(0)
    try:
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        stream.close()
        raise RuntimeError(f"Another runner holds {output}") from None
    try:
        yield
    finally:
        stream.close()


class Tee:
    def __init__(self, terminal, log):
        self.terminal, self.log = terminal, log

    def write(self, value):
        self.terminal.write(value)
        self.log.write(value)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()


def validate(cfg, names):
    for key in ("batch_size", "micro_batch_size", "eval_batch_size", "steps_per_epoch", "eval_every_epochs",
                "patience", "val_samples", "test_samples", "cl_stage_epochs", "cl_stages", "vocab_card"):
        if not isinstance(cfg[key], int) or cfg[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if not cfg["encodings"] or not cfg["seeds"] or len(set(cfg["encodings"])) != len(cfg["encodings"]) or len(set(cfg["seeds"])) != len(cfg["seeds"]):
        raise ValueError("Encodings and seeds must be nonempty and unique")
    for enc in cfg["encodings"]:
        # Validate combinations via ModelConfig, while keeping directory names safe.
        if not enc or any(part not in SLOTS for part in enc.split("+")):
            raise ValueError(f"Invalid encoding: {enc}")
    for name in names:
        for length in points(name, cfg):
            sample = make_task(name, cfg, 42, length=length).sample(1)[0]
            if len(sample.prompt) + len(sample.answer) - 1 > cfg["model"]["block_size"]:
                raise ValueError(f"{name} L={length} exceeds block_size")
        if not any(is_id(name, length, cfg) for length in points(name, cfg)) or not any(not is_id(name, length, cfg) for length in points(name, cfg)):
            raise ValueError(f"{name} needs both ID and OOD test points")
        task = make_task(name, cfg, 42, stage=cfg["cl_stages"])
        for sample in task.sample(50):
            if len(sample.prompt) + len(sample.answer) - 1 > cfg["model"]["block_size"]:
                raise ValueError(f"{name}: train data exceeds block size")
        for enc in cfg["encodings"]:
            model = Model(ModelConfig(**cfg["model"], vocab_size=task.vocab_size, pos_encoding=enc))
            del model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=HERE / "experiment.json")
    parser.add_argument("--output", type=Path, default=HERE / "results" / "rtx2070")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--tasks", nargs="+")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to((HERE / "results").resolve()):
        parser.error("--output must be inside PE_EXP/results")
    output.mkdir(parents=True, exist_ok=True)
    os.environ["TORCH_HOME"] = str(output / "cache" / "torch")
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(output / "cache" / "inductor")
    os.environ["TRITON_CACHE_DIR"] = str(output / "cache" / "triton")
    os.environ["CUDA_CACHE_PATH"] = str(output / "cache" / "cuda")
    os.environ["MPLCONFIGDIR"] = str(output / "cache" / "matplotlib")
    temp_dir = output / "cache" / "tmp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    for key in ("TMP", "TEMP", "TMPDIR"):
        os.environ[key] = str(temp_dir)
    global torch, np, checkpoint, Model, ModelConfig, OptimizerConfig, get_task, SLOTS
    import torch
    import numpy as np
    from core import checkpoint
    from core.config import OptimizerConfig
    from models.positional import Model, Config as ModelConfig, SLOTS
    from tasks import get_task
    from tasks.positional_lab import POSITIONAL_TASK_DEFAULTS
    cfg = read_json(args.config)
    names = args.tasks or [*POSITIONAL_TASK_DEFAULTS, "kv_retrieval", "sorting"]
    if len(names) != len(set(names)) or set(names) - set([*POSITIONAL_TASK_DEFAULTS, "kv_retrieval", "sorting"]):
        parser.error("Unknown or duplicated task")
    with lock_output(output), keep_awake(), (output / "console.log").open("a", encoding="utf-8") as logfile:
        with contextlib.redirect_stdout(Tee(sys.stdout, logfile)), contextlib.redirect_stderr(Tee(sys.stderr, logfile)):
            validate(cfg, names)
            sources = [HERE / "run.py", *sorted(RUNTIME.rglob("*.py"))]
            manifest = {"protocol_version": 1, "config": cfg, "tasks": names,
                        "sources": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}}
            print(f"Plan: {len(names)} tasks x {len(cfg['encodings'])} encodings x {len(cfg['seeds'])} seeds; output={output}", flush=True)
            if args.dry_run:
                atomic_json(output / "plan.json", manifest)
                return 0
            path = output / "manifest.json"
            if path.exists() and read_json(path) != manifest:
                raise RuntimeError("Config/code/task list differs from saved manifest; use a new --output directory")
            atomic_json(path, manifest)
            if args.device.startswith("cuda"):
                if not torch.cuda.is_available():
                    raise RuntimeError("CUDA unavailable. Install a CUDA PyTorch build or explicitly use --device cpu")
                torch.cuda.get_device_properties(torch.device(args.device))
            print(f"Device: {args.device}; precision: {precision(args.device)}; "
                  f"microbatch: {cfg['micro_batch_size']}; eval batch: {cfg['eval_batch_size']}", flush=True)
            torch.set_num_threads(min(8, os.cpu_count() or 1))
            atomic_json(output / "environment.json", {"python": sys.version, "torch": torch.__version__,
                        "numpy": np.__version__, "device": args.device, "started": time.strftime("%Y-%m-%d %H:%M:%S")})
            failures = {}
            try:
                for name in [n for n in names if n not in ("kv_retrieval", "sorting")]:
                    try:
                        for _ in run_task(name, cfg, output, args.device):
                            pass
                    except Exception:
                        failures[name] = traceback.format_exc()
                        print(failures[name], file=sys.stderr, flush=True)
                        atomic_json(output / "failures.json", failures)
                    aggregate(output)
                active = {name: run_task(name, cfg, output, args.device)
                          for name in names if name in ("kv_retrieval", "sorting")}
                while active:
                    for name in list(active):
                        try:
                            next(active[name])
                        except StopIteration:
                            del active[name]
                            aggregate(output)
                        except Exception:
                            failures[name] = traceback.format_exc()
                            print(failures[name], file=sys.stderr, flush=True)
                            del active[name]
                            atomic_json(output / "failures.json", failures)
            finally:
                aggregate(output)
            atomic_json(output / "failures.json", failures)
            atomic_json(output / "completion.json", {"complete": not failures, "failed_tasks": list(failures)})
            return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
