import math
import os
import sys
import tempfile
import multiprocessing.util

# 1. Create a local directory for temporary files within the project
local_temp_dir = os.path.abspath("./tmp_local_system")
os.makedirs(local_temp_dir, exist_ok=True)
os.environ["TMPDIR"] = local_temp_dir
os.environ["TEMP"] = local_temp_dir
os.environ["TMP"] = local_temp_dir
os.environ["TORCHINDUCTOR_CACHE_DIR"] = os.path.join(local_temp_dir, "torch_inductor_cache")
multiprocessing.util.get_temp_dir = lambda: "/dev/shm"


import argparse
from functools import partial
 
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add benchmark_generator to the Python path.
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, "benchmark_generator"))
sys.path.append(current_dir)

# Import tasks
from base import Split
from addition import AdditionTask
from sorting import SortingTask
from dyck import DyckTask
from keyvalue import KVTask
from indexing import IndexTask
from composition import FuncTask

# Import custom modules
from tokenizer.vocab import CharTokenizer
from data.task_mixture import MultitaskIterableDataset, collate_fn
from model.config import ModelConfig, AttentionConfig, FFNConfig
from model.transformer import AlgorithmicTransformer
from model.ffn import compute_iso_param_hidden_dim
 
SEED = 42
MAX_SEQ_LEN = 800
LEARNING_RATE = 1e-3 
MIN_LR = 1e-4
WARMUP_ITERS = 1000
LR_DECAY_ITERS = 100000
TOTAL_STEPS = 120000  
EVAL_INTERVAL = 1000
EVAL_BATCHES = 20        
PATIENCE = 15
MIN_DELTA = 1e-4
 
def get_lr(it: int) -> float:
    """Linear warmup -> cosine decay -> plateau at min_lr."""
    if it < WARMUP_ITERS:
        return LEARNING_RATE * (it + 1) / (WARMUP_ITERS + 1)
    if it > LR_DECAY_ITERS:
        return MIN_LR
    decay_ratio = (it - WARMUP_ITERS) / (LR_DECAY_ITERS - WARMUP_ITERS)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return MIN_LR + coeff * (LEARNING_RATE - MIN_LR)
 
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=str, default="all")
    parser.add_argument("--use_swiglu", action="store_true")
    parser.add_argument("--share_kv", action="store_true", help="Tie Key and Value projections (K=V)")
    parser.add_argument("--n_kv_heads", type=int, default=4,
                         help="Fixed n_heads=4: 4=MHA, 2=GQA (2 groups), 1=MQA")
    parser.add_argument("--positional_encoding", type=str, default="rope",
                         choices=["learned_absolute", "sinusoidal", "rope", "alibi", "nope"])
    parser.add_argument("--pattern", type=str, default="full", choices=["full", "local"])
    parser.add_argument("--window_size", type=int, default=None,
                         help="Required when --pattern local")
    parser.add_argument("--n_global_tokens", type=int, default=0,
                         help="Number of 'sink' tokens, always visible under --pattern local")
    parser.add_argument("--softmax_kind", type=str, default="standard", choices=["standard", "taylor"])
    parser.add_argument("--attention_approx", type=str, default=None,
                         choices=[None, "linformer", "performer", "reformer", "nystromformer"])
    parser.add_argument("--approx_dim", type=int, default=64,
                         help="projected_dim (linformer) / n_random_features (performer) / n_landmarks (nystromformer)")
    parser.add_argument("--n_hash_bits", type=int, default=4, help="Only for --attention_approx reformer")
    parser.add_argument("--looped", action="store_true")
    parser.add_argument("--k_iters", type=int, default=6, help="Number of iterations for the shared block under --looped")
    parser.add_argument("--run_id", type=str, required=True)
    args = parser.parse_args()
 
    if args.pattern == "local" and args.window_size is None:
        parser.error("--window_size is required when --pattern local")

    torch.set_float32_matmul_precision('high')
        
    # 0. Fix the random seed for reproducibility
    torch.manual_seed(SEED)
 
    # 1. Setup device (GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Запуск обучения: {args.run_id} | Устройство: {device}")
 
    # 2. Initialize Tokenizer and filter Tasks
    tokenizer = CharTokenizer()
 
    all_task_objs = {
        "addition": AdditionTask(),
        "sorting": SortingTask(),
        "dyck": DyckTask(),
        "kv": KVTask(),
        "indexing": IndexTask(),
        "func": FuncTask()
    }
 
    if args.tasks == "all":
        active_tasks = list(all_task_objs.values())
    else:
        task_names = [t.strip() for t in args.tasks.split(",")]
        active_tasks = [all_task_objs[name] for name in task_names]
 
    # 3. Setup Dataset (train: infinite stream, val: for early stopping)    
    print(f"Preparing data stream for tasks: {[t.name for t in active_tasks]}...")
    train_dataset = MultitaskIterableDataset(active_tasks, Split.TRAIN, tokenizer, max_seq_len=MAX_SEQ_LEN, seed=SEED)
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        num_workers=8, 
        pin_memory=True,  
        collate_fn=partial(collate_fn, pad_id=tokenizer.PAD_ID),
    )
    data_iterator = iter(train_loader)
 
    val_dataset = MultitaskIterableDataset(active_tasks, Split.VAL, tokenizer, max_seq_len=MAX_SEQ_LEN, seed=SEED + 1)
    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        collate_fn=partial(collate_fn, pad_id=tokenizer.PAD_ID),
    )
    val_iterator = iter(val_loader)
 
    # 4. Create Model Configuration
    print(f"Initializing Transformer (SwiGLU={args.use_swiglu}, Share_KV={args.share_kv}, "
          f"n_kv_heads={args.n_kv_heads}, PE={args.positional_encoding}, pattern={args.pattern}, "
          f"softmax={args.softmax_kind}, attention_approx={args.attention_approx}, looped={args.looped})...")
 
    attn_cfg = AttentionConfig(
        n_heads=4, n_kv_heads=args.n_kv_heads, share_kv=args.share_kv,
        pattern=args.pattern, window_size=args.window_size, n_global_tokens=args.n_global_tokens,
        softmax_kind=args.softmax_kind,
    )
 
    d_model = 256
    target_ffn_params = 2 * d_model * 1024
    # Iso-parameter FFN: Target parameter count is based on the standard ReLU variant
    ffn_kind = "swiglu" if args.use_swiglu else "relu"
    ffn_hidden_dim = compute_iso_param_hidden_dim(ffn_kind, d_model=d_model, target_params=target_ffn_params)
    ffn_cfg = FFNConfig(kind=ffn_kind, hidden_dim=ffn_hidden_dim)
 
    config = ModelConfig(
        d_model=d_model,
        n_layers=6,
        vocab_size=tokenizer.vocab_size,
        attention=attn_cfg,
        ffn=ffn_cfg,
        positional_encoding=args.positional_encoding,
        max_seq_len=MAX_SEQ_LEN,
    )
 
    if args.looped:
        from model.config import LoopConfig
        from model.transformer import LoopedTransformer
        model = LoopedTransformer(config, LoopConfig(k_iters=args.k_iters)).to(device)
        model = torch.compile(model)
        attn_blocks = [model._orig_mod.block] 
    else:
        model = AlgorithmicTransformer(config).to(device)
        model = torch.compile(model)
        attn_blocks = list(model._orig_mod.blocks)
 
    if args.attention_approx is not None:
        from model.attention_approx import AttentionApproxConfig, build_attention_approx
        approx_cfg = AttentionApproxConfig(
            kind=args.attention_approx, n_heads=4, head_dim=attn_cfg.head_dim,
            projected_dim=args.approx_dim, n_random_features=args.approx_dim,
            n_landmarks=args.approx_dim, n_hash_bits=args.n_hash_bits,
        )
        for block in attn_blocks:
            block.attn = build_attention_approx(config.d_model, approx_cfg, config.max_seq_len).to(device)
 
    print(f"Model created! Approx. parameters: {model._orig_mod.num_parameters():,}")

    # 5. Optimizer and Loss Function
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
 
    @torch.no_grad()
    def estimate_val_loss(n_batches: int = EVAL_BATCHES) -> float:
        model.eval()
        losses = []
        for _ in range(n_batches):
            x, y = next(val_iterator)
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            losses.append(loss.item())
        model.train()
        return sum(losses) / len(losses)
 
    os.makedirs("checkpoints", exist_ok=True)
    best_val_loss = float("inf")
    patience_counter = 0
 
    # 6. TRAINING LOOP
    print("Starting training loop...")
 
    model.train()
    for step in range(1, TOTAL_STEPS + 1):
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
 
        x, y = next(data_iterator)
        x, y = x.to(device), y.to(device)
 
        logits = model(x)
        loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
 
        if step % 100 == 0:
            print(f"Step {step}/{TOTAL_STEPS} | lr={lr:.2e} | Loss: {loss.item():.4f}")
            print("  Task skip-statistics (exceeded MAX_SEQ_LEN):")
            print(train_dataset.skip_report())
 
        if step % EVAL_INTERVAL == 0:
            val_loss = estimate_val_loss()
            improved = val_loss < best_val_loss - MIN_DELTA
 
            if improved:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model._orig_mod.state_dict(), f"checkpoints/{args.run_id}_best.pt")
            else:
                patience_counter += 1
 

            print(f"  [val] шаг {step} | val_loss={val_loss:.4f} | best={best_val_loss:.4f} "
                  f"| patience={patience_counter}/{PATIENCE}")
 
            if patience_counter >= PATIENCE:
                print(f"Early stopping at step {step}: val_loss has not improved for "
                      f"{PATIENCE} consecutive checks (best val_loss={best_val_loss:.4f}).")
                break

    checkpoint_path = f"checkpoints/{args.run_id}.pt"
    torch.save(model._orig_mod.state_dict(), checkpoint_path)
    print(f"   Training complete! Final model weights saved to {checkpoint_path}")
    print(f"   Best validation loss version: checkpoints/{args.run_id}_best.pt")
 
if __name__ == "__main__":
    main()