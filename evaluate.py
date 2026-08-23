import os
import sys
import json
import argparse
import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, "synthetic_benchmark_suite", "benchmark_generator"))
sys.path.append(os.path.join(current_dir, "synthetic_benchmark_suite"))

from tokenizer.vocab import CharTokenizer
from model.config import ModelConfig, AttentionConfig, FFNConfig
from model.transformer import AlgorithmicTransformer
from model.ffn import compute_iso_param_hidden_dim

@torch.no_grad()
def generate_batch(model, tokenizer, prompts, device, max_new_tokens=50, max_seq_len=None):
    model.eval()
    encoded = [[tokenizer.BOS_ID] + tokenizer.encode(p + " ") for p in prompts]
    prompt_lens = [len(ids) for ids in encoded]
    max_prompt_len = max(prompt_lens)

    if max_seq_len is not None and max_prompt_len + max_new_tokens > max_seq_len:
        raise ValueError(
            f"The longest in the batch({max_prompt_len}) + max_new_tokens ({max_new_tokens}) "
            f"bigger than max_seq_len ({max_seq_len})."
        )

    batch_size = len(prompts)
    input_ids = torch.full((batch_size, max_prompt_len), tokenizer.PAD_ID, dtype=torch.long)
    for i, ids in enumerate(encoded):
        input_ids[i, max_prompt_len - len(ids):] = torch.tensor(ids, dtype=torch.long)
    input_ids = input_ids.to(device)

    padding_mask = torch.zeros((batch_size, max_prompt_len), dtype=torch.bool)
    for i, ids in enumerate(encoded):
        padding_mask[i, : max_prompt_len - len(ids)] = True
    padding_mask = padding_mask.to(device)

    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
    generated_ids = [[] for _ in range(batch_size)]

    for _ in range(max_new_tokens):
        try:
            logits = model(input_ids, key_padding_mask=padding_mask)
        except TypeError:
            logits = model(input_ids)

        next_ids = logits[:, -1, :].argmax(dim=-1)  # (batch,)

        just_finished = (next_ids == tokenizer.EOS_ID) & ~finished
        finished = finished | just_finished

        for i in range(batch_size):
            if not finished[i] or just_finished[i].item():
                if next_ids[i].item() != tokenizer.EOS_ID:
                    generated_ids[i].append(next_ids[i].item())

        if finished.all():
            break

        next_col = torch.where(finished, torch.full_like(next_ids, tokenizer.PAD_ID), next_ids)
        input_ids = torch.cat([input_ids, next_col.unsqueeze(1)], dim=1)
        padding_mask = torch.cat([padding_mask, finished.unsqueeze(1)], dim=1)

    return [tokenizer.decode(ids).strip() for ids in generated_ids]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--split", type=str, default="test_ood")
    parser.add_argument("--run_id", type=str, required=True)
    parser.add_argument("--max_new_tokens", type=int, default=50)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    
    parser.add_argument("--use_swiglu", action="store_true")
    parser.add_argument("--share_kv", action="store_true")
    parser.add_argument("--n_kv_heads", type=int, default=4)
    parser.add_argument("--positional_encoding", type=str, default="rope",
                         choices=["learned_absolute", "sinusoidal", "rope", "alibi", "nope"])
    parser.add_argument("--pattern", type=str, default="full", choices=["full", "local"])
    parser.add_argument("--window_size", type=int, default=None)
    parser.add_argument("--n_global_tokens", type=int, default=0)
    parser.add_argument("--softmax_kind", type=str, default="standard", choices=["standard", "taylor"])
    parser.add_argument("--attention_approx", type=str, default=None,
                         choices=[None, "linformer", "performer", "reformer", "nystromformer"])
    parser.add_argument("--approx_dim", type=int, default=64)
    parser.add_argument("--n_hash_bits", type=int, default=4)
    parser.add_argument("--looped", action="store_true")
    parser.add_argument("--k_iters", type=int, default=6)
    
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = CharTokenizer()

    # 1. Create the configuration exactly as during training
    attn_cfg = AttentionConfig(
        n_heads=4, n_kv_heads=args.n_kv_heads, share_kv=args.share_kv,
        pattern=args.pattern, window_size=args.window_size, n_global_tokens=args.n_global_tokens,
        softmax_kind=args.softmax_kind,
    )
    d_model = 256
    ffn_kind = "swiglu" if args.use_swiglu else "relu"
    ffn_hidden_dim = compute_iso_param_hidden_dim(ffn_kind, d_model, 2 * d_model * 1024)
    ffn_cfg = FFNConfig(kind=ffn_kind, hidden_dim=ffn_hidden_dim)

    config = ModelConfig(
        d_model=d_model, n_layers=6, vocab_size=tokenizer.vocab_size,
        attention=attn_cfg, ffn=ffn_cfg, positional_encoding=args.positional_encoding, max_seq_len=800
    )

    # 2. Initialize the model (with Looped and Approx support)
    if args.looped:
        from model.config import LoopConfig
        from model.transformer import LoopedTransformer
        model = LoopedTransformer(config, LoopConfig(k_iters=args.k_iters)).to(device)
        attn_blocks = [model.block]
    else:
        model = AlgorithmicTransformer(config).to(device)
        attn_blocks = list(model.blocks)

    if args.attention_approx is not None:
        from model.attention_approx import AttentionApproxConfig, build_attention_approx
        approx_cfg = AttentionApproxConfig(
            kind=args.attention_approx, n_heads=4, head_dim=attn_cfg.head_dim,
            projected_dim=args.approx_dim, n_random_features=args.approx_dim,
            n_landmarks=args.approx_dim, n_hash_bits=args.n_hash_bits,
        )
        for block in attn_blocks:
            block.attn = build_attention_approx(config.d_model, approx_cfg, config.max_seq_len).to(device)

    # 3. Load the weights
    checkpoint_path = os.path.join(current_dir, "checkpoints", f"{args.run_id}.pt")
    if not os.path.exists(checkpoint_path):
        print(f" Error: File {checkpoint_path} not found!")
        return

    state_dict = torch.load(checkpoint_path, map_location=device)
    clean_state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(clean_state_dict)
    print(f" Model {args.run_id} successfully loaded!")

    # 4. Проводим оценку
    test_file = os.path.join(current_dir, "test_datasets", f"{args.task}_{args.split}.jsonl")
    print(f" Reading tests from {test_file}")

    with open(test_file, "r", encoding="utf-8") as f:
        all_lines = f.readlines()
    if args.limit is not None:
        all_lines = all_lines[: args.limit]

    examples = [json.loads(line) for line in all_lines]
    max_prompt_chars = max(len(ex["prompt"]) for ex in examples)
    
    correct = 0
    total_attempted = 0
    skipped = 0

    for batch_start in range(0, len(examples), args.batch_size):
        batch = examples[batch_start: batch_start + args.batch_size]
        prompts = [ex["prompt"] for ex in batch]
        targets = [ex["target"] for ex in batch]

        try:
            predictions = generate_batch(model, tokenizer, prompts, device,
                                          max_new_tokens=args.max_new_tokens, max_seq_len=config.max_seq_len)
        except ValueError as e:
            skipped += len(batch)
            print(f" Skipping a batch of {len(batch)} examples (too long): {e}")
            continue

        for prompt, target, prediction in zip(prompts, targets, predictions):
            total_attempted += 1
            if prediction == target:
                correct += 1

            if total_attempted <= 5:
                status = "true" if prediction == target else f" false (expected: {target})"
                print(f"Prompt: {prompt[:80]}... | Output: {prediction} {status}")

    total_in_file = len(examples)
    print("-" * 40)
    print(f"Summary for {args.run_id} on task {args.task} ({args.split}):")
    if skipped > 0:
        print(f"WARNING: {skipped}/{total_in_file} examples skipped (exceeded max_seq_len).")
    if total_attempted > 0:
        accuracy = (correct / total_attempted) * 100
        print(f"Accuracy (Exact Match): {accuracy:.2f}% ({correct}/{total_attempted})")
    print("-" * 40)

if __name__ == "__main__":
    main()