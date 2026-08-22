"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import os
import time
import math
import json
import random
import dataclasses
from contextlib import nullcontext

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from models import get_model
from generator import get_generator

# -----------------------------------------------------------------------------
# default config values
# I/O
out_dir = 'out'
eval_interval = 2000
log_interval = 100
eval_iters = 200
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval
init_from = 'scratch' # 'scratch' or 'resume'
log_file = '' # optional: path to append one JSON line per eval step (training dynamics); '' disables
# wandb logging
wandb_log = False # disabled by default
wandb_project = 'sandbox'
wandb_run_name = 'run' # 'run' + str(time.time())
# data — training/validation data is streamed live from a Generator (no on-disk bins)
dataset = 'kv' # generator name; selects the task from the generator/ registry
gen_params = {} # task hyperparams, merged over the registry defaults for `dataset`
n_val = 1000 # size of the fixed held-out validation set
eval_accuracy = True # also report answer exact-match accuracy during eval
gradient_accumulation_steps = 5 * 8 # used to simulate larger batch sizes
batch_size = 12 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 1024
# model
model = 'base' # which architecture file to load from the models/ folder (--model=<name>)
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?
# composable positional model
pos_encoding = 'nope' # one name or e.g. 'rope+relative_bias'
rope_theta = 10000.0
rpe_num_buckets = 32
rpe_max_distance = 128
cope_max_position = 64
cape_hidden_dim = 32
fope_train_length = 128
fope_init_gain = 0.3
# adamw optimizer
learning_rate = 6e-4 # max learning rate
max_iters = 600000 # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 2000 # how many steps to warm up for
lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.
# system
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = False # use PyTorch 2.0 to compile the model to be faster
seed = 1337 # shared initialization/data seed; override explicitly for seed sweeps
data_seed = 42 # kept separate to preserve the legacy generator stream exactly
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
config['gen_params'] = gen_params # dicts are excluded by the scalar filter above; record it explicitly
# load the chosen architecture from the models/ folder (selected via --model=<name>)
ModelConfig, Model = get_model(model)
# -----------------------------------------------------------------------------

# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)
    if log_file and init_from == 'scratch':
        open(log_file, 'w').close()  # fresh run starts a fresh log; resume appends instead
torch.manual_seed(seed + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# live data loader — samples straight from the generator, no on-disk bins.
# `gen` and `val_items` are built below (needs seed_offset from the DDP setup).
def get_batch(split):
    if split == 'train':
        items = gen.sample_train(batch_size)
    else:
        items = random.sample(val_items, min(batch_size, len(val_items)))
    x_np, y_np = gen.collate(items, block_size)
    x, y = torch.from_numpy(x_np), torch.from_numpy(y_np)
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# build the generator once and derive vocab_size + the held-out val set from it.
# Offset the seed by seed_offset so DDP ranks draw different train samples. Every
# rank calls generate_val to populate val_hashes (needed by sample_train's dedup);
# only the master process actually evaluates val, so per-rank val divergence is moot.
gen = get_generator(dataset, {**gen_params, 'seed': data_seed + seed_offset})
val_items = gen.generate_val(n_val)
meta_vocab_size = gen.vocab_size
print(f"generator '{dataset}' vocab_size = {meta_vocab_size}, held-out val size = {len(val_items)}")

# model init
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout,
                  pos_encoding=pos_encoding, rope_theta=rope_theta,
                  rpe_num_buckets=rpe_num_buckets, rpe_max_distance=rpe_max_distance,
                  cope_max_position=cope_max_position, cape_hidden_dim=cape_hidden_dim,
                  fope_train_length=fope_train_length, fope_init_gain=fope_init_gain)
# Keep only fields declared by the selected architecture.
_model_fields = {f.name for f in dataclasses.fields(ModelConfig)}
_dropped = sorted(set(model_args) - _model_fields)
if _dropped:
    print(f"note: models/{model}.py's ModelConfig has no {_dropped} field(s) — not passing them")
model_args = {k: v for k, v in model_args.items() if k in _model_fields}
if init_from == 'scratch':
    # init a new model from scratch
    print("Initializing a new model from scratch")
    # determine the vocab size we'll use for from-scratch training
    if meta_vocab_size is None:
        print("no meta.pkl found, defaulting to vocab_size of 50304 (a multiple of 64 for efficiency)")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    modelconf = ModelConfig(**model_args)
    model = Model(modelconf)
elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    # resume training from a checkpoint.
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size',
              'pos_encoding', 'rope_theta', 'rpe_num_buckets', 'rpe_max_distance',
              'cope_max_position', 'cape_hidden_dim', 'fope_train_length', 'fope_init_gain']:
        if k in checkpoint_model_args and k in model_args:
            model_args[k] = checkpoint_model_args[k]
    # create the model
    modelconf = ModelConfig(**model_args)
    model = Model(modelconf)
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
# crop down the model block size if desired, using model surgery
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size # so that the checkpoint will have the right value
model.to(device)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None # free up memory

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model) # requires PyTorch 2.0

# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        accs = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
            if eval_accuracy:
                # answer exact-match: a row is correct iff every answer position
                # (Y != -1) is predicted right. Reduces to token accuracy for
                # single-token answers. Non-answer positions are masked out.
                mask = Y != -1
                pred = logits.argmax(dim=-1)
                row_correct = ((pred == Y) | ~mask).all(dim=1)
                accs[k] = row_correct.float().mean().item()
        out[split] = losses.mean()
        if eval_accuracy:
            out[f'{split}_acc'] = accs.mean()
    model.train()
    return out

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

# logging
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# training loop
X, Y = get_batch('train') # fetch the very first batch
t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
raw_model = model.module if ddp else model # unwrap DDP container if needed
if hasattr(raw_model, 'positional_spec'):
    positional_encoding = raw_model.positional_spec.canonical
elif model == 'base':
    positional_encoding = 'wpe'
else:
    positional_encoding = model
positional_counts = (raw_model.get_positional_param_counts()
                     if hasattr(raw_model, 'get_positional_param_counts')
                     else {'trainable': 0, 'frozen': 0, 'total': 0})
if positional_encoding == 'wpe' and positional_counts['total'] == 0 \
        and hasattr(getattr(raw_model, 'transformer', None), 'wpe'):
    n_wpe = raw_model.transformer.wpe.weight.numel()
    positional_counts = {'trainable': n_wpe, 'frozen': 0, 'total': n_wpe}
run_metadata = {
    'positional_encoding': positional_encoding,
    'combination': positional_encoding if '+' in positional_encoding else '',
    'task': dataset,
    'task_variant': gen_params.get('task', gen_params.get('variant', 'default')),
    'seed': seed,
    'train_distribution': gen_params,
    'number_of_parameters': sum(p.numel() for p in raw_model.parameters()),
    'positional_parameters': positional_counts,
}
running_mfu = -1.0
while True:

    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        msg = f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
        if eval_accuracy:
            msg += f", train acc {losses['train_acc']:.4f}, val acc {losses['val_acc']:.4f}"
        print(msg)
        if wandb_log:
            log = {
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr,
                "mfu": running_mfu*100, # convert to percentage
            }
            if eval_accuracy:
                log["train/acc"] = losses['train_acc']
                log["val/acc"] = losses['val_acc']
            wandb.log(log)
        if log_file:
            entry = {
                'iter': iter_num,
                'train_loss': float(losses['train']),
                'val_loss': float(losses['val']),
                'lr': lr,
                'mfu': running_mfu,
                **run_metadata,
            }
            if eval_accuracy:
                entry['train_acc'] = float(losses['train_acc'])
                entry['val_acc'] = float(losses['val_acc'])
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(entry) + '\n')
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                    'run_metadata': run_metadata,
                }
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
    if iter_num == 0 and eval_only:
        break

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            # in DDP training we only need to sync gradients at the last micro step.
            # the official way to do this is with model.no_sync() context manager, but
            # I really dislike that this bloats the code and forces us to repeat code
            # looking at the source of that context manager, it just toggles this variable
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            logits, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y = get_batch('train')
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()
    # clip the gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        # get loss as float. note: this is a CPU-GPU sync point
        # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5: # let the training loop settle a bit
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        break

if ddp:
    destroy_process_group()
