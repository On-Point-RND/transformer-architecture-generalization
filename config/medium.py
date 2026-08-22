out_dir = 'out-medium'
wandb_run_name = 'medium'

n_layer = 7
n_head = 6
n_embd = 240        
block_size = 128
dropout = 0.0
bias = False
dataset = 'kv'
gen_params = {'k_card': 64, 'v_card': 64, 'n_pairs': 48}  # X len = 2*n_pairs+2 = 98 <= block_size
n_val = 1000

batch_size = 256
gradient_accumulation_steps = 1  
max_iters = 4500
lr_decay_iters = 4500
warmup_iters = 225
learning_rate = 6e-4
min_lr = 6e-5
eval_interval = 250
eval_iters = 200
