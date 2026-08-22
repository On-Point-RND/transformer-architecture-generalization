out_dir = 'out-large'
wandb_run_name = 'large'

n_layer = 8
n_head = 8
n_embd = 512       
block_size = 128
dropout = 0.0
bias = False
dataset = 'kv'
gen_params = {'k_card': 64, 'v_card': 64, 'n_pairs': 48}  # X len = 2*n_pairs+2 = 98 <= block_size
n_val = 1000

batch_size = 256
gradient_accumulation_steps = 1  
max_iters = 15400
lr_decay_iters = 15400
warmup_iters = 770
learning_rate = 6e-4
min_lr = 6e-5
eval_interval = 500
eval_iters = 200

