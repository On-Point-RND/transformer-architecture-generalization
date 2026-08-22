# small model — transformer core ~0.8M params
out_dir = 'out-small-pe-bs256'
wandb_run_name = 'small-pe-bs256'

# model
n_layer = 4
n_head = 4
n_embd = 128      
block_size = 256   
dropout = 0.0
bias = False
dataset = 'kv_retrieval'
gen_params = {'k_card': 80, 'v_card': 80, 'n_pairs': (2, 25)}  # X len = 2*n_pairs+2 = 98 <= block_size
n_val = 1000
init_from = 'resume'

batch_size = 256
gradient_accumulation_steps = 1
max_iters = 80000
lr_decay_iters = 80000
warmup_iters = 100
learning_rate = 1e-4
min_lr = 1e-5
eval_interval = 250
eval_iters = 100
weight_decay = 0.1
