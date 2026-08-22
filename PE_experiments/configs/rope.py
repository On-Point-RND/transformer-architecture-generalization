# Rotary positional encoding. Apply after config/small.py.
model = 'positional'
pos_encoding = 'rope'
init_from = 'scratch'
out_dir = 'out/positional/rope'
wandb_run_name = 'positional-rope'
