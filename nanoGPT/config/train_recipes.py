import time

out_dir = 'out-recipes'
wandb_log = False # feel free to turn on
wandb_project = 'recipes'
wandb_run_name = 'ft-' + str(time.time())

dataset = 'recipes'

# only save checkpoints if the validation loss improves
always_save_checkpoint = False

# these make the total batch size be ~0.5M
# 12 batch size * 1024 block size * 5 gradaccum * 8 GPUs = 491,520
batch_size = 8 
block_size = 896
gradient_accumulation_steps = 5 * 2
n_head = 12
n_layer = 12
n_embd = 768
# this makes total number of tokens be 300B
max_iters = 10000
lr_decay_iters = 2500

# eval stuff
eval_interval = 50
eval_iters = 20
log_interval = 1

# weight decay
weight_decay = 1e-1

