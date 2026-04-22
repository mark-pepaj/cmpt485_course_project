import time

out_dir = 'out-recipes'
wandb_log = False # feel free to turn on
wandb_project = 'recipes'
wandb_run_name = 'ft-' + str(time.time())

dataset = 'recipes'
init_from = 'gpt2' # this is the largest GPT-2 model

# only save checkpoints if the validation loss improves
always_save_checkpoint = False

# these make the total batch size be ~0.5M
# 12 batch size * 1024 block size * 5 gradaccum * 8 GPUs = 491,520
batch_size = 12
block_size = 512
gradient_accumulation_steps = 5 * 8

# this makes total number of tokens be 300B
max_iters = 100000
lr_decay_iters = 100000

# eval stuff
eval_interval = 1000
eval_iters = 200
log_interval = 10

# weight decay
weight_decay = 1e-1
