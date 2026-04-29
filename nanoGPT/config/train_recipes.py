import time

out_dir = 'out-recipes'
wandb_log = False # feel free to turn on
wandb_project = 'recipes'
wandb_run_name = 'ft-' + str(time.time())

dataset = 'recipes'
#init_from = 'gpt2' # this is the largest GPT-2 model

# only save checkpoints if the validation loss improves
always_save_checkpoint = False

# these make the total batch size be ~0.5M
# 12 batch size * 1024 block size * 5 gradaccum * 8 GPUs = 491,520
batch_size = 12
block_size = 256
gradient_accumulation_steps = 1

# this makes total number of tokens be 300B
max_iters = 1000
lr_decay_iters = 1000


#n_embd = 768
#n_embd = 384
n_embd = 256
n_layer = 6
n_head = 6


# eval stuff
eval_interval = 200
eval_iters = 250
log_interval = 25

# weight decay
weight_decay = 1e-1

# dropout
dropout = 0.0


device = 'cpu'
compile = True
eval_iters= 100
log_interval= 1
block_size = 512
batch_size = 12
n_layer = 8
n_head = 8
n_embd = 256
max_iters = 10000
lr_decay_iters = 10000
dropout = 0.0

