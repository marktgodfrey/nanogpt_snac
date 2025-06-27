# config for training GPT-2 (124M) down to very nice loss of ~2.85 on 1 node of 8X A100 40GB
# launch as the following (e.g. in a screen session) and wait ~5 days:
# $ torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py

wandb_log = True
wandb_project = 'snac-bb'
# wandb_run_name = 'mini-gpt'

dataset = 'sh_cork'
# init_from = 'resume'

# tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
# 4 batch size * 2048 block size * 8 gradaccum * 8 GPUs = 524,288
vocab_size = 16384 + 2  # 4 VQ levels, 4096 codes each, plus start/end special tokens
block_size = 4096
batch_size = 2
gradient_accumulation_steps = 8


# 'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
# 'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
# 'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
# 'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params

# "mini-gpt" -> 36M params
# n_layer = 12
# n_head = 12
# n_embd = 768

# 71M params
# n_layer = 12
# n_head = 8
# n_embd = 512

# n_layer = 12
# n_head = 12
# n_embd = 768
# dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+

# n_layer = 24
# n_head = 16
# n_embd = 1024

# n_layer = 2
# n_head = 2
# n_embd = 256

# n_layer = 4
# n_head = 4
# n_embd = 256

n_layer = 24
n_head = 24
n_embd = 1536

dropout = 0.2

# learning_rate = 1e-3  # with baby networks can afford to go a bit higher
learning_rate = 3e-4  # max learning rate
max_iters = 6000
lr_decay_iters = 6000  # make equal to max_iters usually
# min_lr = 1e-4  # learning_rate / 10 usually
min_lr = 3e-5  # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# beta2 = 0.99  # make a bit bigger because number of tokens per iter is small
beta1 = 0.9
beta2 = 0.95

warmup_iters = 2000  # not super necessary potentially

# eval stuff
eval_interval = 100  # keep frequent because we'll overfit
eval_iters = 200
log_interval = 10

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False

# weight decay
weight_decay = 1e-1
# weight_decay = 1e-4
# weight_decay = 0.
