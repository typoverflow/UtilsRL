from UtilsRL.misc.namespace import NameSpace

debug = False

device = 0
seed = 0
actor_lr = 3e-4
critic_lr = 3e-4
actor_hidden_dims = [64, 64]
critic_hidden_dims = [64, 64]

actor_output_hidden_dims = [64, ]
critic_output_hidden_dims = [64, ]

# reward_scale = 5
warmup_epoch = 5
repeat_step = 10
clip_range = 0.2
entropy_coeff = 0.0
target_kl = 3

buffer_size = 2048
batch_size = 64
max_epoch = 2000
sample_per_epoch = 2048
max_traj_length = 1000

eval_interval = 10
eval_num_traj = 10

class UtilsRL(NameSpace):
    precision = "float32"

task = "HalfCheetah-v3"
log_path = "./tests/log"
name = "debug_buffer"

class wandb(NameSpace):
    activate = False
    project = "ppo_mujoco"
    entity = "your_account"