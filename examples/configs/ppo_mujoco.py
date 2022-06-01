actor_lr = 3e-4
critic_lr = 1e-3
actor_hidden_dims = [64, 64]
critic_hidden_dims = [64, 64]

actor_output_hidden_dims = [64, ]
critic_output_hidden_dims = [64, ]

reward_scale = 5

actor_repeat_step = 80
critic_repeat_step = 80
clip_range = 0.2
entropy_coeff = 0.0
target_kl = 0.01

buffer_size = 4096
batch_size = 64
max_epoch = 1000
sample_per_epoch = 4000
max_traj_length = 1000

eval_interval = 10
eval_num_traj = 10

log_path = "./tests/log"
name = "debug"