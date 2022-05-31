# task = "Ant-v3"
# buffer_size = 4096
# hidden_dims = [64, 64]
# lr = 3e-4
# gamma = 0.99
# epoch = 100
# step_per_epoch = 30000
# step_per_collect = 2048
# repeat_per_collect = 10
# batch_size = 64
# training_num = 64
# test_num = 10
# rew_norm = True

# vf_coef = 0.25
# ent_coef = 0.0
# gae_lambda = 0.95
# bound_action_method = "clip"
# lr_decay = True
# max_grad_norm = 0.5
# eps_clip = 0.2
# dual_clip = None
# value_clip = 0
# norm_adv = 0
# recompute_adv = 1

# log_path = "./tests/log"
# name = "debug"

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

buffer_size = 4096
batch_size = 64
max_epoch = 1000
sample_per_epoch = 4000
max_traj_length = 1000

eval_interval = 10
eval_num_traj = 10

log_path = "./tests/log"
name = "debug"