task = "Ant-v3"
buffer_size = 4096
hidden_dims = [64, 64]
lr = 3e-4
gamma = 0.99
epoch = 100
step_per_epoch = 30000
step_per_collect = 2048
repeat_per_collect = 10
batch_size = 64
training_num = 64
test_num = 10
rew_norm = True

vf_coef = 0.25
ent_coef = 0.0
gae_lambda = 0.95
bound_action_method = "clip"
lr_decay = True
max_grad_norm = 0.5
eps_clip = 0.2
dual_clip = None
value_clip = 0
norm_adv = 0
recompute_adv = 1

log_path = "./tests/log"
name = "debug"
