from UtilsRL.misc.namespace import NameSpace

debug = False

device = None
seed = 0

# flags
# use_dueling = True # we force dueling
use_categorical = True
use_double = True
use_noisy = True
use_per = True
use_n_step = True
use_rgb = False
n_step = 20
scale_obs = False

update_every = 1
# max_episode_length = 108000
num_frames = 50000000
warmup_frame = 20000
frame_stack = 4
frame_skip = 4
buffer_size = 1e6
target_update_interval = 2000
eval_interval = 10000
log_interval = 10000
eval_num = 5
batch_size = 64
gamma = 0.99
alpha = 0.5
beta = 0.4
prior_eps = 1e-6
lr = 0.0001

# epsilon
max_epsilon = 0.5
min_epsilon = 0.01
epsilon_decay = (max_epsilon - min_epsilon) / num_frames

# beta
max_beta = 1
min_beta = 0.4
beta_decay = (max_beta - min_beta) / num_frames

# reward clip
reward_min = -1
reward_max = 1

# categorical dqn
v_min = -10
v_max = 10
atom_size = 51

output_channel = 64
hidden_dims = [256, ]
channels = [32, 64]
kernels = [5, 5]
strides = [5, 5]



class UtilsRL(NameSpace):
    precision = "float32"

task = "BreakoutNoFrameskip-v4"
log_path = "./tests/log/rainbow"
name = "rainbow"
