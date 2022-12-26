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
n_step = 3


num_frames = 20000000
warmup_frame = 80000
frame_stack = 4
frame_skip = 4
buffer_size = 1e6
target_update_interval = 3200
eval_interval = 10000
log_interval = 5000
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
epsilon_decay = 2.5e-7

# beta
max_beta = 1
min_beta = 0.6
beta_decay = 2e-7

# reward clip
reward_min = None
reward_max = None

# categorical dqn
v_min = -10
v_max = 10
atom_size = 51

output_channel = 64
hidden_dims = [512, ]



class UtilsRL(NameSpace):
    precision = "float32"

task = "BreakoutNoFrameskip-v4"
log_path = "./tests/log/rainbow"
name = "rainbow"
