import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from UtilsRL.rl.buffer import TransitionReplayPool
from UtilsRL.rl.policy import SquashedDeterministicPolicy
from UtilsRL.rl.net import MLP
from UtilsRL.logger import TensorboardLogger
from UtilsRL.monitor import Monitor
from UtilsRL.exp import parse_args, setup

# set up logger and arguments
args = parse_args("./examples/configs/ppo_mujoco.py")
logger = TensorboardLogger(args.log_path, args.name)
setup(args, logger)




