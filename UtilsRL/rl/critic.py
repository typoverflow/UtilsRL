from typing import Dict, Optional, Any, Sequence, Union

import torch
import torch.nn as nn
import numpy as np

from UtilsRL.math.distributions import TanhNormal
from UtilsRL.net import MLP, EnsembleMLP
from torch.distributions import Categorical, Normal

class Critic(nn.Module):
    """A vanilla state-action critic, which outputs a single value Q(s, a) at a time. 
    
    :param backend: feature extraction backend of the critic. 
    :param input_dim: input dimension of the critic. Usually should be kept the same as `obs_shape + action_shape`.
    :param output_dim: output dimension of the critic, default to 1. 
    :param device: device to use, default to "cpu".
    :param hidden_dims: hidden dimensions of output layers, default to [].
    :param linear_layer: linear type of the output layers.
    """
    def __init__(self, 
                 backend: nn.Module, 
                 input_dim: int, 
                 output_dim: int=1, 
                 device: Union[str, int, torch.device] = "cpu", 
                 hidden_dims: Union[int, Sequence[int]] = [], 
                 ensemble_size: int=1, 
                 share_hidden_layer: Union[Sequence[bool], bool]=False
                 ):
        super().__init__()
        
        self.critic_type = "SingleCritic"
        self.backend = backend
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = device
        
        if isinstance(hidden_dims, int):
            hidden_dims = [hidden_dims]
        if ensemble_size == 1:
            self.output_layer = MLP(
                input_dim = input_dim, 
                output_dim = output_dim, 
                hidden_dims = hidden_dims, 
                device = device
            )
        elif isinstance(ensemble_size, int) and ensemble_size > 1:
            self.output_layer = EnsembleMLP(
                input_dim = input_dim, 
                output_dim = output_dim, 
                hidden_dims = hidden_dims, 
                device = device, 
                ensemble_size = ensemble_size, 
                share_hidden_layer = share_hidden_layer
            )
        else:
            raise ValueError(f"ensemble size should be int >= 1.")
        
    def forward(self, state: torch.Tensor, action: Optional[torch.Tensor]=None):
        """Just state-action compute Q(s, a). 

        :param state: state of the environment. 
        :param action: action of the agent. 
        """
        if action is not None:
            state = torch.cat([state, action], dim=-1)
        return self.output_layer(self.backend(state))
        