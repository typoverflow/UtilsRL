from typing import Dict, Optional, Any, Sequence, Union

import torch
import torch.nn as nn
import numpy as np

from UtilsRL.math.distributions import TanhNormal
from UtilsRL.rl.net import MLP
from torch.distributions import Categorical, Normal

class SingleCritic(nn.Module):
    def __init__(self, 
                 backend: nn.Module, 
                 input_dim: int, 
                 output_dim: int=1, 
                 device: Union[str, int, torch.device] = "cpu", 
                 hidden_dims: Union[int, Sequence[int]] = [], 
                 linear_layer: nn.Module=nn.Linear
                 ):
        super().__init__()
        
        self.critic_type = "SingleCritic"
        self.backend = backend
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = device
        
        if isinstance(hidden_dims, int):
            hidden_dims = [hidden_dims]
        self.output_layer = MLP(
            input_dim = input_dim, 
            output_dim = output_dim, 
            hidden_dims = hidden_dims, 
            device = device, 
            linear_layer=linear_layer
        )
        
    def forward(self, state: torch.Tensor, action: Optional[torch.Tensor]=None):
        if action is not None:
            state = torch.cat([state, action], dim=-1)
        return self.output_layer(self.backend(state))
        