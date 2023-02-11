from typing import Dict, Optional, Any, Sequence, Union, Callable, Type
from collections import defaultdict
from functools import partial

import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal
import numpy as np

from UtilsRL.math.distributions import TanhNormal
from UtilsRL.net import MLP, EnsembleMLP

ModuleType = Type[nn.Module]

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
                 ensemble_size: int=1, 
                 device: Union[str, int, torch.device] = "cpu", 
                 *, 
                 hidden_dims: Sequence[int] = [], 
                 norm_layer: Optional[Union[ModuleType, Sequence[ModuleType]]] = None, 
                 activation: Optional[Union[ModuleType, Sequence[ModuleType]]] = nn.ReLU, 
                 dropout: Optional[Union[float, Sequence[float]]] = None, 
                 share_hidden_layer: Union[Sequence[bool], bool] = False, 
                 ):
        super().__init__()
        
        self.critic_type = "Critic"
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
                norm_layer = norm_layer, 
                activation = activation, 
                dropout = dropout, 
                device = device
            )
        elif isinstance(ensemble_size, int) and ensemble_size > 1:
            self.output_layer = EnsembleMLP(
                input_dim = input_dim, 
                output_dim = output_dim, 
                hidden_dims = hidden_dims, 
                norm_layer = norm_layer, 
                activation = activation, 
                dropout = dropout, 
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


class DoubleCritic(nn.Module):
    _reduce_fn_ = {
        "min": partial(torch.min, dim=0), 
        "max": partial(torch.max, dim=0), 
        "average": partial(torch.mean, dim=0)
    }
    def __init__(self, 
                 backend: nn.Module, 
                 input_dim: int, 
                 output_dim: int=1, 
                 critic_num: int=2, 
                 reduce: Union[str, Callable]="min", 
                 device: Union[str, int, torch.device]="cpu", 
                 *, 
                 hidden_dims: Union[int, Sequence[int]] = [], 
                 norm_layer: Optional[Union[ModuleType, Sequence[ModuleType]]] = None, 
                 activation: Optional[Union[ModuleType, Sequence[ModuleType]]] = nn.ReLU, 
                 dropout: Optional[Union[float, Sequence[float]]] = None, 
                 ):
        super().__init__()
        self.critic_type = "DoubleCritic"
        self.backend = backend
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = device
        self.critic_num = critic_num
        
        if isinstance(hidden_dims, int):
            hidden_dims = [hidden_dims]
        self.output_layer = EnsembleMLP(
            input_dim = input_dim, 
            output_dim = output_dim, 
            ensemble_size = critic_num, 
            hidden_dims = hidden_dims, 
            norm_layer = norm_layer, 
            activation = activation, 
            dropout = dropout, 
            share_hidden_layer = False, 
            device = device
        )
        
        if isinstance(reduce, str):
            self.reduce = self._reduce_fn_[reduce]
        else:
            self.reduce = reduce
        
    def forward(self, state: torch.Tensor, action: Optional[torch.Tensor]=None, reduce: bool=True):
        if action is not None:
            state = torch.cat([state, action], dim=-1)
        output = self.output_layer(self.backend(state))
        if reduce: 
            return self.reduce(output)
        else:
            return output
        
class C51DQN(nn.Module):
    def __init__(self, 
                 backend: nn.Module, 
                 input_dim: int, 
                 output_dim_adv: int, 
                 output_dim_value: int=1, 
                 num_atoms: int=51, 
                 v_min: float=0.0, 
                 v_max: float=200.0, 
                 device: Union[str, int, torch.device]="cpu", 
                 *, 
                 hidden_dims: Union[int, Sequence[int]] = [], 
                 norm_layer: Optional[Union[ModuleType, Sequence[ModuleType]]] = None, 
                 activation: Optional[Union[ModuleType, Sequence[ModuleType]]] = nn.ReLU, 
                 dropout: Optional[Union[float, Sequence[float]]] = None, 
                 ):
        super().__init__()
        self.actor_type = "C51Actor"
        self.backend = backend
        self.input_dim = input_dim
        self.output_dim_adv = output_dim_adv
        self.output_dim_value = output_dim_value
        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.hidden_dims = hidden_dims.copy()
        self.device = device
        
        self.register_buffer("support", torch.linspace(v_min, v_max, num_atoms))
        self.delta_z = (v_max - v_min) / (num_atoms - 1)
        
        if isinstance(hidden_dims, int):
            hidden_dims = [hidden_dims]
        self.adv_output_layer = MLP(
            input_dim = input_dim, 
            output_dim = output_dim_adv*num_atoms, 
            hidden_dims = hidden_dims, 
            norm_layer = norm_layer, 
            activation = activation, 
            dropout = dropout, 
            device = device, 
        )
        self.value_output_layer = MLP(
            input_dim = input_dim, 
            output_dim = output_dim_value*num_atoms, 
            hidden_dims = hidden_dims, 
            norm_layer = norm_layer, 
            activation = activation, 
            dropout = dropout, 
            device = device, 
        )
        
    def forward(self, state: torch.Tensor):
        dist = self.dist(state)
        q = torch.sum(dist * self.support, dim=2)
        return q
        
    def dist(self, state: torch.Tensor):
        o_backend = self.backend(state)
        o_adv = self.adv_output_layer(o_backend).view(-1, self.output_dim_adv, self.num_atoms)
        o_value = self.value_output_layer(o_backend).view(-1, self.output_dim_value, self.num_atoms)
        q_atoms = o_value + o_adv - o_adv.mean(dim=-2, keepdim=True)
        return nn.functional.softmax(q_atoms, dim=-1).clamp(min=1e-3)
        
            