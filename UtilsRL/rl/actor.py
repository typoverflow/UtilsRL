from typing import Dict, Optional, Any, Sequence, Union

import torch
import torch.nn as nn
import numpy as np

from UtilsRL.math.distributions import TanhNormal
from UtilsRL.rl.net import MLP
from torch.distributions import Categorical, Normal

from abc import ABC, abstractmethod

class BaseActor(nn.Module):
    def __init__(self):
        super().__init__()
        
    @abstractmethod
    def forward(self, state: torch.Tensor, *args, **kwargs):
        raise NotImplementedError
    
    @abstractmethod
    def sample(self, state: torch.Tensor, *args, **kwargs):
        raise NotImplementedError
    
    @abstractmethod
    def evaluate(self, state, action, *args, **kwargs):
        raise NotImplementedError

class DeterministicActor(BaseActor):
    def __init__(self, 
                 backend: nn.Module, 
                 input_dim: int, 
                 output_dim: int, 
                 device: Union[str, int, torch.device]="cpu", 
                 hidden_dims: Union[int, Sequence[int]]=[],
                 linear_layer: nn.Module=nn.Linear
                 ):
        super().__init__()
        self.actor_type = "DeterministicActor"
        self.backend = backend
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims.copy()
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
    
    def forward(self, input: torch.Tensor):
        return self.output_layer(self.backend(input))
    
    def sample(self, input: torch.Tensor):
        return self(input)
    
    def evaluate(self, state: torch.Tensor, action: torch.Tensor):
        raise NotImplementedError("Evaluation shouldn't be called for SquashedDeterministicActor.")
        
        
class SquashedDeterministicActor(DeterministicActor):
    def __init__(self,
                 backend: nn.Module, 
                 input_dim: int, 
                 output_dim: int, 
                 device: Union[str, int, torch.device]="cpu", 
                 hidden_dims: Union[int, Sequence[int]]=[],
                 linear_layer: nn.Module=nn.Linear, 
                 ):
        super().__init__(backend, input_dim, output_dim, device, hidden_dims, linear_layer)
        self.actor_type = "SqushedDeterministicActor"
        
    def sample(self, input: torch.Tensor):
        action_prev_tanh = super().forward(input)
        return torch.tanh(action_prev_tanh)
            

class ClippedDeterministicActor(DeterministicActor):
    def __init__(self, 
                 backend: nn.Module, 
                 input_dim: int, 
                 output_dim: int, 
                 device: Union[str, int, torch.device]="cpu", 
                 hidden_dims: Union[int, Sequence[int]]=[], 
                 linear_layer: nn.Module=nn.Linear
                 ):
        super().__init__(backend, input_dim, output_dim, device, hidden_dims, linear_layer)
        self.actor_type = "ClippedDeterministicActor"
        
    def sample(self, input: torch.Tensor):
        action = super().forward(input)
        return torch.clip(action, min=-1, max=1)
    
    
class GaussianActor(BaseActor):
    def __init__(self, 
                 backend: nn.Module, 
                 input_dim: int, 
                 output_dim: int, 
                 device: Union[str, int, torch.device]="cpu", 
                 reparameterize: bool=True, 
                 conditioned_logstd: bool=True, 
                 fix_logstd: Optional[float]=None, 
                 hidden_dims: Union[int, Sequence[int]]=[],
                 linear_layer: nn.Module=nn.Linear, 
                 logstd_min: int = -20, 
                 logstd_max: int = 2,
                 ):
        super().__init__()
        
        self.actor_type = "GaussianActor"
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.reparameterize = reparameterize
        self.device = device
        self.backend = backend
        
        if fix_logstd is not None:
            self._logstd_is_layer = False
            self.logstd = nn.Parameter(torch.tensor(fix_logstd, dtype=torch.float), requires_grad=False)
        elif not conditioned_logstd:
            self._logstd_is_layer = False
            self.logstd = nn.Parameter(-0.5 * torch.ones([self.output_dim], dtype=torch.float), requires_grad=True)
        else:
            self._logstd_is_layer = True
            self.output_dim = output_dim = 2*output_dim

        if isinstance(hidden_dims, int):
            hidden_dims = [hidden_dims]
        self.output_layer = MLP(
            input_dim = input_dim,
            output_dim = output_dim, 
            hidden_dims = hidden_dims, 
            device = device, 
            linear_layer=linear_layer
        )
        
        self.logstd_min = nn.Parameter(torch.tensor(logstd_min, dtype=torch.float), requires_grad=False)
        self.logstd_max = nn.Parameter(torch.tensor(logstd_max, dtype=torch.float), requires_grad=False)
        
    def forward(self, input: torch.Tensor):
        out = self.output_layer(self.backend(input))
        if self._logstd_is_layer:
            mean, logstd = torch.split(out, self.output_dim // 2, dim=-1)
        else:
            mean = out
            logstd = self.logstd.broadcast_to(mean.shape)
        logstd = torch.clip(logstd, min=self.logstd_min, max=self.logstd_max)
        return mean, logstd
    
    def sample(self, input: torch.Tensor, deterministic: bool=False, return_mean_logstd: bool=False):
        mean, logstd = self(input)
        dist = Normal(mean, logstd.exp())
        if deterministic:
            action, logprob = dist.mean, None
        elif self.reparameterize:
            action = dist.rsample()
            logprob = dist.log_prob(action).sum(-1, keepdim=True)
        else:
            action = dist.sample()
            logprob = dist.log_prob(action).sum(-1, keepdim=True)
        
        info = {"mean": mean, "logstd": logstd} if return_mean_logstd else {}
        return action, logprob, info
    
    def evaluate(self, state: torch.Tensor, action: torch.Tensor):
        mean, logstd = self(state)
        dist = Normal(mean, logstd.exp())
        return dist.log_prob(action).sum(-1, keepdim=True), dist.entropy().sum(-1, keepdim=True)


class SquashedGaussianActor(GaussianActor):
    def __init__(self, 
                 backend: nn.Module, 
                 input_dim: int, 
                 output_dim: int, 
                 device: Union[str, int, torch.device]="cpu",
                 reparameterize: bool = True, 
                 conditioned_logstd: bool = True, 
                 fix_logstd: Optional[float] = None, 
                 hidden_dims: Union[int, Sequence[int]] = [],
                 linear_layer: nn.Module=nn.Linear,
                 logstd_min: int = -20, 
                 logstd_max: int = 2, 
                 ):
        super().__init__(backend, input_dim, output_dim, device, reparameterize, conditioned_logstd, fix_logstd, hidden_dims, linear_layer, logstd_min, logstd_max)
        self.actor_type = "SquashedGaussianActor"
        
    def sample(self, input: torch.Tensor, deterministic: bool=False, return_mean_logstd=False):
        mean, logstd = self.forward(input)
        dist = TanhNormal(mean, logstd.exp())
        if deterministic:
            action, logprob = dist.tanh_mean, None
        elif self.reparameterize:
            action = dist.rsample()
            logprob = dist.log_prob(action).sum(-1, keepdim=True)
        else:
            action = dist.sample()
            logprob = dist.log_prob(action).sum(-1, keepdim=True)
        
        info = {"mean": mean, "logstd": logstd} if return_mean_logstd else {}
        return action, logprob, info
    
    def evaluate(self, state: torch.Tensor, action: torch.Tensor):
        mean, logstd = self(state)
        dist = TanhNormal(mean, logstd.exp())
        return dist.log_prob(action).sum(-1, keepdim=True), dist.entropy().sum(-1, keepdim=True)
    
    
class ClippedGaussianActor(GaussianActor):
    def __init__(self, 
                 backend: nn.Module, 
                 input_dim: int, 
                 output_dim: int, 
                 device: Union[str, int, torch.device]="cpu",
                 reparameterize: bool = True, 
                 conditioned_logstd: bool = True, 
                 fix_logstd: Optional[float] = None, 
                 hidden_dims: Union[int, Sequence[int]] = [],
                 linear_layer: nn.Module=nn.Linear,
                 logstd_min: int = -20, 
                 logstd_max: int = 2, 
                 ):
        super().__init__(backend, input_dim, output_dim, device, reparameterize, conditioned_logstd, fix_logstd, hidden_dims, linear_layer, logstd_min, logstd_max)
        self.actor_type = "ClippedGaussianActor"
        
    def sample(self, input: torch.Tensor, deterministic: bool=False, return_mean_logstd=False):
        action, logprob, info = super().sample(input, deterministic, return_mean_logstd)
        return torch.clip(action, min=-1, max=1), logprob, info

class CategoricalActor(BaseActor):
    def __init__(self, 
                 backend: nn.Module, 
                 input_dim: int, 
                 output_dim: int, 
                 device: Union[str, int, torch.device]="cpu", 
                 hidden_dims: Union[int, Sequence[int]] = [],
                 linear_layer: nn.Module=nn.Linear,
                 ):
        super().__init__()
        
        self.actor_type = "CategoricalActor"
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
        
    def forward(self, input: torch.Tensor):
        out = self.output_layer(self.backend(input))
        return torch.softmax(out, dim=-1)
    
    def sample(self, state: torch.Tensor, deterministic: bool=False, return_probs: bool=False):
        probs = self.forward(state)
        if deterministic: 
            action = torch.argmax(probs, dim=-1, keepdim=True)
            logprob = torch.log(torch.max(probs, dim=-1, keepdim=True)[0] + 1e-6)
        else:
            dist = Categorical(probs=probs)
            action = dist.sample()
            logprob = dist.log_prob(action).unsqueeze(-1)
            action = action.unsqueeze(-1)
        
        info = {"probs": probs} if return_probs else {}
        return action, logprob, info
    
    def evaluate(self, state: torch.Tensor, action: torch.Tensor):
        if len(action.shape) == 2:
            action = action.view(-1)
        probs = self.forward(state)
        dist = Categorical(probs=probs)
        return dist.log_prob(action).unsqueeze(-1), dist.entropy().unsqueeze(-1)
            
            

        
        