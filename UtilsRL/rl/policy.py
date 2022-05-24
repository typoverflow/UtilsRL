from typing import Dict, Optional, Any, Sequence, Union

import torch
import torch.nn as nn
import numpy as np

from UtilsRL.math.distributions import TanhNormal
from torch.distributions import Categorical

from abc import ABC, abstractmethod

class BasePolicy(nn.Module):
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

class SquashedDeterministicPolicy(BasePolicy):
    def __init__(self,
                 backend: nn.Module, 
                 input_dim: int, 
                 output_dim: int, 
                 device: Union[str, int, torch.device]="cpu"
                 ):
        super().__init__()
        
        self.policy_type = "SqushedDeterministicPolicy"
        self.backend = backend
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = device
        
        self.output_layer = nn.Linear(input_dim, output_dim)
        
    def forward(self, state: torch.Tensor):
        return self.output_layer(self.backend(state))
        
    def sample(self, state: torch.Tensor):
        action_prev_tanh = self.forward(state)
        return torch.tanh(action_prev_tanh)
    
    def evaluate(self, state: torch.Tensor, action: torch.Tensor):
        raise NotImplementedError("Evaluation shouldn't be called for SquashedDeterministicPolicy.")
            
    
class SquashedGaussianPolicy(BasePolicy):
    def __init__(self, 
                 backend: nn.Module, 
                 input_dim: int, 
                 output_dim: int, 
                 device: Union[str, int, torch.device]="cpu",
                 reparameterize: bool = True, 
                 conditioned_logstd: bool = True, 
                 fix_logstd: Optional[float] = None, 
                 logstd_min: int = -20, 
                 logstd_max: int = 2, 
                 ):
        super().__init__()
        
        self.policy_type = "SquashedGaussianPolicy"
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.reparameterize = reparameterize
        
        self.device = device
        self.backend = backend.to(device)
        
        # check std
        if fix_logstd is not None:
            self._logstd_is_layer = False
            self.logstd = nn.Parameter(torch.tensor(fix_logstd, dtype=torch.float), requires_grad=False).to(device)
        elif not conditioned_logstd:
            self._logstd_is_layer = False
            self.logstd = -0.5 * torch.ones([self.output_dim], dtype=torch.float)
            self.logstd = nn.Parameter(self.logstd, requires_grad=True).to(device)
        else:
            self._logstd_is_layer = True
            self.output_dim *= 2
        self.output_layer = nn.Linear(input_dim, self.output_dim).to(device)
        
        self.logstd_min = nn.Parameter(torch.tensor(logstd_min, dtype=torch.float), requires_grad=False).to(device)
        self.logstd_max = nn.Parameter(torch.tensor(logstd_max, dtype=torch.float), requires_grad=False).to(device)
        
    def forward(self, state: torch.Tensor):
        out = self.output_layer(self.backend(state))
        if self._logstd_is_layer:
            mean, logstd = torch.split(out, self.output_dim // 2, dim=-1)
        else:
            mean = out
            logstd = self.logstd.broadcast_to(mean.shape)
        logstd = torch.clamp(logstd, min=self.logstd_min, max=self.logstd_max)
        return mean, logstd
        
    def sample(self, state: torch.Tensor, deterministic: bool=False, return_mean_logstd=False):
        mean, logstd = self.forward(state)
        dist = TanhNormal(mean, logstd.exp())
        if deterministic:
            action, logprob = dist.tanh_mean, None
        elif self.reparameterize:
            action = dist.rsample()
            logprob = dist.log_prob(action).sum(-1, keepdim=True)
        else:
            action = dist.sample()
            logprob = dist.log_prob(action).sum(-1, keepdim=True)
        
        return [action, logprob] if not return_mean_logstd else [action, logprob, mean, logstd]
    
    def evaluate(self, state: torch.Tensor, action: torch.Tensor):
        mean, logstd = self.forward(state)
        logprob = TanhNormal(mean, logstd.exp()).log_prob(action).sum(-1, keepdim=True)
        return logprob
    

class CategoricalPolicy(BasePolicy):
    def __init__(self, 
                 backend: nn.Module, 
                 input_dim: int, 
                 output_dim: int, 
                 device: Union[str, int, torch.device]="cpu", 
                 ):
        super().__init__()
        
        self.policy_type = "CategoricalPolicy"
        self.backend = backend
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = device
        
        self.output_layer = nn.Linear(input_dim, output_dim)
        
    def forward(self, state: torch.Tensor):
        out = self.output_layer(self.backend(state))
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
        
        return [action, logprob] if not return_probs else [action, logprob, probs]
    
    def evaluate(self, state: torch.Tensor, action: torch.Tensor):
        if len(action.shape) == 2:
            action = action.view(-1)
        probs = self.forward(state)
        return Categorical(probs=probs).log_prob(action).unsqueeze(-1)
            
            

        
        