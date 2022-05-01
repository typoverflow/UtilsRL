from typing import Optional, Any, Dict, Sequence, Union

import torch
import math

from abc import ABC

class TanhNormal(torch.distributions.Distributions):
    def __init__(self, 
                 mean: torch.Tensor, 
                 std: torch.Tensor, 
                 epsilon: float=1e-6
                 ):
        super().__init__()
        self.mean = mean
        self.std = std
        self.epsilon = epsilon
        self.normal = torch.distributions.Normal(mean, std)
        
    def log_prob(self, 
                 value: torch.Tensor, 
                 pre_tanh_value: bool=None
                 ):
        if not pre_tanh_value:
            clip_value = torch.clamp(value, -1.0+1e-6, 1.0-1e-6)
            pre_tanh_value = 0.5 * (clip_value.log1p() - (-clip_value).log1p())
        return self.normal.log_prob(pre_tanh_value) - torch.log(1 - value.pow(2) + self.epsilon)
    
    def sample(self, sample_shape: Union[Sequence[int], int]=torch.Size([])):
        z = self.normal.sample(sample_shape)
        return torch.tanh(z)
    
    def rsample(self, sample_shape: Union[Sequence[int], int]=torch.Size([])):
        z = self.normal.rsample(sample_shape)
        return torch.tanh(z)
    
    def entropy(self):
        self.normal.entropy()
        
    @property
    def mean(self):
        return self.mean
    
    @property
    def tanh_mean(self):
        return torch.tanh(self.mean)
    
    