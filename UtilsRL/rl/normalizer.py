from typing import Union, Sequence, Any, Optional, Dict
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import numpy as np

class RunningMeanStd(object):
    def __init__(self, epsion: float=1e-4, shape=(), device=Union[str, int, torch.device]):
        super().__init__()
        self.mean = torch.zeros(shape, dtype=torch.float32, device=device, requires_grad=False)
        self.var = torch.ones(shape, dtype=torch.float32, device=device, requires_grad=False)
        self.count = epsion
        
    def update(self, data: torch.Tensor):
        num_shape = len(data.shape)
        batch_mean = torch.mean(data, dim=[_ for _ in range(num_shape-1)])
        batch_var = torch.var(data, dim=[_ for _ in range(num_shape-1)])
        batch_count = np.prod(data.shape[:-1])
        self._update_from_moments(batch_mean, batch_var, batch_count)
        
    def _update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + np.square(delta) * self.count * batch_count / (self.count + batch_count)
        new_var = m_2 / (self.count + batch_count)
        
        self.mean = new_mean
        self.var = new_var
        self.count += batch_count


class BaseNormalizer(ABC):
    @abstractmethod
    def transform(self, x: torch.Tensor, inverse: bool = False):
        raise NotImplementedError
    
    @abstractmethod
    def update(self, x: torch.Tensor):
        raise NotImplementedError
    
    def forward(self, *args, **kwargs):
        return self.transform(*args, **kwargs)


class DummyNormalizer(BaseNormalizer, nn.Module):
    def __init__(self, **kwargs):
        BaseNormalizer.__init__(self)
        nn.Module.__init__(self)
        
    def transform(self, x: torch.Tensor, inverse: bool = False):
        return x
    
    def update(self, x: torch.Tensor):
        pass
      

class RunningNormalizer(BaseNormalizer, nn.Module):
    def __init__(self, eps=1e-6, **kwargs):
        BaseNormalizer.__init__(self)
        nn.Module.__init__(self)
        self._initialized = nn.Parameter(torch.tensor(False), requires_grad=False)
        self.eps = eps
        if "shape" in kwargs:
            self._initialize(kwargs["shape"])
        
        self.count = nn.Parameter(torch.tensor(0), requires_grad=False)
        
    def _initialize(self, shape: Union[Sequence[int], int]):
        if shape is None:
            raise ValueError("shape must be specified for Running Nomralizer.")
        if isinstance(shape, int):
            shape = [shape]
        
        self.register_parameter("mean", nn.Parameter(torch.zeros(shape, dtype=torch.float32), requires_grad=False))
        self.register_parameter("var", nn.Parameter(torch.ones(shape, dtype=torch.float32), requires_grad=False))
        self._initialized.data = torch.tensor(True)
        
    def transform(self, x: torch.Tensor, inverse: bool = False):
        if not self._initialized:
            self._initialize(x.shape[1:])
        if inverse:
            return x * torch.sqrt(self.var+self.eps) + self.mean
        return (x-self.mean) / torch.sqrt(self.var + self.eps)

    def update(self, data: torch.Tensor):
        if not self._initialized:
            self._initialize(data.shape[1:])
        num_shape = len(data.shape)
        batch_mean = torch.mean(data, dim=0).detach().clone()
        batch_var = torch.var(data, dim=0).detach().clone()
        batch_count = data.shape[0]
        device = batch_mean.device
        
        if self.mean.shape != batch_mean.shape:
            raise ValueError(f"Expecting tensors of shape (B, {self.mean.shape}), found (B, {batch_mean.shape}).")
        old_mean = self.mean.data.to(device)
        old_var = self.var.data.to(device)
        count = self.count.data.to(device)
        
        delta = batch_mean - old_mean
        tot_count = count + batch_count
        new_mean = old_mean + delta * batch_count / tot_count
        m_a = old_var * count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + torch.square(delta) * count * batch_count / (count + batch_count)
        new_var = m_2 / (tot_count)
        
        self.mean.data = new_mean
        self.var.data = new_var
        self.count.data = tot_count
        
        
class StaticNormalizer(BaseNormalizer, nn.Module):
    def __init__(self, eps=1e-6, **kwargs):
        BaseNormalizer.__init__(self)
        nn.Module.__init__(self)
        self._initialized = nn.Parameter(torch.tensor(False), requires_grad=False)
        self.eps = eps
        if "mean" in kwargs:
            if "var" in kwargs:
                self._initialize(mean=kwargs["mean"], std=None, var=kwargs["var"])
            elif "std" in kwargs:
                self._initialize(mean=kwargs["mean"], std=kwargs["std"], var=None)
            else:
                raise KeyError("mean and var must be specified at the same time.")
        
    def _initialize(self,
                    mean: Optional[torch.Tensor], 
                    std: Optional[torch.Tensor], 
                    var: Optional[torch.Tensor]
                    ):
        if mean is None:
            raise ValueError("Mean must be provided when initializing StaticNormalizer!")
        if std is None and var is None:
            raise ValueError("Either std or var must be provided when initializing StaticNormalizer!")
        
        if hasattr(self, "mean"):
            self.mean.data = mean.detach().clone()
        else:
            self.register_parameter("mean", nn.Parameter(mean.detach().clone(), requires_grad=False))
        if std is not None:
            if hasattr(self, "std"):
                self.std.data = std.detach().clone()
            else:
                self.register_parameter("std", nn.Parameter(std.detach().clone(), requires_grad=False))
        elif var is not None:
            if hasattr(self, "std"):
                self.std.data = torch.sqrt(var + self.eps).detach().clone()
            else:
                self.register_parameter("std", nn.Parameter(torch.sqrt(var + self.eps).detach().clone(), requires_grad=False))
            
        self._initialized.data = torch.tensor(True).to(mean.device)
        
    def transform(self, x: torch.Tensor, inverse: bool=False):
        if not self._initialized:
            raise ValueError("Static Normalizers must be initialized before transforming.")
        if inverse:
            return x * self.std + self.mean
        return (x-self.mean) / (self.std)
    
    def update(self, data: torch.Tensor):
        num_shape = len(data.shape)
        batch_mean = torch.mean(data, dim=0)
        batch_std = torch.std(data, dim=0)
        
        self._initialize(mean=batch_mean, std=batch_std, var=None)
           
        
class MinMaxNormalizer(BaseNormalizer, nn.Module):
    def __init__(self, eps=1e-6, **kwargs):
        BaseNormalizer.__init__(self)
        nn.Module.__init__(self)
        self._initialized = nn.Parameter(torch.tensor(False), requires_grad=False)
        self.eps = eps
        if "min" in kwargs or "max" in kwargs:
            self._initialize(min=kwargs.get("min", None), max=kwargs.get("max", None))
        
    def _initialize(self, 
                    min: Optional[torch.Tensor], 
                    max: Optional[torch.Tensor]
                    ):
        if min is None or max is None:
            raise ValueError("Both min and max must be provided when initializing MinMaxNormalizer!")
        if hasattr(self, "min"):
            self.min.data = min.detach().clone()
        else:
            self.register_parameter("min", min.detach().clone())
        if hasattr(self, "max"):
            self.max.data = max.detach().clone()
        else:
            self.register_parameter("max", max.detach().clone())
        
        self._initialized.data = torch.tensor(True).to(min.device)
        
    def transform(self, x: torch.Tensor, inverse: bool=False):
        if not self._initialized:
            raise ValueError("MinMax Normalizers must be initialized before transforming.")
        if inverse:
            return x*(self.max - self.min) + self.min
        return (x-self.min) / (self.max - self.min+1e-6)
    
    def update(self, data: torch.Tensor):
        num_shape = len(data.shape)
        batch_min = torch.min(data, dim=0)
        batch_max = torch.max(data, dim=0)
        
        self._initialize(min=batch_min, max=batch_max)
        