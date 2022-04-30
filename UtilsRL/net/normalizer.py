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

    @abstractmethod
    def state_dict(self): 
        raise NotImplementedError
    
    @abstractmethod
    def load_state_dict(self, state_dict: Dict):
        raise NotImplementedError
    
    def __call__(self, x: torch.Tensor, inverse: bool = False):
        return self.transform(x, inverse)        

class RunningNormalizer(BaseNormalizer):
    def __init__(self, eps=1e-6, device: Union[str, int, torch.device]="cpu", **kwargs):
        super().__init__()
        self._initialized, self.mean, self.var = False, None, None
        self.eps = eps
        self.device = device
        if "shape" in kwargs:
            self._initialize(kwargs["shape"])
        
        self.count = 0
        
    def _initialize(self, shape: Union[Sequence[int], int]):
        if shape is None:
            raise ValueError("shape must be specified for Running Nomralizer.")
        if isinstance(shape, int):
            shape = [shape]
        
        self.mean = torch.zeros(shape, dtype=torch.float32, device=self.device, requires_grad=False)
        self.var = torch.ones(shape, dtype=torch.float32, device=self.device, requires_grad=False)
        self._initialized = True
        
    def transform(self, x: torch.Tensor, inverse: bool = False):
        if not self._initilized:
            self._initialize(x.shape[-1])
        if inverse:
            return x * torch.sqrt(self.var+self.eps) + self.mean
        return (x-self.mean) / self.sqrt(self.var + self.eps)

    def update(self, data: torch.Tensor):
        num_shape = len(data.shape)
        batch_mean = torch.mean(data, dim=[_ for _ in range(num_shape-1)]).detach().clone().to(self.device)
        batch_var = torch.var(data, dim=[_ for _ in range(num_shape-1)]).detach().clone().to(self.device)
        batch_count = np.prod(data.shape[:-1])

        delta = batch_mean - self.mean
        tot_count = self.count + batch_count + 1e-4
        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + np.square(delta) * self.count * batch_count / (self.count + batch_count)
        new_var = m_2 / (tot_count)
        
        self.mean = new_mean
        self.var = new_var
        self.count += batch_count
    
    def state_dict(self):
        return {
            "mean": self.mean, 
            "var": self.var, 
            "count": self.count, 
            "_initialized": self._initialized
        }
    
    def load_state_dict(self, state_dict: Dict):
        self.mean = state_dict["mean"]
        self.var = state_dict["var"]
        self.count = state_dict["count"]
        self._initialized = state_dict["_initialized"]
        
        
class StaticNormalizer(BaseNormalizer):
    def __init__(self, eps=1e-6, device: Union[str, int, torch.device]="cpu", **kwargs):
        super().__init__()
        self._initialized, self.mean, self.std = False, None, None
        self.eps = eps
        self.device = device
        if "mean" in kwargs:
            if "var" in kwargs:
                self._initialize(mean=kwargs["mean"], std=None, var=kwargs["var"])
            elif "std" in kwargs:
                self._initialize(mean=kwargs["mean"], std=kwargs["std"], var=None)
            else:
                raise KeyError("mean and var must be specified at the same time.")
        
        self.count = 0
        
    def _initialize(self,
                    mean: Optional[torch.Tensor], 
                    std: Optional[torch.Tensor], 
                    var: Optional[torch.Tensor]
                    ):
        self.mean = mean.detach().clone().to(self.device)
        if std is not None:
            self.std = std.detach().clone().to(self.device)
        elif var is not None:
            self.std = torch.sqrt(var + self.eps).detach().clone().to(self.device)
            
        self._initialized = True
        
    def transform(self, x: torch.Tensor, inverse: bool=False):
        if not self._initialized:
            raise ValueError("Static Normalizers must be initialized before transforming.")
        if inverse:
            return x * self.std + self.mean
        return (x-self.mean) / (self.std)
    
    def update(self, data: torch.Tensor):
        num_shape = len(data.shape)
        batch_mean = torch.mean(data, dim=[_ for _ in range(num_shape-1)])
        batch_std = torch.std(data, dim=[_ for _ in range(num_shape-1)])
        batch_count = np.prod(data.shape[:-1])
        
        self._initialize(mean=batch_mean, std=batch_std, var=None)
        self.count += batch_count
        
    def state_dict(self):
        return {
            "mean": self.mean, 
            "std": self.std, 
            "count": self.count,
            "_initialized": self._initialized
        }
        
    def load_state_dict(self, state_dict: Dict):
        self.mean = state_dict["mean"]
        self.std = state_dict["std"]
        self.count = state_dict["count"]
        self._initialized = state_dict["_initialized"]
        
class MinMaxNormalizer(BaseNormalizer):
    def __init__(self, eps=1e-6, device: Union[str, int, torch.device]="cpu", **kwargs):
        super().__init__()
        self._initialized = False
        self.eps = eps
        self.device = device
        if "min" in kwargs and "max" in kwargs:
            self._initialize(min=kwargs["min"], max=kwargs["max"])
        
        self.count = 0
        
    def _initialize(self, 
                    min: torch.Tensor, 
                    max: torch.Tensor
                    ):
        self.min = min.detach().clone().to(self.device)
        self.max = max.detach().clone().to(self.device)
        self._initialized = True
        
    def transform(self, x: torch.Tensor, inverse: bool=False):
        if not self._initialized:
            raise ValueError("MinMax Normalizers must be initialized before transforming.")
        if inverse:
            return x*(self.max - self.min) + self.min
        return (x-self.min) / (self.max - self.min+1e-6)
    
    def update(self, data: torch.Tensor):
        num_shape = len(data.shape)
        batch_min = torch.min(data, dim=[_ for _ in range(num_shape-1)])
        batch_max = torch.max(data, dim=[_ for _ in range(num_shape-1)])
        batch_count = np.prod(data.shape[:-1])
        
        self._initialize(min=batch_min, max=batch_max)
        self.count += batch_count
        
    def state_dict(self):
        return {
            "min": self.min, 
            "max": self.max, 
            "count": self.count, 
            "_initialized": self._initialized
        }
    
    def load_state_dict(self, state_dict: Dict):
        self.min = state_dict["min"]
        self.max = state_dict["max"]
        self.count = state_dict["count"]
        self._initialized = state_dict["_initialized"]
         