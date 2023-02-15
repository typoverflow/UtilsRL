from typing import Union, Sequence, Any, Optional, Dict
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import numpy as np

class RunningMeanStd(object):
    def __init__(self, epsion: float=1e-4, shape=(), device=Union[str, int, torch.device]):
        super().__init__()
        self.mean = torch.zeros(shape, device=device, requires_grad=False)
        self.var = torch.ones(shape, device=device, requires_grad=False)
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
    def transform(self, x: torch.Tensor, inverse: bool = False) -> torch.Tensor:
        """Transform the input data. 

        Parameters
        ----------
        x : The input data, should be torch.Tensor. 
        inverse : If True, reverse the transformation.
        
        Returns
        -------
        torch.Tensor :  The transformed version of the input. 
        """
        raise NotImplementedError
    
    @abstractmethod
    def update(self, x: torch.Tensor) -> None:
        """Use the input data to update the normalizer statistics. 

        Parameters
        ----------
        x : The data to update the normalizer statistics.
        """
        raise NotImplementedError
    
    def forward(self, *args, **kwargs):
        return self.transform(*args, **kwargs)


class DummyNormalizer(BaseNormalizer, nn.Module):
    """A dummy normalizer for debug use. """
    
    def __init__(self, **kwargs):
        BaseNormalizer.__init__(self)
        nn.Module.__init__(self)
        
    def transform(self, x: torch.Tensor, inverse: bool = False):
        return x
    
    def update(self, x: torch.Tensor):
        pass
      

class RunningNormalizer(BaseNormalizer, nn.Module):
    """
    A normalizer which normalizes data by data = (data - running_mean) / running_std. 
    
    Parameters
    ----------
    shape :  The shape of the data. If None, the data shape will be automatically inferred during the first call of `transform`. Default to None. 
    """
    def __init__(self, shape=None, eps=1e-8) -> None:
        BaseNormalizer.__init__(self)
        nn.Module.__init__(self)
        self.register_buffer("_initialized", torch.tensor(False))
        self.eps = eps
        if shape: 
            self._initialize(shape)
        
        self.register_buffer("count", torch.tensor(0))
        
    def _initialize(self, shape: Union[Sequence[int], int]) -> None:
        if shape is None:
            raise ValueError("shape must be specified for Running Nomralizer.")
        if isinstance(shape, int):
            shape = [shape]
        
        self.register_buffer("mean", torch.zeros(shape))
        self.register_buffer("var", torch.ones(shape))
        self._initialized.data = torch.tensor(True)
        
    def transform(self, x: torch.Tensor, inverse: bool = False) -> torch.Tensor:
        """Transform the input data by data = (data - running_mean) / running_std. 

        Notes
        -----
        All dimensions except for the dim=0 will be transformed. 

        Parameters
        ----------
        x :  The input data, should be torch.Tensor.
        inverse :  If True, inverse the transformation defined by the normalizer.
        
        Returns
        -------
        torch.Tensor :  The transformed version of the input. 
        """
        if not self._initialized:
            self._initialize(x.shape[1:])
        if inverse:
            return x * torch.sqrt(self.var+self.eps) + self.mean
        return (x-self.mean) / torch.sqrt(self.var + self.eps)

    def update(self, x: torch.Tensor) -> None:
        """Use the input data to update the normalizer statistics. 

        Parameters
        ----------
        x : The data to update the normalizer statistics.
        """
        if not self._initialized:
            self._initialize(x.shape[1:])
        num_shape = len(x.shape)
        batch_mean = torch.mean(x, dim=0).detach().clone()
        batch_var = torch.var(x, dim=0).detach().clone()
        batch_count = x.shape[0]
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
    """
    A normalizer which normalizes data by data = (data - static_mean) / static_std. 
    
    Notes
    -----
    To initialize the mean and std/var for this normalizer, there are three options: 
    1) __init__(mean=xxx, std=xxx) or __init__(mean=xxx, var=xxx). Note that `mean` and at least one of `std` and `var` 
        must be provided when construcing. 
    2) Leave `mean/std/var` as None, and explicitly call `_initilize` method to post initialize the normalizer. The requirement 
        is the same as in 1). 
    3) Call `update` before the first call to `transform`, and the static mean and static std will be inferred from the data. 
    
    Parameters
    ----------
    mean :  The static mean of the normalizer. Default to None. 
    std :  The static std of the normalizer. Default to None.
    var :  The static var of the normalizer. Default to None.
    """
    def __init__(self, mean=None, std=None, var=None, eps=1e-8) -> None:
        BaseNormalizer.__init__(self)
        nn.Module.__init__(self)
        self.register_buffer("_initialized", torch.tensor(False))
        self.eps = eps
        if mean is not None:
            if var is not None:
                self._initialize(mean=mean, std=None, var=var)
            elif std is not None:
                self._initialize(mean=mean, std=std, var=None)
            else:
                raise KeyError("mean and var must be specified at the same time.")
        
    def _initialize(self,mean: Optional[torch.Tensor], std: Optional[torch.Tensor], var: Optional[torch.Tensor]) -> None:
        if mean is None:
            raise ValueError("Mean must be provided when initializing StaticNormalizer!")
        if std is None and var is None:
            raise ValueError("Either std or var must be provided when initializing StaticNormalizer!")
        
        if hasattr(self, "mean"):
            self.mean.data = mean.detach().clone()
        else:
            self.register_buffer("mean", mean.detach().clone())
        if std is not None:
            if hasattr(self, "std"):
                self.std.data = std.detach().clone()
            else:
                self.register_buffer("std", std.detach().clone())
        elif var is not None:
            if hasattr(self, "std"):
                self.std.data = torch.sqrt(var + self.eps).detach().clone()
            else:
                self.register_buffer("std", torch.sqrt(var + self.eps).detach().clone())
            
        self._initialized.data = torch.tensor(True).to(mean.device)
        
    def transform(self, x: torch.Tensor, inverse: bool=False) -> torch.Tensor:
        """
        Transform the input data by data = (data - static_mean) / static_std. 

        Notes
        -----
        All dimensions except for the dim=0 will be transformed. 

        Parameters
        ----------
        x :  The input data, should be torch.Tensor.
        inverse :  If True, inverse the transformation defined by the normalizer.
        
        Returns
        -------
        torch.Tensor :  The transformed version of the input. 
        """
        if not self._initialized:
            raise ValueError("Static Normalizers must be initialized before transforming.")
        if inverse:
            return x * self.std + self.mean
        return (x-self.mean) / (self.std)
    
    def update(self, data: torch.Tensor) -> None:
        """Use the input data to update the normalizer statistics. 

        Parameters
        ----------
        x : The data to update the normalizer statistics.
        """
        num_shape = len(data.shape)
        batch_mean = torch.mean(data, dim=0)
        batch_std = torch.std(data, dim=0)
        
        self._initialize(mean=batch_mean, std=batch_std, var=None)
           
        
class MinMaxNormalizer(BaseNormalizer, nn.Module):
    """
    A normalizer which normalizes data by data = (data - min) / (max - min). 
    
    Notes
    -----
    To initialize the mean and min/var for this normalizer, there are three options: 
    1) __init__(min=xxx, max=xxx). Note that both `min` and `max` should be provided at the same time. 
    2) Leave `min/max` as None, and explicitly call `_initilize` method to post initialize the normalizer. The requirement 
        is the same as in 1). 
    3) Call `update` before the first call to `transform`, and the static min and static max will be inferred from the data. 
    
    Parameters
    ----------
    min :  The static min of the normalizer. Default to None. 
    max :  The static max of the normalizer. Default to None.
    """
    def __init__(self, min=None, max=None, eps=1e-8):
        BaseNormalizer.__init__(self)
        nn.Module.__init__(self)
        self.register_buffer("_initialized", torch.tensor(False))
        self.eps = eps
        if min is not None and max is not None:
            self._initialize(min=min, max=max)
        
    def _initialize(self, min: Optional[torch.Tensor], max: Optional[torch.Tensor]) -> None:
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
        
    def transform(self, x: torch.Tensor, inverse: bool=False) -> torch.Tensor:
        """
        Transform the input data by data = (data - min) / (max - min). 

        Notes
        -----
        All dimensions except for the dim=0 will be transformed. 

        Parameters
        ----------
        x :  The input data, should be torch.Tensor.
        inverse :  If True, inverse the transformation defined by the normalizer.
        
        Returns
        -------
        torch.Tensor :  The transformed version of the input. 
        """
        if not self._initialized:
            raise ValueError("MinMax Normalizers must be initialized before transforming.")
        if inverse:
            return x*(self.max - self.min) + self.min
        return (x-self.min) / (self.max - self.min+self.eps)
    
    def update(self, x: torch.Tensor) -> None:
        """Use the input data to update the normalizer statistics. 

        Parameters
        ----------
        x : The data to update the normalizer statistics.
        """
        batch_min = torch.min(x, dim=0)
        batch_max = torch.max(x, dim=0)
        
        self._initialize(min=batch_min, max=batch_max)
        