from typing import Dict, Optional, Any, Sequence, Union

import torch
import torch.nn as nn
import numpy as np

from UtilsRL.math.distributions import TanhNormal
from UtilsRL.rl.net import MLP
from torch.distributions import Categorical, Normal

from abc import ABC, abstractmethod

class BaseActor(nn.Module):
    """BaseActor interface. 
    
    All actors should implement `forward`, `sample` and `evaluate`. 
    """
    def __init__(self):
        super().__init__()
        
    @abstractmethod
    def forward(self, state: torch.Tensor, *args, **kwargs):
        """Forward pass of the actor. 
        
        :param state: state / obs of the environment. 
        """
        raise NotImplementedError
    
    @abstractmethod
    def sample(self, state: torch.Tensor, *args, **kwargs):
        """Sampling procedure of the actor.
        
        :param state: state / obs of the environment.
        """
        raise NotImplementedError
    
    @abstractmethod
    def evaluate(self, state, action, *args, **kwargs):
        """Evaluate the log_prob of the given `action`. 
        
        :param state: state / obs of the environment.
        :param action: action to evaluate. 
        """
        raise NotImplementedError

class DeterministicActor(BaseActor):
    """Actor which outputs a deterministic action for a given state. 
    
    Note that the output action is not bounded within [-1, 1].

    :param backend: feature extraction backend of the actor. 
    :param input_dim: input dimension of the actor. When using `backend`, this should match the output dimension of `backend`. 
    :param output_dim: output dimension of the actor. 
    :param device: device of the actor. 
    :param hidden_dims: hidden dimension of the MLP between `backend` and the output layer. 
    :param linear_layer: linear type of the output layers. 
    """
    
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
        """Forward pass of the actor. 
        
        :param input: state / obs of the environment. 
        """
        return self.output_layer(self.backend(input))
    
    def sample(self, input: torch.Tensor):
        """Sampling process of the actor. However, deterministic actor directly take the output \
            as actions, and there is no further operations. 

        :param input: state / obs of the environent. 
        """
        return self(input)
    
    def evaluate(self, state: torch.Tensor, action: torch.Tensor):
        """Determinisitic actors do not support evaluation. This will raise an error.

        :param state: state / obs of the environment.
        :param action: action to evaluate.
        """
        raise NotImplementedError("Evaluation shouldn't be called for SquashedDeterministicActor.")
        
        
class SquashedDeterministicActor(DeterministicActor):
    """A deterministic actor whose output is squashed into [-1, 1] using `Tanh`. 
    
    Parameters are kept the same as :class:`~UtilsRL.rl.actor.DeterministicActor`.
    """
    
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
        """Sampling process of actor. Note that output is squashed into [-1, 1] with `Tanh`.
        
        :param input: state / obs of the environent. 
        """
        action_prev_tanh = super().forward(input)
        return torch.tanh(action_prev_tanh)
            

class ClippedDeterministicActor(DeterministicActor):
    """A deterministic actor whose output is hard-clipped into [-1, 1]. 
    
    Parameters are kept the same as :class:`~UtilsRL.rl.actor.DeterministicActor`.
    """
    
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
        """Sampling process of actor. Note that output is hard-clipped into [-1, 1].

        :param input: state / obs of the environent. 
        """
        action = super().forward(input)
        return torch.clip(action, min=-1, max=1)
    
    
class GaussianActor(BaseActor):
    """Actor which samples from a gaussian distribution whose mean and std are predicted by networks. 
    
    :param backend: feature extraction backend of the actor. 
    :param input_dim: input dimension of the actor. When using `backend`, this should match the output dimension of `backend`. 
    :param output_dim: output dimension of the actor. 
    :param device: device of the actor. 
    :param reparameterize: whether to use reparameterization trick when sampling. 
    :param conditioned_logstd: whether condition the logstd on inputs. 
    :param fix_logstd: if not `None`, actor will fix the logstd of the sampling distribution. 
    :param hidden_dims: hidden dimension of the MLP between `backend` and the output layer. 
    :param linear_layer: linear type of the output layers. 
    :param logstd_min: minimum value of the logstd, will be used to clip the logstd value. 
    :param logstd_max: maximum value of the logstd, will be used to clip the logstd value.
    """
    
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
        """Forward pass of the actor, will predict mean and logstd of the distribution for sampling. 
        
        Note that if ``fix_logstd`` is not `None`, the logstd will be fixed values; otherwise, if `conditioned_logstd==True`, logstd \
            will conditione on input; if not, logstd are shared across all inputs. 

        :param input: state / obs of the environent.
        """
        out = self.output_layer(self.backend(input))
        if self._logstd_is_layer:
            mean, logstd = torch.split(out, self.output_dim // 2, dim=-1)
        else:
            mean = out
            logstd = self.logstd.broadcast_to(mean.shape)
        logstd = torch.clip(logstd, min=self.logstd_min, max=self.logstd_max)
        return mean, logstd
    
    def sample(self, input: torch.Tensor, deterministic: bool=False, return_mean_logstd: bool=False):
        """Sampling process of the actor. 
        
        :param input: state / obs of the environent.
        :param deterministic: whether to use deterministic sampling. If `True`, mean will be returned as action. 
        :param return_mean_logstd: whether to return mean and logstd of the sampling distribution. If `True`, they will be included in `info` dict
        """
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
        """Evaluate the action given the state. Note that entropy will also be returned. 

        :param state: state of the environment.
        :param action: action to be evaluated.
        """
        mean, logstd = self(state)
        dist = Normal(mean, logstd.exp())
        return dist.log_prob(action).sum(-1, keepdim=True), dist.entropy().sum(-1, keepdim=True)


class SquashedGaussianActor(GaussianActor):
    """Actor which samples from a gaussian distribution whose mean and std are predicted by networks. The output action will be \
        squashed into [-1, 1] with `Tanh`.
    
    Parameters are kept the same as :class:`~UtilsRL.rl.actor.GaussianActor`
    """
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
        """After sampling from gaussian distributions, samples will be squashed into [-1, 1] with `Tanh`.
        
        :param input: state / obs of the environent.
        :param deterministic: whether to use deterministic sampling. If `True`, `Tanh(mean)` will be returned as action.
        :param return_mean_logstd: whether to return mean and logstd of the sampling distribution. If `True`, they will be included in `info` dict. 
        """
        
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
        """Evaluate the action given the state. Log-probability will be corrected according to `SAC` paper.
        """
        mean, logstd = self(state)
        dist = TanhNormal(mean, logstd.exp())
        return dist.log_prob(action).sum(-1, keepdim=True), dist.entropy().sum(-1, keepdim=True)
    
    
class ClippedGaussianActor(GaussianActor):
    """Actor which samples from a gaussian distribution whose mean and std are predicted by networks. The output action will be \
        hard-clippped into [-1, 1].
    
    Parameters are kept the same as :class:`~UtilsRL.rl.actor.GaussianActor`
    """
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
    """Actor which samples from a categorical distribution whose logits are predicted by networks. 
    
    :param backend: feature extraction backend of the actor. 
    :param input_dim: input dimension of the actor. When using `backend`, this should match the output dimension of `backend`. 
    :param output_dim: output dimension of the actor. 
    :param device: device of the actor. 
    :param hidden_dims: hidden dimension of the MLP between `backend` and the output layer. 
    :param linear_layer: linear type of the output layers. 
    """
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
    
    def sample(self, input: torch.Tensor, deterministic: bool=False, return_probs: bool=False):
        """Sampling from a categorical distribution where probs are predicted by networks. 

        :param input: state / obs of the environent.
        :param deterministic: whether to use deterministic sampling. If `True`, `argmax` will be returned as action.
        :param return_probs: whether to return probs of the sampling distribution. If `True`, they will be included in `info` dict.
        """
        probs = self.forward(input)
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
        """Evaluate the action given the state, will return log-probability of the action and entropy of the distribution.

        :param state: state of the environent.
        :param action: action to be evaluated.
        """
        if len(action.shape) == 2:
            action = action.view(-1)
        probs = self.forward(state)
        dist = Categorical(probs=probs)
        return dist.log_prob(action).unsqueeze(-1), dist.entropy().unsqueeze(-1)
            
            

        
        