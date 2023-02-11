from typing import List, Optional, Type, Any

import math
import torch
import torch.nn as nn

ModuleType = Type[nn.Module]

def miniblock(
    input_dim: int,
    output_dim: int = 0,
    norm_layer: Optional[ModuleType] = None,
    activation: Optional[ModuleType] = None,
    dropout: Optional[ModuleType] = None, 
    linear_layer: ModuleType = nn.Linear,
    *args, 
    **kwargs
) -> List[nn.Module]:
    """Construct a miniblock with given input/output-size, norm layer and
    activation."""
    layers: List[nn.Module] = [linear_layer(input_dim, output_dim, *args, **kwargs)]
    if norm_layer is not None:
        layers += [norm_layer(output_dim)]  # type: ignore
    if activation is not None:
        layers += [activation()]
    if dropout is not None and dropout > 0:
        layers += [nn.Dropout(dropout)]
    return layers


class EnsembleLinear(nn.Module):
    def __init__(
        self,
        in_features, 
        out_features, 
        ensemble_size: int = 1,
        bias: bool = True, 
        device: Optional[Any] = None, 
        dtype: Optional[Any] = None
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ensemble_size = ensemble_size
        self.register_parameter("weight", torch.nn.Parameter(torch.empty([ensemble_size, in_features, out_features], **factory_kwargs)))
        if bias:
            self.register_parameter("bias", torch.nn.Parameter(torch.empty([ensemble_size, 1, out_features], **factory_kwargs)))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()
        
    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            torch.nn.init.uniform_(self.bias, -bound, bound)
        
    def forward(self, input: torch.Tensor):
        if self.bias is None:
            bias = 0
        else:
            bias = self.bias
        if input.shape[0] != self.ensemble_size:
            return torch.einsum('ij,bjk->bik', input, self.weight) + bias
        else:
            return torch.einsum('bij,bjk->bik', input, self.weight) + bias


class NoisyLinear(nn.Linear):
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        std_init: float=0.5, 
        bias: bool=True, 
        device=None, 
        dtype=None
    ):
        super().__init__(in_features, out_features, bias, device, dtype) # this will do the initialization

        self.std_init = std_init
        self.register_parameter("weight_std", torch.nn.Parameter(torch.empty(out_features, in_features)))
        self.weight_std.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.register_buffer("weight_noise", torch.empty_like(self.weight))
        if bias:
            self.register_parameter("bias_std", torch.nn.Parameter(torch.empty(out_features)))
            self.bias_std.data.fill_(self.std_init / math.sqrt(self.out_features))
            self.register_buffer("bias_noise", torch.empty_like(self.bias))
        else:
            self.register_parameter("bias_std", None)
            self.register_buffer("bias_noise", None)
        
        self.reset_noise()

    @staticmethod
    def scaled_noise(size: int):
        x = torch.randn(size)
        return x.sign().mul(x.abs().sqrt())

    def forward(self, x: torch.Tensor, reset_noise=False):
        if self.training:
            if reset_noise:
                self.reset_noise()
            if self.bias is not None:
                return torch.nn.functional.linear(
                        x, 
                        self.weight + self.weight_std * self.weight_noise, 
                        self.bias + self.bias_std * self.bias_noise
                    )
            else:
                return torch.nn.functional.linear(
                    x, 
                    self.weight + self.weight_std * self.weight_noise, 
                    None
                )
        else:
            return torch.nn.functional.linear(x, self.weight, self.bias)
        
    def reset_noise(self):
        device = self.weight.data.device
        epsilon_in = self.scaled_noise(self.in_features)
        epsilon_out = self.scaled_noise(self.out_features)
        self.weight_noise.data = torch.matmul(
            torch.unsqueeze(epsilon_out, -1), other=torch.unsqueeze(epsilon_in, 0)
        ).to(device)
        if self.bias is not None:
            self.bias_noise.data = epsilon_out.to(device)
        
        