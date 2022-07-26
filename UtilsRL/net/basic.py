from typing import List, Optional, Type, Any

import torch
import torch.nn as nn

ModuleType = Type[nn.Module]

def miniblock(
    input_dim: int,
    output_dim: int = 0,
    norm_layer: Optional[ModuleType] = None,
    activation: Optional[ModuleType] = None,
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
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ensemble_size = ensemble_size
        self.register_parameter("weight", torch.nn.Parameter(torch.zeros(ensemble_size, in_features, out_features)))
        self.register_parameter("bias", torch.nn.Parameter(torch.zeros(ensemble_size, 1, out_features)))
        torch.nn.init.trunc_normal_(self.weight, std=1/(2*in_features**0.5))
        
    def forward(self, input: torch.Tensor):
        if input.shape[0] != self.ensemble_size:
            return torch.einsum('ij,bjk->bik', input, self.weight) + self.bias
        else:
            return torch.einsum('bij,bjk->bik', input, self.weight) + self.bias