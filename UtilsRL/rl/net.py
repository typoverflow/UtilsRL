# from https://github.com/thu-ml/tiamshou


from typing import Any, Dict, List, Optional, Sequence, Tuple, Type, Union

import torch
import torch.nn as nn
import numpy as np

ModuleType = Type[nn.Module]

def miniblock(
    input_dim: int,
    output_dim: int = 0,
    norm_layer: Optional[ModuleType] = None,
    activation: Optional[ModuleType] = None,
    linear_layer: Type[nn.Linear] = nn.Linear,
) -> List[nn.Module]:
    """Construct a miniblock with given input/output-size, norm layer and
    activation."""
    layers: List[nn.Module] = [linear_layer(input_dim, output_dim)]
    if norm_layer is not None:
        layers += [norm_layer(output_dim)]  # type: ignore
    if activation is not None:
        layers += [activation()]
    return layers

class MLP(nn.Module):
    def __init__(
        self, 
        input_dim: int, 
        output_dim: int = 0, 
        hidden_dims: Sequence[int] = [], 
        norm_layer: Optional[Union[ModuleType, Sequence[ModuleType]]] = None, 
        activation: Optional[Union[ModuleType, Sequence[ModuleType]]] = nn.ReLU, 
        device: Optional[Union[str, int, torch.device]] = "cpu", 
        linear_layer: Type[nn.Linear] = nn.Linear,
    ) -> None:
        super().__init__()
        self.device = device
        if norm_layer:
            if isinstance(norm_layer, list):
                assert len(norm_layer) == len(hidden_dims)
                norm_layer_list = norm_layer
            else:
                morm_layer_list = [norm_layer for _ in range(len(hidden_dims))]
        else:
            norm_layer_list = [None]*len(hidden_dims)
        
        if activation:
            if isinstance(activation, list):
                assert len(activation) == len(hidden_dims)
                activation_list = activation
            else:
                activation_list = [activation for _ in range(len(hidden_dims))]
        else:
            activation_list = [None]*len(hidden_dims)
        
        hidden_dims = [input_dim] + list(hidden_dims)
        model = []
        for in_dim, out_dim, norm, activ in zip(
            hidden_dims[:-1], hidden_dims[1:], norm_layer_list, activation_list
        ):
            model += miniblock(in_dim, out_dim, norm, activ, linear_layer)
        if output_dim > 0:
            model += [linear_layer(hidden_dims[-1], output_dim)]
        self.output_dim = output_dim or hidden_dims[-1]
        
        self.model = nn.Sequential(*model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)        # do we need to flatten x staring at dim=1 ?
    

class Recurrent(nn.Module):
    def __init__(
        self, 
        input_dim: int, 
        rnn_hidden_dim: int = 128, 
        rnn_layer_num: int = 1, 
        device: Union[str, int, torch.device] = "cpu", 
        rnn_type: ModuleType = nn.GRU
    ) -> None:
        super().__init__()
        self.device = device
        self.rnn_hidden_dim = rnn_hidden_dim
        self.rnn_layer_num = rnn_layer_num
        self.rnn = rnn_type(
            input_dim, 
            rnn_hidden_dim, 
            rnn_layer_num, 
            batch_first=True
        )
        
    def forward(self, x, lengths, pre_hidden=None):
        if pre_hidden is None:
            pre_hidden = self.zero_hidden(batch_size=x.shape[0]).to(self.device)
        packed = torch.nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        output, hidden = self.rnn(packed, pre_hidden)
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        return output, hidden
    
    def zero_hidden(self, batch_size):
        return torch.zeros([self.rnn_layer_num, batch_size, self.rnn_hidden_dim])