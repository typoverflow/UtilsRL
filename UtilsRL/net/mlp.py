from typing import Any, Dict, List, Optional, Sequence, Tuple, Type, Union

import torch
import torch.nn as nn

from UtilsRL.net.basic import miniblock, EnsembleLinear

ModuleType = Type[nn.Module]

class MLP(nn.Module):
    def __init__(
        self, 
        input_dim: int, 
        output_dim: int = 0, 
        hidden_dims: Sequence[int] = [], 
        norm_layer: Optional[Union[ModuleType, Sequence[ModuleType]]] = None, 
        activation: Optional[Union[ModuleType, Sequence[ModuleType]]] = nn.ReLU, 
        dropout: Optional[Union[float, Sequence[float]]] = None, 
        device: Optional[Union[str, int, torch.device]] = "cpu", 
        linear_layer: nn.Module=nn.Linear
    ) -> None:
        super().__init__()
        if norm_layer:
            if isinstance(norm_layer, list):
                assert len(norm_layer) == len(hidden_dims)
                norm_layer_list = norm_layer
            else:
                norm_layer_list = [norm_layer for _ in range(len(hidden_dims))]
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
        
        if dropout:
            if isinstance(dropout. list):
                assert len(dropout) == len(hidden_dims)
                dropout_list = dropout
            else:
                dropout_list = [dropout for _ in range(len(hidden_dims))]
        else:
            dropout_list = [None]*len(hidden_dims)
                        
        hidden_dims = [input_dim] + list(hidden_dims)
        model = []
        for in_dim, out_dim, norm, activ, dropout in zip(
            hidden_dims[:-1], hidden_dims[1:], norm_layer_list, activation_list, dropout_list
        ):
            model += miniblock(in_dim, out_dim, norm, activ, dropout, device=device, linear_layer=linear_layer)
        if output_dim > 0:
            model += [linear_layer(hidden_dims[-1], output_dim, device=device)]
        self.output_dim = output_dim or hidden_dims[-1]
        
        self.model = nn.Sequential(*model)
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.model(input)        # do we need to flatten x staring at dim=1 ?


class EnsembleMLP(nn.Module):
    def __init__(
        self, 
        input_dim: int, 
        output_dim: int = 0, 
        ensemble_size: int = 1, 
        hidden_dims: Sequence[int] = [], 
        norm_layer: Optional[Union[ModuleType, Sequence[ModuleType]]] = None, 
        activation: Optional[Union[ModuleType, Sequence[ModuleType]]] = nn.ReLU, 
        dropout: Optional[Union[float, Sequence[float]]] = None, 
        share_hidden_layer: Union[Sequence[bool], bool] = False, 
        device: Optional[Union[str, int, torch.device]] = "cpu", 
    ) -> None:
        super().__init__()
        self.ensemble_size = ensemble_size
        
        if norm_layer:
            if isinstance(norm_layer, list):
                assert len(norm_layer) == len(hidden_dims)
                norm_layer_list = norm_layer
            else:
                norm_layer_list = [norm_layer for _ in range(len(hidden_dims))]
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
            
        if dropout:
            if isinstance(dropout. list):
                assert len(dropout) == len(hidden_dims)
                dropout_list = dropout
            else:
                dropout_list = [dropout for _ in range(len(hidden_dims))]
        else:
            dropout_list = [None]*len(hidden_dims)
            
        if share_hidden_layer:
            if isinstance(share_hidden_layer, list):
                assert len(share_hidden_layer) == len(hidden_dims)
                share_hidden_layer_list = share_hidden_layer
            else:
                share_hidden_layer_list = [share_hidden_layer for _ in len(hidden_dims)]
        else:
            share_hidden_layer_list = [False]*len(hidden_dims)
                
        
        hidden_dims = [input_dim] + list(hidden_dims)
        model = []
        for in_dim, out_dim, norm, activ, dropout, share_layer in zip(
            hidden_dims[:-1], hidden_dims[1:], norm_layer_list, activation_list,dropout_list, share_hidden_layer_list
        ):
            if share_layer:      
                model += miniblock(in_dim, out_dim, norm, activ, dropout, linear_layer=nn.Linear, device=device)
            else:
                model += miniblock(in_dim, out_dim, norm, activ, dropout, linear_layer=EnsembleLinear, ensemble_size=ensemble_size, device=device)
        if output_dim > 0:
            model += [EnsembleLinear(hidden_dims[-1], output_dim, ensemble_size, device=device)]
        self.output_dim = output_dim or hidden_dims[-1]
        
        self.model = nn.Sequential(*model)
        
    def forward(self, input: torch.Tensor):
        return self.model(input)