from typing import Any, Dict, List, Optional, Sequence, Tuple, Type, Union

import torch
import torch.nn as nn

from UtilsRL.net.basic import miniblock, EnsembleLinear

ModuleType = Type[nn.Module]

class Recurrent(nn.Module):
    """
    Creates a Recurrent Neural Network Block.

    Parameters
    ----------
    input_dim :  The dimensions of input. 
    rnn_hidden_dim :  The dimensions of rnn hidden. Default is 128.
    rnn_layer_num :  The num of rnn layers, each layer if of shape `rnn_hidden_dim`. Default is 1.
    device :  The device to run the model on. Default is cpu.
    rnn_type :  The ModuleType to use. Default is nn.GRU. 
    """
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
            batch_first = True
        )
        
    def forward(
        self, 
        input: torch.Tensor, 
        lengths: Sequence[int], 
        pre_hidden: Optional[torch.Tensor] = None
    ):
        if pre_hidden is None:
            pre_hidden = self.zero_hidden(batch_size=input.shape[0]).to(self.device)
        packed = torch.nn.utils.rnn.pack_padded_sequence(input, lengths, batch_first=True, enforce_sorted=False)
        output, hidden = self.rnn(packed, pre_hidden)
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        return output, hidden
    
    def zero_hidden(self, batch_size):
        return torch.zeros([self.rnn_layer_num, batch_size, self.rnn_hidden_dim])

    def to(self, *args, **kwargs):
        device, dtype, _, _ = torch._C._nn._parse_to(*args, **kwargs)
        self.device = device
        super().to(*args, **kwargs)