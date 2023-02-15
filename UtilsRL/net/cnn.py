from typing import Any, Dict, List, Optional, Sequence, Type, Union

import torch
import torch.nn as nn

from UtilsRL.net.utils import conv2d_out_size

ModuleType = Type[nn.Module]

class AtariConv2d(nn.Module):
    """
    Conolutional encoder for Atari 2000 Games.
    
    Parameters
    ----------
    input_channel :  The number of input channels. Default is 4.
    output_channel :  The number of output channels. Default is 64.
    mode :  The mode to use. Default is RGB.
    activate_last :  Whether or not to activate the last layer in the network. Default is True.
    device :  The device to run the network on. Default is CPU.
    """
    def __init__(
        self, 
        input_channel: int=4, 
        output_channel: int=64, 
        mode: str="RGB",
        activate_last: bool=True,  
        device: Union[str, int, torch.device] = "cpu"
    ) -> None:
        super().__init__()
        self.output_channel = output_channel
        self.input_channel = 3 if mode == "RGB" else 1
        self.model = nn.Sequential(
            nn.Conv2d(input_channel, 32, kernel_size=8, stride=4), 
            nn.BatchNorm2d(32), 
            nn.ReLU(), 
            nn.Conv2d(32, 64, kernel_size=4, stride=2), 
            nn.BatchNorm2d(64), 
            nn.ReLU(), 
            nn.Conv2d(64, output_channel, kernel_size=3, stride=1), 
            nn.BatchNorm2d(output_channel), 
            nn.ReLU() if activate_last else nn.Identity()
        )
    
    def forward(self, img: torch.Tensor, permute_for_torch: bool=False):
        if permute_for_torch:
            img = img.permute(0, 3, 1, 2)
        return self.model(img).reshape(img.shape[0], -1)

atari_conv2d_out_size = conv2d_out_size(conv2d_out_size(conv2d_out_size(84, 8, 4), 4, 2), 3, 1)
        

class CNN(nn.Module):
    """
    Standard convolutional neural network module. 
    
    Parameters
    ----------
    input_channel :  The number of input channels. Default is 4.
    channels : Output channels list. 
    kernels : Kernel sizes. 
    strides : Strides
    activate_last : Whether to activate the output of last CNN layer or not.
    device : The device to use, default to cpu.
    """
    def __init__(
        self, 
        input_channel: int, 
        channels: Union[Sequence[int], int]=[], 
        kernels: Union[Sequence[int], int]=[], 
        strides: Union[Sequence[int], int]=[], 
        activate_last: bool=True, 
        device: Union[str, int, torch.device]="cpu"
    ) -> None:
        super().__init__()
        self.input_channel = input_channel
        if isinstance(channels, int):
            channels = [channels, ]
        if isinstance(kernels, int):
            kernels = [kernels, ]
        if isinstance(strides, int):
            strides = [strides, ]
            
        if len(channels) == 0:
            channels = [input_channel, input_channel]
        else:
            channels = [input_channel] + list(channels)
        model = []
        for ic, oc, k, s in zip(channels[:-1], channels[1:], kernels, strides):
            model.extend([
                nn.Conv2d(ic, oc, k, s), 
                nn.BatchNorm2d(oc), 
                nn.ReLU()
            ])
        if not activate_last:
            model.pop(-1)
        self.model = nn.Sequential(*model)
        
    def forward(self, img: torch.Tensor, permute_for_torch: bool=False):
        if permute_for_torch:
            img = img.permute(0, 3, 1, 2)
        return self.model(img).reshape(img.shape[0], -1)