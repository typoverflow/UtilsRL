from typing import Any, Dict, List, Optional, Sequence, Type, Union

import torch
import torch.nn as nn

from UtilsRL.net.utils import conv2d_out_size

ModuleType = Type[nn.Module]

class AtariConv2d(nn.Module):
    def __init__(
        self, 
        input_channel: int=3, 
        output_channel: int=32, 
        mode: str="RGB",
        activate_last: bool=True,  
        device: Union[str, int, torch.device] = "cpu"
    ) -> None:
        super.__init__()
        self.output_channel = output_channel
        self.input_channel = 3 if mode == "RGB" else 1
        self.model = nn.Sequential(
            nn.Conv2d(input_channel, 16, kernel_size=8, stride=4), 
            nn.BatchNorm2d(16), 
            nn.ReLU(), 
            nn.Conv2d(16, 32, kernel_size=4, stride=2), 
            nn.BatchNorm2d(32), 
            nn.ReLU(), 
            nn.Conv2d(32, output_channel, kernel_size=3, stride=1), 
            nn.BatchNorm2d(output_channel), 
            nn.ReLU() if activate_last else nn.Identity()
        )
    
    def forward(self, img: torch.Tensor, permute_for_torch: False):
        if permute_for_torch:
            img = img.permute(0, 3, 1, 2)
        return self.model(img).reshape(img.shape[0], -1)

atari_conv2d_out_size = conv2d_out_size(conv2d_out_size(conv2d_out_size(84, 8, 4), 4, 2))
        