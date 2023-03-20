from typing import Any, Dict, List, Optional, Sequence, Tuple, Type, Union

import torch
import torch.nn as nn


class TransformerEncoderLayer(nn.TransformerEncoderLayer):
    def __init__(
        self, 
        embed_dim: int, 
        num_heads: int, 
        backbone_dim: int=2048, 
        dropout: Optional[float]=0.1, 
        norm_first: bool=False, 
        device=None, 
        dtype=None, 
    ):
        super().__init__(
            d_model=embed_dim, 
            nhead=num_heads, 
            dim_feedforward=backbone_dim, 
            dropout=dropout, 
            batch_first=True, 
            norm_first=norm_first, 
            device=device, 
            dtype=dtype
        )

    def forward(
        self, 
        input: torch.Tensor, 
        attention_mask: Optional[torch.Tensor]=None, 
        key_padding_mask: Optional[torch.Tensor]=None, 
        # is_causal: bool=False
    ):
        return super().forward(input, attention_mask, key_padding_mask)
    
