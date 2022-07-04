from typing import Any, Dict, List, Optional, Sequence, Tuple, Type, Union

import torch
import torch.nn as nn

from UtilsRL.net.basic import miniblock, EnsembleLinear

class Attention(nn.Module):
    def __init__(
        self, 
        input_dim: int,
        embed_dim: int,  
        n_head: int, 
        scale: bool = True, 
        self_attention: bool = True, 
        dropout: float = 0.1, 
    ):
        if embed_dim % n_head != 0:
            raise ValueError(f"Attention embed dim must be divisible by n_head, found {output_dim} and {n_head}.")
        self.head_dim = embed_dim // n_head
        self.n_head = n_head
        self.embed_dim = self.embed_dim
        self._do_scale = scale
        self._do_self_attention = self_attention
        
        if self._do_self_attention:
            self.to_context = EnsembleLinear(input_dim, self.head_dim, n_head*3)
        else:
            self.to_context = EnsembleLinear(input_dim, self.head_dim, n_head*2)
            self.to_query = EnsembleLinear(input_dim, self.head_dim, n_head)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        
    def _do_attention(
        self, 
        q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, 
        attention_mask: Optional[Any] = None, 
        head_mask: Optional[Any] = None, 
    ):
        # data shape: (Head, Batch, Length, Embedding)
        o = torch.matmul(q, k.transpose(-1, -2))
        if self._do_scale:
            o /= self.head_dim ** (-0.5)
        if attention_mask:
            o += attention_mask
            
        # 还没怎么搞明白self.cross_attention    
        o = torch.softmax(o, dim=-1)
        o = self.attn_dropout(o)
        
        if head_mask:
            o *= head_mask
            
        att_value = torch.matmul(o, v)
        return att_value, o
    
    def forward(
        self, 
        input: torch.Tensor, 
        query_input: Optional[torch.Tensor] = None, 
        attention_mask: Optional[Any] = None, 
        query_attention_mask: Optional[Any] = None, 
        head_mask: Optional[Any] = None, 
        return_attn_weight: bool = False, 
    ):
        B, L= input.shape[0], input.shape[1]
        if self._do_self_attention:
            q, k, v = self.to_context(input).split(self.head_dim, dim=0)
        else:
            q = self.to_query(query_input)
            k, v = self.to_context(input).split(self.head_dim, dim=0)
        
        attn_value, attn_weight = self._do_attention(q, k, v, attention_mask, head_mask)
        attn_value = self.proj(attn_value.permute(1, 2, 0, 3).reshape(B, L, self.embed_dim))
        attn_value = self.proj_dropout(attn_value)
        
        return attn_value, attn_weight if return_attn_weight else attn_value
    
        