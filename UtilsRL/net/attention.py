from turtle import back
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
        dropout: Optional[float] = None, 
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
        if dropout:
            self.attn_dropout = nn.Dropout(dropout)
            self.proj_dropout = nn.Dropout(dropout)
        else:
            self.attn_dropout = self.proj_dropout = nn.Identity()
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
        return_attn_weight: bool = True, 
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
        
        if return_attn_weight:
            return attn_value, attn_weight
        else:
            return attn_value
    

# How about not implementing decoder for now ?
class TransformerBlock(nn.Module):
    def __init__(
        self, 
        input_dim: int, 
        embed_dim: int, 
        n_head: int, 
        backbone_dim: Optional[int] = None, 
        scale: bool = True, 
        self_attention: bool = True, 
        dropout: Optional[float] = None
    ): 
        if backbone_dim is None:
            backbone_dim = 4 * embed_dim
        self.attention = Attention(input_dim, embed_dim, n_head, scale, self_attention, dropout)
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, backbone_dim), 
            nn.ReLU(), 
            nn.Linear(backbone_dim, embed_dim), 
            nn.Dropout(dropout) if dropout else nn.Identity()
        )
        
    def forward(
        self, 
        input: torch.Tensor, 
        query_input: Optional[torch.Tensor] = None, 
        attention_mask: Optional[Any] = None, 
        query_attention_mask: Optional[Any] = None, 
        head_mask: Optional[Any] = None, 
        return_attn_weight: bool = True
    ): 
        residual = input
        attn_outputs = self.attention.forward(
            self.layer_norm1(input), 
            query_input=None, 
            attention_mask=attention_mask, 
            head_mask=head_mask, 
            return_attn_weight=return_attn_weight
        )
        if return_attn_weight:
            attn_value, attn_weight = attn_outputs
        else:
            attn_value = attn_outputs
        attn_value += residual
        
        residual = attn_value
        attn_value = self.ff(self.layer_norm2(attn_value))
        attn_value = attn_value + residual
        
        if return_attn_weight:
            return attn_value, attn_weight
        else:
            return attn_value


# class Transformer(nn.Module):
#     def __init__(
#         self, 
#         input_dim: int, 
#         embed_dim: int, 
#         n_head: int, 
#         n_layer: int, 
#         output_dim: int = 0
#     ):
#         self.embed_layer = nn.Linear(input_dim, embed_dim)
#         blocks = []
#         for i in range(n_layer):
#             blocks.append(TransformerBlock(
#                 embed_dim, embed_dim, n_head, return_attn_weight=False
#             ))
#         self.transformer_blocks = nn.Sequential(*blocks)
#         if output_dim > 0:
#             self.output_layer = nn.Linear(embed_dim, output_dim)
#         else:
#             self.output_layer = nn.Identity()
        
#     def forward(
#         self, 
#         input: torch.Tensor, 
#     ):
#         input = self.output_layer(self.transformer_blocks(self.embed_layer(input)))
        