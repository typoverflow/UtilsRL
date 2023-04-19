import math, os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load
from torch.nn import functional as F
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type, Union

T_MAX = 1024

wkv_op = load(
    name="wkv",
    sources=["UtilsRL/operator/wkv_op.cpp", "UtilsRL/operator/wkv_op.cu"],
    verbose=True,
    extra_cuda_cflags=[
        '-res-usage',
        '--maxrregcount 60',
        '--use_fast_math',
        '-O3',
        '-Xptxas -O3',
        f'-DTmax={T_MAX}'
    ],
    build_directory="build"
)

class WKV(torch.autograd.Function):
    @staticmethod
    def forward(ctx, B, T, C, w, u, k, v, h1, h2, h3):
        ctx.B = B
        ctx.T = T
        ctx.C = C
        assert T <= T_MAX
        assert B * C % min(C, 1024) == 0
        w = - torch.exp(w.contiguous())
        u = u.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        h1 = h1.contiguous()
        h2 = h2.contiguous()
        h3 = h3.contiguous()
        y = torch.empty((B, T, C), device=w.device, memory_format=torch.contiguous_format)
        wkv_op.forward(B, T, C, w, u, k, v, h1, h2, h3, y)
        ctx.save_for_backward(w, u, k, v, h1, h2, h3, y)
        return y

    @staticmethod
    def backward(ctx: Any, gy: Any) -> Any:
        B = ctx.B
        T = ctx.T
        C = ctx.C
        assert T <= T_MAX
        assert B * C % min(C, 1024) == 0
        w, u, k, v, h1, h2, h3, y = ctx.saved_tensors
        device = w.device
        gw = torch.zeros((B, C), device=device).contiguous()
        gu = torch.zeros((B, C), device=device).contiguous()
        gk = torch.zeros((B, T, C), device=device).contiguous()
        gv = torch.zeros((B, T, C), device=device).contiguous()
        wkv_op.backward(B, T, C, w, u, k, v, h1, h2, h3, y, gy.contiguous(), gw, gu, gk, gv)
        gw = torch.sum(gw, dim=0)
        gu = torch.sum(gu, dim=0)
        return (None, None, None, gw, gu, gk, gv)
        

def RUN_CUDA(B, T, C, w, u, k, v):
    return WKV.apply(B, T, C, w.cuda(), u.cuda(), k.cuda(), v.cuda())


class RWKVChannelMix(nn.Module):
    def __init__(
        self, 
        embed_dim: int, 
        backbone_dim: Optional[int]=None
    ) -> None:
        super().__init__()
        if backbone_dim is None:
            backbone_dim = 4 * embed_dim
        self.key = nn.Linear(embed_dim, backbone_dim, bias=False)
        self.value = nn.Linear(embed_dim, backbone_dim, bias=False)
        self.receptance = nn.Linear(embed_dim, backbone_dim, bias=False)

    def forward(self, input: torch.Tensor):
        k = torch.square(torch.relu(self.key(input)))
        kv = self.value(k)
        rkv = torch.sigmoid(self.receptance(input)) * kv
        return rkv
    
    
class RWKVTimeMix(nn.Module):
    def __init__(
        self, 
        embed_dim: int, 
    ) -> None:
        super().__init__()
        self.rvk = nn.Linear(embed_dim, embed_dim*3, bias=False)
        self.output = nn.Linear(embed_dim, embed_dim, bias=False)
        self.time_decay = nn.Parameter(torch.ones([embed_dim, ]))
        self.time_first = nn.Parameter(torch.ones([embed_dim, ]))
        
    def forward(self, input: torch.Tensor):
        B, T, C = input.shape
        r, v, k = torch.split(self.rvk(input), 3)
        rwkv = torch.sigmoid(r) * RUN_CUDA(B, T, C, self.time_decay, self.time_first, k, v)
        return self.output(rwkv)
    

class RWKVBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int, 
    ) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.attention = RWKVTimeMix(embed_dim)
        self.ffn = RWKVChannelMix(embed_dim)
        
    def forward(self, input: torch.Tensor):
        residual = input
        residual = residual + self.attention(self.ln1(residual))
        residual = residual + self.ffn(self.ln2(residual))
        return residual
    
    
class RWKV(nn.Module):
    def __init__(
        self, 
        input_dim: int, 
        embed_dim: int, 
        num_layers: int, 
        output_dim: int=0, 
    ) -> None:
        super().__init__()
        self.input_embed = nn.Linear(input_dim, embed_dim)
        pos_len = pos_len or 4096
        
        self.blocks = nn.ModuleList([ RWKVBlock(embed_dim) ] for _ in range(num_layers))
        self.ln_in = nn.LayerNorm(embed_dim)
        self.ln_out = nn.LayerNorm(embed_dim)
        
        if output_dim > 0:
            self.output = nn.Linear(embed_dim, output_dim)
        else:
            self.output = nn.Identity()
            
    def forward(
        self, 
        inputs: torch.Tensor, 
        do_embedding: bool=True
    ):
        if do_embedding:
            inputs = self.input_embed(inputs)
        inputs = self.ln_in(inputs)
        inputs = self.blocks(inputs)
        inputs = self.output(self.ln_out(inputs))
        return inputs
        
        
        
        
        