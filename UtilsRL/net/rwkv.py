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
    build_directory="build")


class WKV(torch.autograd.Function):
    @staticmethod
    def forward(ctx, B, T, C, w, u, k, v):
        ctx.B = B
        ctx.T = T
        ctx.C = C
        assert T <= T_MAX
        assert B * C % min(C, 1024) == 0
        if '32' in os.environ['RWKV_FLOAT_MODE']:
            w = -torch.exp(w.contiguous())
            u = u.contiguous()
            k = k.contiguous()
            v = v.contiguous()
        else:
            w = -torch.exp(w.float().contiguous())
            u = u.float().contiguous()
            k = k.float().contiguous()
            v = v.float().contiguous()
        ctx.save_for_backward(w, u, k, v)
        y = torch.empty((B, T, C), device='cuda', memory_format=torch.contiguous_format)
        wkv_op.forward(B, T, C, w, u, k, v, y)
        if '32' in os.environ['RWKV_FLOAT_MODE']:
            return y
        elif os.environ['RWKV_FLOAT_MODE'] == 'fp16':
            return y.half()
        elif os.environ['RWKV_FLOAT_MODE'] == 'bf16':
            return y.bfloat16()
        return None
    
    @staticmethod
    def backward(ctx: Any, grad_outputs: Any) -> Any:
        B = ctx.B
        T = ctx.T
        C = ctx.C
        assert T <= T_MAX
        assert B * C % min(C, 1024) == 0
        w, u, k, v = ctx.saved_tensors
        gw = torch.zeros((B, C), device='cuda').contiguous()
        gu = torch.zeros((B, C), device='cuda').contiguous()
        gk = torch.zeros((B, T, C), device='cuda').contiguous()
        gv = torch.zeros((B, T, C), device='cuda').contiguous()
        if '32' in os.environ['RWKV_FLOAT_MODE']:
            wkv_op.backward(B, T, C, w, u, k, v, grad_outputs.contiguous(), gw, gu, gk, gv)
        else:
            wkv_op.backward(B, T, C, w, u, k, v, grad_outputs.float().contiguous(), gw, gu, gk, gv)
        gw = torch.sum(gw, dim=0)
        gu = torch.sum(gu, dim=0)
        if '32' in os.environ['RWKV_FLOAT_MODE']:
            return (None, None, None, gw, gu, gk, gv)
        elif os.environ['RWKV_FLOAT_MODE'] == 'fp16':
            return (None, None, None, gw.half(), gu.half(), gk.half(), gv.half())
        elif os.environ['RWKV_FLOAT_MODE'] == 'bf16':
            return (None, None, None, gw.bfloat16(), gu.bfloat16(), gk.bfloat16(), gv.bfloat16())


def call_wkv_op(
    B,
    T,
    C,
    w,
    u,
    k,
    v):
    return WKV.apply(
        B,
        T,
        C,
        w.cuda(),
        u.cuda(),
        k.cuda(),
        v.cuda())


class RWKVTimeMix(nn.Module):
    def __init__(
        self, 
        embed_dim
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        hidden_size = self.embed_dim
        self.key = nn.Linear(self.embed_dim, hidden_size, bias=False)
        self.value = nn.Linear(self.embed_dim, hidden_size, bias=False)
        self.receptance = nn.Linear(self.embed_dim, hidden_size, bias=False)
        self.output = nn.Linear(hidden_size, self.embed_dim, bias=False)
        
        decay_speed = torch.ones(hidden_size)
        for h in range(hidden_size):
            decay_speed[h] = -5 + 8 * (h / (hidden_size - 1)) ** (0.7 + 1.3 * 1)
        self.time_decay = nn.Parameter(decay_speed)
        zigzag = (torch.tensor([(i+1)%3 - 1 for i in range(hidden_size)]) * 0.5)
        self.time_first = nn.Parameter(torch.ones(hidden_size) * math.log(0.3) + zigzag)
        
        
        self.key.scale_init = 0
        self.receptance.scale_init = 0
        self.output.scale_init = 0
    
    def forward(self, input):
        B, T, C = input.size()
        k = self.key(input)
        v = self.receptance(input)
        r = self.receptance(input)
        sr = torch.sigmoid(r)
        
        rwkv = sr * call_wkv_op(
            B,
            T,
            C,
            self.time_decay,
            self.time_first,
            k,
            v
        )
        rwkv = self.output(rwkv)
        return rwkv


class RWKVChannelMix(nn.Module):
    def __init__(
        self,
        embed_dim
    ):
        self.embed_dim = embed_dim
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        hidden_size = 4 * self.embed_dim
        self.key = nn.Linear(self.embed_dim, hidden_size, bias=False)
        self.value = nn.Linear(hidden_size, self.embed_dim, bias=False)
        self.receptance = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        
        self.value.scale_init = 0
        self.receptance.scale_init = 0
        
    def forward(self, input):
        key = self.key(input)
        kv = self.square(torch.relu(key))
        kv = self.value(kv)
        rkv = torch.sigmoid(self.receptance(input)) * kv
        return rkv


class RWKVBlock(nn.Module):
    def __init__(
        self, 
        embed_dim: int,
        layer_id: int,
        model_type: Optional[str]="ffnpre"
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        if layer_id == 0:
            self.ln0 = nn.LayerNorm(self.embed_dim)
        if layer_id == 0 and model_type == "ffnpre":
            self.rwkv = RWKVChannelMix(self.embed_dim)
        else:
            self.rwkv = RWKVTimeMix(self.embed_dim)
        self.ln1 = nn.LayerNorm(self.embed_dim)
        self.ln2 = nn.LayerNorm(self.embed_dim)
        self.ff = RWKVChannelMix(self.embed_dim)

    def forward(
        self, 
        input: torch.Tensor
    ):
        if self.layer_id == 0:
            residual = self.ln0(input)
        else:
            residual = input
        rwkv_output = self.rwkv(input)
        residual = residual + rwkv_output
        residual = residual + self.ff(self.ln2(residual))
        return residual

class RWKV(nn.Module):
    def __init__(
        self,
        input_dim: int,
        embed_dim: int,
        num_layers: int,
        ctx_len: int
    ) -> None:
        super().__init__()
        self.input_embed = nn.Linear(input_dim, embed_dim)
        self.ctx_len = ctx_len
        
        self.blocks = nn.Module([
            RWKVBlock(
                embed_dim=embed_dim,
                layer_id=i
            ) for i in range(num_layers)
        ])
        self.ln_out = nn.LayerNorm(embed_dim)
        
        self.head = nn.Linear(embed_dim, input_dim)
    
    @property
    def ctx_len(self):
        return self.ctx_len
    
    def forward(self, inputs):
        B, T = inputs.size()
        assert T <= self.ctx_len, "len(input) too long"
        x = self.emb(inputs)
        x = self.blocks(x)
        x = self.ln_out(x)
        return self.head(x)
