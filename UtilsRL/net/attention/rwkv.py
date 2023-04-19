import numpy as np
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load
from torch.nn import functional as F
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type, Union

T_MAX = 1024

wkv_op = load(
    name="wkv_extend",
    sources=["UtilsRL/operator/wkv_op_extend.cpp", "UtilsRL/operator/wkv_op_extend.cu"],
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
        device = w.device
        w = - torch.exp(w.contiguous())
        u = u.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        h1 = h1.contiguous()
        h2 = h2.contiguous()
        h3 = h3.contiguous()
        y = torch.empty((B, T, C), device=device, memory_format=torch.contiguous_format)
        oh1 = torch.empty((B, C), device=device, memory_format=torch.contiguous_format)
        oh2 = torch.empty((B, C), device=device, memory_format=torch.contiguous_format)
        oh3 = torch.empty((B, C), device=device, memory_format=torch.contiguous_format)
        wkv_op.forward(B, T, C, w, u, k, v, h1, h2, h3, y, oh1, oh2, oh3)
        ctx.save_for_backward(w, u, k, v, h1, h2, h3, y)
        return y, oh1, oh2, oh3

    @staticmethod
    def backward(ctx: Any, gy, oh1, oh2, oh3) -> Any:
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
        return (None, None, None, gw, gu, gk, gv, None, None, None)
        

def RUN_CUDA(B, T, C, w, u, k, v, h1, h2, h3):
    return WKV.apply(B, T, C, w, u, k, v, h1, h2, h3)


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
        
    def forward(
        self, 
        input: torch.Tensor, 
        hidden: Optional[torch.Tensor]=None, 
        cell_state: Optional[torch.Tensor]=None, 
    ):
        B, T, C = input.shape
        r, v, k = torch.split(self.rvk(input), 3)
        if hidden is None:
            hidden = torch.zeros([B, C]).to(input.device)
        if cell_state is None:
            h2 = torch.zeros([B, C]).to(input.device)
            h3 = torch.full([B, C], fill_value=-1e38).to(input.device)
        else:
            h2, h3 = torch.split(cell_state, 2, -1)
        
        wkv, h1, h2, h3 = RUN_CUDA(B, T, C, self.time_decay, self.time_first, k, v, hidden, h2, h3)
        rwkv = torch.sigmoid(r) * wkv
        c = torch.concat([h2, h3], dim=-1)
        return self.output(rwkv), h1, c
    

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
        
    def forward(
        self, 
        input: torch.Tensor, 
        hidden: Optional[torch.Tensor]=None, 
        cell_state: Optional[torch.Tensor]=None
    ):
        residual = input
        input, h, c = self.attention(self.ln1(input), hidden=hidden, cell_state=cell_state)
        residual = residual + input
        residual = residual + self.ffn(self.ln2(residual))
        return residual, h, c
    
    
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
        rwkv_hiddens: Optional[None], 
        rwkv_cell_states: Optional[None], 
        do_embedding: bool=True
    ):
        if do_embedding:
            inputs = self.input_embed(inputs)
        inputs = self.ln_in(inputs)
        if rwkv_hiddens is not None and rwkv_cell_states is not None:
            rwkv_hiddens = torch.split(rwkv_hiddens, len(self.blocks))
            rwkv_cell_states = torch.split(rwkv_hiddens, len(self.blocks))
        else:
            rwkv_hiddens = [None] * len(self.blocks)
            rwkv_cell_states = [None] * len(self.blocks)
        new_hiddens = []
        new_cell_states = []
        for idx, block in enumerate(self.blocks):
            inputs, h, c = block(inputs, rwkv_hiddens[idx], rwkv_cell_states[idx])
            new_hiddens.append(h)
            new_cell_states.append(c)
        
        inputs = self.output(self.ln_out(inputs))
        new_hiddens = torch.concat(new_hiddens, dim=0)
        new_cell_states = torch.concat(new_cell_states, dim=0)
        return inputs, new_hiddens, new_cell_states
        
        
class DecisionRWKV(RWKV):
    def __init__(
        self,
        obs_dim: int, 
        action_dim: int, 
        embed_dim: int, 
        num_layers: int, 
        seq_len: int, 
        episode_len: int, 
    ) -> None:
        super().__init__(
            input_dim=embed_dim, # actually not used
            embed_dim=embed_dim, 
            num_layers=num_layers, 
            output_dim=0 # we manually handle the output outside
        )
        # RWKV only encode relative timestep but not absolute timestep, so here we still need to  embed
        # the absolute timestep info
        self.pos_embed = nn.Linear(1, embed_dim)
        self.obs_embed = nn.Linear(obs_dim, embed_dim)
        self.act_embed = nn.Linear(action_dim, embed_dim)
        self.ret_embed = nn.Linear(1, embed_dim)
        self.action_head = nn.Sequential(nn.Linear(embed_dim, action_dim), nn.Tanh())

    def forward(
        self, 
        states: torch.Tensor, 
        actions: torch.Tensor, 
        returns_to_go: torch.Tensor, 
        timesteps: torch.Tensor, 
        hiddens: Optional[torch.Tensor]=None, 
        cell_states: Optional[torch.Tensor]=None
    ):
        B, L, *_ = states.shape
        time_embedding = self.pos_embed(timesteps)
        state_embedding = self.obs_embed(states) + time_embedding
        action_embedding = self.act_embed(actions) + time_embedding
        return_embedding = self.ret_embed(returns_to_go) + time_embedding
        stacked_input = torch.stack([return_embedding, state_embedding, action_embedding], dim=2).reshape(B, 3*L, state_embedding.shape(-1))
        out = super().forward(
            stacked_input, 
            hiddens, 
            cell_states, 
            do_embedding=False
        )
        out = self.action_head(out[:, 1::3])
        return out
        
        