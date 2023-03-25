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
            y = y
        elif os.environ['RWKV_FLOAT_MODE'] == 'fp16':
            y = y.half()
        elif os.environ['RWKV_FLOAT_MODE'] == 'bf16':
            y = y.bfloat16()
        return y
    
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
        self.model_type = model_type
        if layer_id == 0:
            self.ln0 = nn.LayerNorm(self.embed_dim)
        if layer_id == 0 and self.model_type == "ffnpre":
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
        if self.layer_id == 0 and self.model_type == "ffnpre":
            residual = self.ln0(input)
        else:
            residual = input
        input = self.ln1(residual)
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
        seq_len: int,
        weight_decay: Optional[bool] = False
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.weight_decay = weight_decay
        self.input_embed = nn.Linear(input_dim, embed_dim)
        self.seq_len = seq_len
        
        self.blocks = nn.ModuleList([
            RWKVBlock(
                embed_dim=embed_dim,
                layer_id=i
            ) for i in range(num_layers)
        ])
        self.out_ln = nn.LayerNorm(embed_dim)
    
    def init_params(self):
        for mn, m in self.named_modules():
            if not isinstance(m, (nn.Linear, nn.Embedding)):
                continue
            for pn, p in m.named_parameters():
                if id(m.weight) == id(p):
                    break
            p_shape = m.weight.data.shape
            gain = 1.0
            scale = 1.0
            if isinstance(m, nn.Embedding):
                gain = math.sqrt(max(p_shape[0], p_shape[1]))
                if (p_shape[0] == self.input_dim) and (p_shape[1] == self.embed_dim):
                    scale = 1e-4
                else:
                    scale = 0
            if isinstance(m, nn.Linear):
                if m.bias is not None:
                    m.bias.data.zero_()
                if p_shape[0] > p_shape[1]:
                    gain = math.sqrt(p_shape[0] / p_shape[1])
                if (p_shape[0] == self.input_dim) and (p_shape[1] == self.embed_dim):
                    scale = 0.5

            if hasattr(m, "scale_init"):
                scale = m.scale_init

            gain *= scale
            if gain == 0:
                nn.init.zeros_(m.weight)
            elif gain > 0:
                nn.init.orthogonal_(m.weight, gain=gain)
            else:
                nn.init.normal_(m.weight, mean=0.0, std=-scale)
    
    def configure_params(self):
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.MultiheadAttention)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
                if self.weight_decay:
                    if pn.endswith('bias'):
                        no_decay.add(fpn)
                    elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                        decay.add(fpn)
                    elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                        no_decay.add(fpn)
                else:
                    no_decay.add(fpn)

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        return [param_dict[pn] for pn in sorted(list(decay))], [param_dict[pn] for pn in sorted(list(no_decay))]

    def forward(
        self,
        inputs: torch.Tensor,
        do_embedding: bool=True
    ) -> torch.Tensor:
        B, T, *_ = inputs.size()
        assert T <= self.seq_len, "len(input) too long"
        if do_embedding:
            x = self.emeb(inputs)
        for _, block in enumerate(self.blocks):
            x = block(x)
        x = self.out_ln(x)
        return x


class DecisionRWKV(RWKV):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        embed_dim: int,
        num_layers: int,
        seq_len: int,
        weight_decay: Optional[bool] = False
    ) -> None:
        super().__init__(
            input_dim=embed_dim,
            embed_dim=embed_dim,
            num_layers=num_layers,
            seq_len=seq_len,
            weight_decay=weight_decay
        )

        self.obs_embed = nn.Lineare(obs_dim, embed_dim)
        self.act_embed = nn.Linear(action_dim, embed_dim)
        self.ret_embed = nn.Linear(1, embed_dim)

        self.action_head = nn.Sequential(nn.Linear(embed_dim, action_dim), nn.Tanh())

    def forward(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        returns_to_go: torch.Tensor
    ):
        B, T, *_ = states.shape
        state_embedding = self.obs_embed(states)
        action_embedding = self.act_embed(actions)
        return_embedding = self.ret_embed(returns_to_go)
        
        stacked_input = torch.stack([return_embedding, state_embedding, action_embedding], dim=2).reshape(B, 3*T, state_embedding.shape[-1])
        out = super().forward(
            inputs=stacked_input, 
            do_embedding=False
        )
        out = self.action_head(out[:, 1::3])
        return out
