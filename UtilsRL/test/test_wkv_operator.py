import torch
from torch.utils.cpp_extension import load

T_MAX = 1024
wkv_cuda = load(name="wkv", sources=["UtilsRL/operator/wkv_op.cpp", "UtilsRL/operator/wkv_op.cu"], verbose=True, extra_cuda_cflags=['-res-usage', '--maxrregcount 60', '--use_fast_math', '-O3', '-Xptxas -O3', f'-DTmax={T_MAX}'], build_directory="build")
