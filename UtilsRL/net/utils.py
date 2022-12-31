import torch.nn as nn

from UtilsRL.net.basic import NoisyLinear

def reset_noise_layer(m: nn.Module, noisy_layer=NoisyLinear):
    modules = [module for module in m.modules() if isinstance(module, noisy_layer)]
    for module in modules:
        module.reset_noise()

def conv2d_out_size(in_size, kernel_size, stride):
    return (in_size - (kernel_size - 1) - 1) // stride + 1