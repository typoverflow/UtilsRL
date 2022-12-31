import torch
import numpy as np

def safe_eval(expr: str):
    if not isinstance(expr, str):
        raise TypeError("Expr for safe eval must be string.")
    try:
        ret = eval(expr)
    except Exception as e:
        ret = expr
    return ret

def convert_data_to_tensor(data):
    if isinstance(data, torch.Tensor):
        if len(data.shape) == 0:
            data = torch.unsqueeze(0)
    if isinstance(data, (int, float)):
        data = np.asarray([data, ])
    if len(data.shape) == 0:
        data = np.asarray([data, ])
    return torch.from_numpy(data)
    
def convert_data_to_numpy(data):
    return np.asarray(data)
    