import os
import torch


def select_device(id=None):
    if not torch.cuda.is_available() or id=="cpu":
        return torch.device("cpu")
    try:
        id = int(id)
    except ValueError:
        try:
            id = int(id.split(":")[-1])
        except Exception:
            raise ValueError("Invalid cuda ID format: {}".format(id))
    
    if id < 0:
        return torch.device("cpu")
    else:
        ret =  torch.device(f"cuda:{id}")
        os.environ["CUDA_VISIBLE_DEVICES"] = str(id)