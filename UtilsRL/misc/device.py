import os
from time import get_clock_info
import torch


def select_device(id=None):
    """Selects the device according to given ID. ID can be:
        - (if torch.cuda.is_available() is False, then cpu will be selected anyway.)
        - None, then a most free GPU will be selected for use. 
        - "cpu", then cpu will be returned.
        - int or "x" or "cuda:x", then gpu with corresponding ID will be 
            selected unless `id` or `x` < 0. 
    """
    
    if isinstance(id, torch.device):
        return id
    if not torch.cuda.is_available() or id == "cpu":
        return torch.device("cpu")
    if id is None:
        return torch.device("cuda:{}".format(select_free_cuda()))
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
        return ret

def select_free_cuda():
    def get_volume(t):
        return np.asarray([int(i.split()[2]) for i in t])

    try: 
        import numpy as np
        cmd1 = os.popen("nvidia-smi -q -d Memory | grep -A4 GPU | grep Total")
        total = cmd1.readlines()
        cmd1.close()
        
        cmd2 = os.popen("nvidia-smi -q -d Memory | grep -A4 GPU | grep Reserved")
        reserved = cmd2.readlines()
        cmd2.close()
        
        cmd3 = os.popen("nvidia-smi -q -d Memory | grep -A4 GPU | grep Used")
        used = cmd3.readlines()
        cmd3.close()
        
        def get_volume(t):
            return np.asarray([int(i.split()[2]) for i in t])
        
        total, used, reserved = get_volume(total), get_volume(used), get_volume(reserved)
        return np.argmax(total-used-reserved)
    except Exception as e:
        cmd4 = os.popen("nvidia-smi -q -d Memory | grep -A4 GPU | grep Free")
        free = cmd4.readlines()
        cmd4.close()
        
        free = get_volume(free)
        return np.argmax(free)
        
    