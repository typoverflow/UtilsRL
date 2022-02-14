import os
import torch


def select_device(id=None):
    """Selects the device according to given id.
    
    Args: 
        id: If None, then a free cuda will be selected. If "cpu" or gpu is 
            not available, then cpu device will be returned. Otherwise, check
            whether the given device is valid and return that device if so.
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
    