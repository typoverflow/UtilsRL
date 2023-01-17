import os
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
    
def wandb_sync_run(root_path: str, *args):
    import wandb
    from UtilsRL.logger import logger
    
    if not os.path.exists(root_path):
        raise FileNotFoundError(f"wandb_sync_offline: path {root_path} not found.")
    
    def recursive_upload(path):
        if os.path.isfile(path):
            return
        for sub_path in os.listdir(path):
            full_path = os.path.join(path, sub_path)
            if sub_path == "wandb":
                final_path = None
                for sub_sub_path in os.listdir(full_path):
                    if sub_sub_path.startswith("run-") or sub_sub_path.startswith("offline-run-"):
                        final_path = os.path.join(full_path, sub_sub_path)
                if final_path:
                    logger.warning(f"sync {final_path} to wandb ...")
                    os.system(f"wandb sync {final_path} {' '.join(args)}")
            else:
                recursive_upload(full_path)
    recursive_upload(root_path)
    
    