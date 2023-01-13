import os
from abc import ABC, abstractmethod
from datetime import datetime
from functools import partial
from typing import Optional, Sequence, Union, Dict, Any

import torch
import pickle
import numpy as np

def make_unique_name(name):
    now = datetime.now()
    suffix = now.strftime("%m-%d-%H-%M")
    pid_str = os.getpid()
    if name == "":
        return f"{suffix}-{pid_str}"
    else:
        return f"{name}-{suffix}-{pid_str}"

def save_fn(protocol: str="torch"):
    def pickle_save(obj, file):
        with open(file, "w") as fp:
            pickle.dump(obj, fp)
            
    def numpy_save(obj, file):
        with open(file, "w") as fp:
            np.save(fp, obj)
            
    return {
        "torch": torch.save, 
        "pickle": pickle_save,
        "numpy": numpy_save 
    }.get(protocol)
    
def load_fn(protocol: str="torch"):
    def torch_load(file):
        return torch.load(file, map_location="cpu")
    
    def pickle_load(file):
        with open(file, "r") as fp:
            ret = pickle.load(file)
        return ret
    
    def numpy_load(file):
        with open(file, "r") as fp:
            ret = np.load(file)
        return ret
    
    return {
        "torch": torch_load, 
        "pickle": pickle_load, 
        "numpy": numpy_load
    }.get(protocol)
        

class LogLevel:
    NOTSET = 0
    DEBUG = 1
    WARNING = 2
    ERROR = 3
    INFO = 4


class BaseLogger(ABC):
    def __init__(self, activate=True, level: int=LogLevel.WARNING):
        self.level = level
        self.activate = activate
        
        