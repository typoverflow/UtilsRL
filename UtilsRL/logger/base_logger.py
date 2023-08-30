from typing import Any, Dict, Optional, Sequence, Union

import os
import pickle
from abc import ABC, abstractmethod
from datetime import datetime

import numpy as np
import torch


def make_unique_name(name):
    name = name or ""
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
    
def fmt_time_now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class LogLevel:
    NOTSET = 0
    DEBUG = 1
    WARNING = 2
    ERROR = 3
    INFO = 4


class BaseLogger():
    """
    Base class for loggers, providing basic string logging. 
    """
    
    cmap = {
        None: "\033[0m",  
        "error": "\033[1;31m", 
        "debug": "\033[0m", 
        "warning": "\033[1;33m",
        "info": "\033[1;34m", 
        "reset": "\033[0m", 
    }
    
    def __init__(
        self, 
        log_dir: str, 
        name: Optional[str]=None, 
        unique_name: Optional[str]=None, 
        activate: bool=True, 
        level: int=LogLevel.WARNING, 
        *args, **kwargs
    ):
        if unique_name:
            self.unique_name = unique_name
        else:
            self.unique_name = make_unique_name(name)
        self.log_dir = os.path.join(log_dir, self.unique_name)
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        self.activate = activate
        self.level = level
        
    def can_log(self, level):
        return self.activate and level >= self.level
        
    def info(self, msg: str, level: int=LogLevel.INFO):
        if self.can_log(level):
            time_str = fmt_time_now()
            print("{}[{}]{}\t{}".format(
                self.cmap["info"], 
                time_str, 
                self.cmap["reset"], 
                msg
            ))

    def debug(self, msg: str, level: int=LogLevel.DEBUG):
        if self.can_log(level):
            time_str = fmt_time_now()
            print("{}[{}]{}\t{}".format(
                self.cmap["debug"], 
                time_str, 
                self.cmap["reset"], 
                msg
            ))

    def warning(self, msg: str, level: int=LogLevel.WARNING):
        if self.can_log(level):
            time_str = fmt_time_now()
            print("{}[{}]{}\t{}".format(
                self.cmap["warning"], 
                time_str, 
                self.cmap["reset"], 
                msg
            ))
            
    def error(self, msg: str, level: int=LogLevel.ERROR):
        if self.can_log(level):
            time_str = fmt_time_now()
            print("{}[{}]{}\t{}".format(
                self.cmap["error"], 
                time_str, 
                self.cmap["reset"], 
                msg
            ))

    def log_str(self, msg: str, level: Optional[str]=None, *args, **kwargs):
        if level: level = level.lower()
        level = {
            None: LogLevel.DEBUG, 
            "error": LogLevel.ERROR, 
            "log": LogLevel.INFO, 
            "warning": LogLevel.WARNING, 
            "debug": LogLevel.DEBUG
        }.get(level)
        if self.can_log(level):
            time_str = fmt_time_now()
            print("{}[{}]{}\t{}".format(
                self.cmap[level], 
                time_str, 
                self.cmap["reset"], 
                msg
            ))         
        
