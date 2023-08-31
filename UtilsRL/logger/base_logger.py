from typing import Any, Dict, Optional, Sequence, Union

import os
import torch
import pickle
from datetime import datetime

import numpy as np
from UtilsRL.misc.namespace import NameSpaceMeta

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

cmap = {
    None: "\033[0m",  
    "error": "\033[1;31m", 
    "debug": "\033[0m", 
    "warning": "\033[1;33m",
    "info": "\033[1;34m", 
    "reset": "\033[0m", 
}

def log(msg: str, type: str):
    time_str = fmt_time_now()
    print("{}[{}]{}\t{}".format(
        cmap.get(type.lower(), "\033[0m"), 
        time_str, 
        cmap["reset"], 
        msg
    )) 


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
        backup_stdout: bool=False, 
        activate: bool=True, 
        level: int=LogLevel.WARNING, 
        *args, **kwargs
    ):
        self.activate = activate
        if not self.activate:
            return
        if unique_name:
            self.unique_name = unique_name
        else:
            self.unique_name = make_unique_name(name)
        self.log_dir = os.path.join(log_dir, self.unique_name)
        os.makedirs(self.log_dir, exist_ok=True)
            
        self.backup_stdout = backup_stdout
        if self.backup_stdout:
            self.stdout_file = os.path.join(self.log_dir, "stdout.txt")
            self.stdout_fp = open(self.stdout_file, "w+")
        self.level = level
        
    def can_log(self, level=LogLevel.INFO):
        return self.activate and level >= self.level
    
    def _write(self, time_str: str, msg: str, type="info"):
        type = type.upper()
        self.stdout_fp.write("[{}] ({})\t{}\n".format(
            time_str, 
            type, 
            msg
        ))
        self.stdout_fp.flush()
        
    def info(self, msg: str, level: int=LogLevel.INFO):
        if self.can_log(level):
            time_str = fmt_time_now()
            print("{}[{}]{}\t{}".format(
                self.cmap["info"], 
                time_str, 
                self.cmap["reset"], 
                msg
            ))
            if self.backup_stdout:
                self._write(time_str, msg, "info")
            

    def debug(self, msg: str, level: int=LogLevel.DEBUG):
        if self.can_log(level):
            time_str = fmt_time_now()
            print("{}[{}]{}\t{}".format(
                self.cmap["debug"], 
                time_str, 
                self.cmap["reset"], 
                msg
            ))
            if self.backup_stdout:
                self._write(time_str, msg, "debug")

    def warning(self, msg: str, level: int=LogLevel.WARNING):
        if self.can_log(level):
            time_str = fmt_time_now()
            print("{}[{}]{}\t{}".format(
                self.cmap["warning"], 
                time_str, 
                self.cmap["reset"], 
                msg
            ))
            if self.backup_stdout:
                self._write(time_str, msg, "warning")
            
    def error(self, msg: str, level: int=LogLevel.ERROR):
        if self.can_log(level):
            time_str = fmt_time_now()
            print("{}[{}]{}\t{}".format(
                self.cmap["error"], 
                time_str, 
                self.cmap["reset"], 
                msg
            ))
            if self.backup_stdout:
                self._write(time_str, msg, "error")
    
    def log_config(self, config: Dict):
        if self.can_log():
            with open(os.path.join(self.log_dir, "config.txt"), "w") as fp:
                if isinstance(config, NameSpaceMeta):
                    config_str = str(config)
                else:
                    if not isinstance(config, dict):
                        config = config.as_dict()
                    import json
                    config_str = json.dumps(config, sort_keys=True, indent=2)
                fp.write(config_str)

    def log_str(self, msg: str, type: Optional[str]=None, *args, **kwargs):
        if type: type = type.lower()
        level = {
            None: LogLevel.DEBUG, 
            "error": LogLevel.ERROR, 
            "log": LogLevel.INFO, 
            "warning": LogLevel.WARNING, 
            "debug": LogLevel.DEBUG
        }.get(type)
        if self.can_log(level):
            time_str = fmt_time_now()
            print("{}[{}]{}\t{}".format(
                self.cmap[type], 
                time_str, 
                self.cmap["reset"], 
                msg
            ))      
            if self.backup_stdout:
                self._write(time_str, msg, type)
    
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if hasattr(self, "stdout_fp"):
            self.stdout_fp.close()
           
