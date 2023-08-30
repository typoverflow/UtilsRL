from .base_logger import BaseLogger, fmt_time_now
from .tensorboard_logger import TensorboardLogger
from .wandb_logger import WandbLogger
from .composite_logger import CompositeLogger

cmap = {
    None: "\033[0m",  
    "error": "\033[1;31m", 
    "debug": "\033[0m", 
    "warning": "\033[1;33m",
    "info": "\033[1;34m", 
    "reset": "\033[0m", 
}

class logger:
    cmap = {
        None: "\033[0m",  
        "error": "\033[1;31m", 
        "debug": "\033[0m", 
        "warning": "\033[1;33m",
        "info": "\033[1;34m", 
        "reset": "\033[0m", 
    }
    
    @staticmethod
    def info(msg):
        time_str = fmt_time_now()
        print("{}[{}]{}\t{}".format(
            cmap["info"], 
            time_str, 
            cmap["reset"], 
            msg
        ))

    @staticmethod
    def debug(msg):
        time_str = fmt_time_now()
        print("{}[{}]{}\t{}".format(
            cmap["debug"], 
            time_str, 
            cmap["reset"], 
            msg
        ))
        
    @staticmethod
    def warning(msg):
        time_str = fmt_time_now()
        print("{}[{}]{}\t{}".format(
            cmap["warning"], 
            time_str, 
            cmap["reset"], 
            msg
        ))

    @staticmethod
    def error(msg):
        time_str = fmt_time_now()
        print("{}[{}]{}\t{}".format(
            cmap["error"], 
            time_str, 
            cmap["reset"], 
            msg
        ))
        
    @staticmethod
    def log_str(msg: str, type=None):
        if type: type = type.lower()
        time_str = fmt_time_now()
        print("{}[{}]{}\t{}".format(
            cmap[type], 
            time_str, 
            cmap["reset"], 
            msg
        ))      

