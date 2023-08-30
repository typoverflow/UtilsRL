from typing import Any, List, Optional, Sequence, Union
from typing import Dict as DictLike

import os
import numpy as np
import torch
from UtilsRL.logger.base_logger import (
    BaseLogger, 
    LogLevel, 
    fmt_time_now,
    load_fn, 
    save_fn, 
    make_unique_name,
)


class StdoutBackupLogger(BaseLogger):
    """
    StdOut Backup Logger, which in the meantime backups the stdout to files. 
    
    Parameters
    ----------
    log_dir :  The base dir where the logger logs to. 
    name :  The name of the experiment, will be used to construct the event file name. A suffix 
            will be added to the name to ensure the uniqueness of the log path. 
    unique_name :  The name of the experiment, but no suffix will be appended. 
    activate :  Whether this logger is activated.
    level :  The level threshold of the logging message. 
    """
    def __init__(
        self, 
        log_dir: str, 
        name: Optional[str]=None, 
        unique_name: Optional[str]=None, 
        activate: bool=True, 
        level=LogLevel.WARNING, 
        *args, **kwargs
    ):
        super().__init__(log_dir, name, unique_name, activate, level)
        if not self.activate:
            return
        self.stdout_dir = os.path.join(self.log_dir, "stdout")
        self.stdout_file = os.path.join(self.stdout_dir, "output.txt")
        if not os.path.exists(self.stdout_dir):
            os.makedirs(self.stdout_dir)
        self.stdout_fp = open(self.stdout_file, "w+")

    def _write(
        self, 
        time_str: str, 
        msg: str, 
        type="info"
    ):
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
        self._write(msg, level, "info")
        
    def debug(self, msg: str, level: int=LogLevel.DEBUG):
        self._write(msg, level, "debug")
        
    def warning(self, msg: str, level: int=LogLevel.WARNING):
        self._write(msg, level, "warning")
        
    def error(self, msg: str, level: int=LogLevel.ERROR):
        self._write(msg, level, "error")