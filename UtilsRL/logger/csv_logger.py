from typing import Any, List, Optional, Sequence, Union
from typing import Dict as DictLike

import os
import numpy as np
from UtilsRL.logger.base_logger import (
    BaseLogger, 
    LogLevel, 
)


class CsvLogger(BaseLogger):
    """
    CSV Logger

    Parameters
    ----------
    log_dir :  The base dir where the logger logs to. 
    name :  The name of the experiment, will be used to construct the event file name. A suffix 
            will be added to the name to ensure the uniqueness of the log path. 
    unique_name :  The name of the experiment, but no suffix will be appended. 
    backup_stdout :  Whether or not backup stdout to files. 
    activate :  Whether this logger is activated.
    level :  The level threshold of the logging message. 
    """
    
    def __init__(
        self, 
        log_dir: str, 
        name: Optional[str]=None, 
        unique_name: Optional[str]=None, 
        backup_stdout: bool=False, 
        activate: bool=True, 
        level=LogLevel.WARNING, 
        *args, **kwargs
    ):
        super().__init__(log_dir, name, unique_name, backup_stdout, activate, level)
        if not self.activate:
            return
        self.csv_dir = os.path.join(self.log_dir, "csv")
        self.csv_file = os.path.join(self.csv_dir, "output.csv")
        if not os.path.exists(self.csv_dir):
            os.makedirs(self.csv_dir)
        self.csv_fp = open(self.csv_file, "w+")
        self.csv_sep = ","
        self.csv_keys = ["step"]
        
    def log_scalars(
        self, 
        main_tag: str, 
        tag_scalar_dict: DictLike[str, Union[float, int]], 
        step: Optional[int]=None
    ):
        """Add scalar to CSV file. 
        
        tag :  the identifier of the scalar. 
        value :  value to record. 
        step :  global timestep of the scalar. 
        """
        if not self.can_log():
            return
        if main_tag is None or main_tag == "":
            pass
        else:
            main_tag = main_tag + "/"
            tag_scalar_dict = {main_tag+tag:value for tag, value in tag_scalar_dict.items()}
        # handle new keys
        extra_keys = list(tag_scalar_dict.keys() - self.csv_keys)
        extra_keys.sort()
        if extra_keys:
            self.csv_keys.extend(extra_keys)
            self.csv_fp.seek(0)
            lines = self.csv_fp.readlines()
            self.csv_fp = open(self.csv_file, "w+t")
            self.csv_fp.seek(0)
            self.csv_fp.write(",".join(self.csv_keys)+"\n")
            for line in lines[1:]:
                self.csv_fp.write(line[:-1])
                self.csv_fp.write(self.csv_sep*len(extra_keys))
                self.csv_fp.write("\n")
            self.csv_fp = open(self.csv_file, "a+t")
        # write new entry
        values_to_write = [
            str(tag_scalar_dict.get(key, "")) if key != "step" else str(int(step))
                for key in self.csv_keys
        ]
        self.csv_fp.write(",".join(values_to_write)+"\n")
        self.csv_fp.flush()

    def log_scalar(
        self, 
        tag: str, 
        value: Union[float, int], 
        step: Optional[int]=None
    ):
        """Add scalar to CSV summary. 
        
        tag :  the identifier of the scalar. 
        value :  value to record. 
        step :  global timestep of the scalar. 
        """
        self.log_scalars(
            main_tag=None, 
            tag_scalar_dict={tag: value}, 
            step=step
        )
        
    def __enter__(self):
        return self
    
    def __exit__(self, exec_type, exc_val, exc_tb):
        self.csv_fp.close()
        
