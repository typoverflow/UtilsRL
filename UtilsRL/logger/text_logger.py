import os
from datetime import datetime
from typing import Optional

from UtilsRL.logger.base_logger import BaseLogger
from UtilsRL.logger.base_logger import LogLevel, make_unique_name


class ColoredLogger(BaseLogger):
    """Create a logger which prints msg to terminal with ansi colors. 
    """
    cmap = {
        "error": "\033[1;31m", 
        "debug": "\033[0m", 
        "warning": "\033[1;33m",
        "info": "\033[1;34m", 
        "reset": "\033[0m", 
    }
    
    def __init__(self, activate:bool=True, level: int=LogLevel.WARNING, *args, **kwargs):
        super().__init__(activate, level)
        
    def info(self, msg: str, level: int=LogLevel.INFO):
        if not self.activate or level < self.level:
            return
        time_fmt = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print("{}[{}]{}\t{}".format(self.cmap["info"], time_fmt, self.cmap["reset"], msg))
        
    def debug(self, msg: str, level: int=LogLevel.DEBUG):
        if not self.activate or level < self.level:
            return
        time_fmt = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print("{}[{}]{}\t{}".format(self.cmap["debug"], time_fmt, self.cmap["reset"], msg))
        
    def warning(self, msg: str, level: int=LogLevel.WARNING):
        if not self.activate or level < self.level:
            return
        time_fmt = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print("{}[{}]{}\t{}".format(self.cmap["warning"], time_fmt, self.cmap["reset"], msg))

    def error(self, msg: str, level: int=LogLevel.ERROR):
        if not self.activate or level < self.level:
            return
        time_fmt = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print("{}[{}]{}\t{}".format(self.cmap["error"], time_fmt, self.cmap["reset"], msg))


class FileLogger(BaseLogger):
    def __init__(self, 
                 log_path: str, 
                 name: str, 
                 unique_name: Optional[str]=None, 
                 activate: bool=True, 
                 level: int=LogLevel.WARNING, 
                 *args, **kwargs):
        super().__init__(activate, level)
        if unique_name:
            self.unique_name = unique_name
        else:
            self.unique_name = make_unique_name(name)
        self.log_path = os.path.join(log_path, self.unique_name)
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        self.txt_file = os.path.join(log_path, self.unique_name, "output.txt")
        
    def _write(self, msg, level, type="info"):
        if not self.activate or level < self.level:
            return
        type = type.upper()
        time_fmt = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.txt_file, "a+") as fp:
            fp.write("[{}] ({})\t{}\n".format(time_fmt, type, msg))
        
    def info(self, msg: str, level: int=LogLevel.INFO):
        self._write(msg, level, "info")
        
    def debug(self, msg: str, level: int=LogLevel.DEBUG):
        self._write(msg, level, "debug")
        
    def warning(self, msg: str, level: int=LogLevel.WARNING):
        self._write(msg, level, "warning")
        
    def error(self, msg: str, level: int=LogLevel.ERROR):
        self._write(msg, level, "error")