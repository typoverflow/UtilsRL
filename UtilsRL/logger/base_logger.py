import os
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional, Sequence, Union, Dict, Any

def make_unique_name(name):
    now = datetime.now()
    suffix = now.strftime("%m-%d-%H-%M")
    pid_str = os.getpid()
    if name == "":
        return f"{suffix}-{pid_str}"
    else:
        return f"{name}-{suffix}-{pid_str}"

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
        
        