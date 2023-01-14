import os
import numpy as np
from operator import itemgetter

from typing import Dict as DictLike
from typing import Optional, Any, Union, Sequence

from UtilsRL.logger.base_logger import make_unique_name, LogLevel
from UtilsRL.logger.base_logger import BaseLogger
from UtilsRL.logger.tensorboard_logger import TensorboardLogger
from UtilsRL.logger.text_logger import ColoredLogger, FileLogger
from UtilsRL.logger.wandb_logger import WandbLogger


numpy_compatible = np.ndarray
try:
    import torch
    numpy_compatible = torch.Tensor
except ImportError:
    pass


class CompositeLogger(BaseLogger):
    logger_registry = {
        "ColoredLogger": ColoredLogger, 
        "FileLogger": FileLogger, 
        "TensorboardLogger": TensorboardLogger, 
        "WandbLogger": WandbLogger, 
    }
    logger_default_args = {
        "ColoredLogger": {"activate": True}, 
    }
    def __init__(self, 
                 log_path: str, 
                 name: str, 
                 loggers_config: DictLike={}, 
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
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)
            
        self.loggers = dict()
        self.loggers_config = dict()
        self.loggers_cls = set()
        for _logger_cls in self.logger_registry:
            # iterate over all registered logger classes
            _config = self.logger_default_args.get(_logger_cls, {})
            _config.update(loggers_config.get(_logger_cls, {}))
            self.loggers_config[_logger_cls] = _config
            if _config.get("activate", True) == False:
                # if activate is designated as false, continue
                continue
            self.loggers[_logger_cls] = self.logger_registry[_logger_cls](
                log_path=log_path, name=name, unique_name=self.unique_name, 
                **_config, 
            )
            self.loggers_cls.add(_logger_cls)
        
    def __getattr__(self, __name: str):
        # if the method does not exist for CompositeLogger
        for _logger_cls, _logger in self.loggers.items():
            if hasattr(_logger, __name):
                return _logger.__getattribute__(__name)

    def _call_by_group(self, func: str, group: list, *args, **kwargs):
        return {
            _logger_cls: getattr(self.loggers[_logger_cls], func)(*args, **kwargs)\
                for _logger_cls in group if _logger_cls in self.loggers_cls
        }
    
    def info(self, msg: str, level: int=LogLevel.WARNING):
        return self._call_by_group(
            func="info", 
            group=["ColoredLogger", "FileLogger"], 
            msg=msg, level=level
        )
        
    def debug(self, msg: str, level: int=LogLevel.WARNING):
        return self._call_by_group(
            func="debug", 
            group=["ColoredLogger", "FileLogger"], 
            msg=msg, level=level
        )
    
    def warning(self, msg: str, level: int=LogLevel.WARNING):
        return self._call_by_group(
            func="warning", 
            group=["ColoredLogger", "FileLogger"], 
            msg=msg, level=level
        )

    def error(self, msg: str, level: int=LogLevel.WARNING):
        return self._call_by_group(
            func="error", 
            group=["ColoredLogger", "FileLogger"], 
            msg=msg, level=level
        )
        
    def log_scalar(
        self, 
        tag: str, 
        value: Union[float, numpy_compatible], 
        step: Optional[int] = None):
        
        return self._call_by_group(
            func="log_scalar", 
            group=["TensorboardLogger", "WandbLogger"], 
            tag=tag, value=value, step=step
        )
        
    def log_scalars(
        self, 
        main_tag: str, 
        tag_scalar_dict: DictLike[str, Union[float, int, numpy_compatible]], 
        step: Optional[int]=None):
        
        return self._call_by_group(
            func="log_scalars", 
            group=["TensorboardLogger", "WandbLogger"], 
            main_tag=main_tag, tag_scalar_dict=tag_scalar_dict, step=step
        )
        
    def log_image(self, 
                  tag: str, 
                  img_tensor: numpy_compatible, 
                  step: Optional[int]=None, 
                  dataformat: str="CHW"):
        
        return self._call_by_group(
            func="log_image", 
            group=["TensorboardLogger"], 
            tag=tag, img_tensor=img_tensor, step=step, dataformat=dataformat
        )
        
    def log_video(
        self, 
        tag: str, 
        vid_tensor: numpy_compatible, 
        step: Optional[int] = None, 
        fps: Optional[Union[int, float]] = 4, 
        dataformat: Optional[str] = "NTCHW"):
        
        return self._call_by_group(
            func="log_video", 
            group=["TensorboardLogger"], 
            tag=tag, vid_tensor=vid_tensor, step=step, fps=fps, dataformat=dataformat
        )
        
    def log_object(self, 
                   name: str, 
                   object: Any, 
                   path: Optional[str]=None):
        
        return self._call_by_group(
            func="log_object",
            group=["TensorboardLogger"], 
            name=name, object=object, path=path
        )
        