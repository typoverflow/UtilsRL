from typing import Any, List, Optional, Sequence, Union
from typing import Dict as DictLike

import numpy as np
from UtilsRL.logger.base_logger import (
    BaseLogger, 
    LogLevel, 
)
from UtilsRL.logger.tensorboard_logger import TensorboardLogger
from UtilsRL.logger.wandb_logger import WandbLogger
from UtilsRL.logger.csv_logger import CsvLogger


class CompositeLogger(BaseLogger):
    """
    Composite Logger, which composes multiple logger implementations to provide a unified
    interface for various types of log recording. 
    
    Parameters
    ----------
    log_dir :  The base dir where the logger logs to.
    name :  The name of the experiment, will be used to construct the event file name. A suffix 
            will be added to the name to ensure the uniqueness of the log dir. 
    logger_config :  The ad-hoc configs for each component logger. 
    unique_name :  The name of the experiment, but no suffix will be appended. 
    backup_stdout :  Whether or not backup stdout to files. 
    activate :  Whether this logger is activated.
    level :  The level threshold of the logging message. 
    """
    logger_registry = {
        "CsvLogger": CsvLogger, 
        "TensorboardLogger": TensorboardLogger, 
        "WandbLogger": WandbLogger, 
    }
    def __init__(
        self, 
        log_dir: str, 
        name: Optional[str]=None, 
        logger_config: DictLike={}, 
        unique_name: Optional[str]=None, 
        backup_stdout: bool=False, 
        activate: bool=True, 
        level: int=LogLevel.WARNING, 
        *args, **kwargs
    ):
        super().__init__(log_dir, name, unique_name, backup_stdout, activate, level)
        if not self.activate:
            return
        # create loggers
        default_config = {
            "log_dir": log_dir, 
            "name": name, 
            "unique_name": unique_name, 
            "backup_stdout": False,
            "activate": activate, 
            "level": level
        }
        self.loggers = dict()
        self.logger_config = dict()
        self.logger_cls = set()
        for logger_cls in logger_config:
            config = default_config.copy()
            config.update(logger_config[logger_cls])
            config["backup_stdout"] = False # force sub loggers not to backup to avoid multiple file handles
            self.logger_config[logger_cls] = config
            # print(logger_cls, config)
            if config.get("activate", True) == False:
                continue
            self.loggers[logger_cls] = self.logger_registry[logger_cls](
                **config
            )
            self.logger_cls.add(logger_cls)
        
    # def __getattr__(self, __name: str):
    #     # if the method does not exist for CompositeLogger
    #     for _logger_cls, _logger in self.loggers.items():
    #         if hasattr(_logger, __name):
    #             return _logger.__getattribute__(__name)
    #     else:
    #         raise AttributeError(f"CompositeLogger, as well as its components {self.logger_cls}, does not have attribute {str(__name)}.")

    def _try_call_by_group(self, func: str, group: list, *args, **kwargs):
        if self.can_log():
            return {
                _logger_cls: getattr(self.loggers[_logger_cls], func)(*args, **kwargs)\
                    for _logger_cls in group if _logger_cls in self.logger_cls
            }
        
    def log_scalar(
        self, 
        tag: str, 
        value: Union[float, int], 
        step: Optional[int] = None
    ):
        return self._try_call_by_group(
            func="log_scalar", 
            group=["TensorboardLogger", "WandbLogger"], 
            tag=tag, value=value, step=step
        )
        
    def log_scalars(
        self, 
        main_tag: str, 
        tag_scalar_dict: DictLike[str, Union[float, int]], 
        step: Optional[int]=None
    ):
        return self._try_call_by_group(
            func="log_scalars", 
            group=["TensorboardLogger", "WandbLogger"], 
            main_tag=main_tag, tag_scalar_dict=tag_scalar_dict, step=step
        )
        
    def log_image(
        self, 
        tag: str, 
        img_tensor: Any, 
        step: Optional[int]=None, 
        dataformat: str="CHW"
    ):
        return self._try_call_by_group(
            func="log_image", 
            group=["TensorboardLogger"], 
            tag=tag, img_tensor=img_tensor, step=step, dataformat=dataformat
        )
        
    def log_video(
        self, 
        tag: str, 
        vid_tensor: Any, 
        step: Optional[int] = None, 
        fps: Optional[Union[float, int]] = 4, 
    ):
        return self._try_call_by_group(
            func="log_video", 
            group=["TensorboardLogger"], 
            tag=tag, vid_tensor=vid_tensor, step=step, fps=fps
        )
        
    def log_histogram(
        self, 
        tag: str, 
        values: Union[np.ndarray, List],
        step: Optional[int]=None, 
    ):
        return self._try_call_by_group(
            func="log_histogram", 
            group=["TensorboardLogger"], 
            tag=tag, values=values, step=step
        )
        
    def log_object(
        self, 
        name: str, 
        object: Any, 
        path: Optional[str]=None, 
        protocol: str="torch"
    ):
        return self._try_call_by_group(
            func="log_object",
            group=["TensorboardLogger"], 
            name=name, object=object, path=path, protocol=protocol
        )
        
