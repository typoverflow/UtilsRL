from typing import Any, List, Optional, Sequence, Union
from typing import Dict as DictLike

import os
import numpy as np
from UtilsRL.logger.base_logger import (
    BaseLogger, 
    LogLevel, 
)
from UtilsRL.misc.namespace import NameSpaceMeta


class WandbLogger(BaseLogger):
    """
    WandB Logger
    
    Parameters
    ----------
    log_dir :  The base dir where the logger logs to.
    name :  The name of the experiment, will be used to construct the event file name. A suffix 
            will be added to the name to ensure the uniqueness of the log dir. 
    config :  The the configs or hyper-parameters of the experiment, should be dict-like. 
    project :  The project for wandb. 
    entity :  The entity for wandb. 
    unique_name :  The name of the experiment, but no suffix will be appended. 
    backup_stdout :  Whether or not backup stdout to files. 
    activate :  Whether this logger is activated.
    level :  The level threshold of the logging message. 
    """

    def __init__(
        self, 
        log_dir: str, 
        name: Optional[str]=None, 
        config: Optional[DictLike]=None, 
        project: Optional[str]=None, 
        entity: Optional[str]=None, 
        unique_name: Optional[str]=None, 
        backup_stdout: bool=False, 
        activate: bool=True, 
        level: int=LogLevel.WARNING, 
        *args, **kwargs
    ):
        super().__init__(log_dir, name, unique_name, backup_stdout, activate, level)
        if not self.activate:
            return
        import wandb
        if isinstance(config, NameSpaceMeta):
            config = config.as_dict()
        self.run = wandb.init(
            dir=self.log_dir, 
            name=self.unique_name, 
            config=config, 
            project=project, 
            entity=entity, 
            **kwargs
        ) # this create the `self.log_dir/wandb` dir
        # define the custom timestep metric
        self.run.define_metric("step")
        self.keys = set()
        
    def log_scalars(
        self, 
        main_tag: str, 
        tag_scalar_dict: DictLike[str, Union[float, int]], 
        step: Optional[int]=None
    ):
        if not self.can_log():
            return
        if main_tag is None or main_tag == "":
            pass
        else:
            main_tag = main_tag + "/"
            tag_scalar_dict = {main_tag+key:value for key, value in tag_scalar_dict.items()}
        # handle new keys
        extra_keys = set(tag_scalar_dict.keys()).difference(self.keys)
        for ek in extra_keys:
            self.run.define_metric(ek, step_metric="step")
        self.keys = self.keys.union(extra_keys)
        tag_scalar_dict["step"] = step
        self.run.log(tag_scalar_dict, step=None)
        
    def log_scalar(
        self, 
        tag: str, 
        value: float, 
        step: Optional[int]=None
    ):
        self.log_scalars(
            main_tag=None, 
            tag_scalar_dict={tag: value}, 
            step=step
        )
        
    def __enter__(self):
        return self
    
    def __exit__(self, exec_type, exc_val, exc_tb):
        self.run.finish()
        
