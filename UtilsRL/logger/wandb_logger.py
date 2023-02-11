import os
import numpy as np

from typing import Dict as DictLike
from typing import Optional, Any, Union, Sequence

from UtilsRL.logger.base_logger import BaseLogger, make_unique_name, LogLevel, save_fn, load_fn
from UtilsRL.misc.namespace import NameSpaceMeta

numpy_compatible = np.ndarray
try:
    import torch
    numpy_compatible = torch.Tensor
except ImportError:
    pass

class WandbLogger(BaseLogger):
    def __init__(self, 
                 log_path: str, 
                 name: str, 
                 config: Optional[DictLike]=None, 
                 project: Optional[str]=None, 
                 entity: Optional[str]=None, 
                 unique_name: Optional[str]=None, 
                 activate: bool=True, 
                 level: int=LogLevel.WARNING, 
                 *args, **kwargs):
        super().__init__(activate, level)
        
        import wandb
        if unique_name:
            self.unique_name = unique_name
        else:
            self.unique_name = make_unique_name(name)
        self.log_path = os.path.join(log_path, self.unique_name)
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)
        if isinstance(config, NameSpaceMeta):
            config = config.as_dict()
        self.run = wandb.init(
            dir = self.log_path, 
            name = self.unique_name, 
            config = config, 
            project = project, 
            entity = entity, 
            **kwargs
        ) # this create the `self.log_path/wandb` dir
        self.output_path = os.path.join(self.log_path, "wandb")
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
            
    def log_scalar(self, tag: str, value: Union[float, int, numpy_compatible], step: Optional[int]=None):
        if not self.activate:
            return
        self.run.log({tag: value}, step=step)
        
    def log_scalars(
        self, 
        main_tag: str, 
        tag_scalar_dict: DictLike[str, Union[float, int, numpy_compatible]], 
        step: Optional[int]=None
    ):
        if not self.activate:
            return
        if main_tag is None or main_tag == "":
            main_tag = ""
        else:
            main_tag = main_tag + "/"
        tag_scalar_dict = {(main_tag+_key):_value for _key, _value in tag_scalar_dict.items()}
        self.run.log(tag_scalar_dict, step=step)

        
    def __enter__(self):
        return self
    
    def __exit__(self, exec_type, exc_val, exc_tb):
        self.run.finish()
