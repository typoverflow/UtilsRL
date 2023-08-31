from typing import Any, List, Optional, Sequence, Union
from typing import Dict as DictLike

import os
import numpy as np
from UtilsRL.logger.base_logger import (
    BaseLogger, 
    LogLevel, 
    load_fn, 
    save_fn, 
)


class TensorboardLogger(BaseLogger):
    """
    Tensorboard Logger
    
    Parameters
    ----------
    log_dir :  The base dir where the logger logs to.
    name :  The name of the experiment, will be used to construct the event file name. A suffix 
            will be added to the name to ensure the uniqueness of the log dir. 
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
        from torch.utils.tensorboard.writer import SummaryWriter
        self.tb_dir = os.path.join(self.log_dir, "tb")
        os.makedirs(self.tb_dir, exist_ok=True)
        self.output_dir = self.tb_dir
        self.tb_writer = SummaryWriter(self.tb_dir)

    def log_scalar(
        self, 
        tag: str, 
        value: Union[float, int], 
        step: Optional[int]=None
    ):
        """Add scalar to tensorboard summary.
        
        tag :  the identifier of the scalar.
        value :  value to record.
        step :  global timestep of the scalar. 
        """
        if not self.can_log():
            return
        self.tb_writer.add_scalar(tag, value, step)
        
    def log_scalars(
        self,
        main_tag: str, 
        tag_scalar_dict: DictLike[str, Union[float, int]], 
        step: Optional[int]=None
    ):
        """Add scalars which share the main tag to tensorboard summary.
        
        main_tag :  the shared main tag of the scalars, can be a null string.
        tag_scalar_dict :  a dictionary of tag and value.
        step :  global timestep of the scalars.
        """
        if not self.can_log():
            return
        if main_tag is None or main_tag == "":
            main_tag = ""
        else:
            main_tag = main_tag+"/"
            
        for tag, value in tag_scalar_dict.items():
            self.tb_writer.add_scalar(main_tag+tag, value, step)

    def log_image(
        self, 
        tag: str, 
        img_tensor: Any, 
        step: Optional[int]=None, 
        dataformat: str="CHW"
    ):
        """Add image to tensorboard summary. Note that this requires ``pillow`` package. 
        
        :param tag: the identifier of the image.
        :param img_tensor: an `uint8` or `float` Tensor of shape `
                [channel, height, width]` where `channel` is 1, 3, or 4.
                The elements in img_tensor can either have values
                in [0, 1] (float32) or [0, 255] (uint8).
                Users are responsible to scale the data in the correct range/type.
        :param global_step: global step. 
        :param dataformats: This parameter specifies the meaning of each dimension of the input tensor.
        """
        if not self.can_log():
            return
        self.tb_writer.add_image(tag, img_tensor, step, dataformats=dataformat)
        
    def log_video(
        self, 
        tag: str, 
        vid_tensor: Any, 
        step: Optional[int]=None, 
        fps: Optional[Union[float, int]]=4, 
    ):
        """Add a piece of video to tensorboard summary. Note that this requires ``moviepy`` package.

        :param tag: the identifier of the video.
        :param vid_tensor: video data. 
        :param global_step: global step.
        :param fps: frames per second.
        :param dataformat: specify different permutation of the video tensor.
        """
        if not self.can_log():
            return
        self.tb_writer.add_video(tag, vid_tensor, step, fps)

    def log_histogram(
        self, 
        tag: str, 
        values: Union[np.ndarray, List], 
        step: Optional[int]=None, 
    ):
        """Add histogram to tensorboard. 
        
        :param tag: the identifier of the histogram.
        :param values: the values, should be list or np.ndarray. 
        :param global_step: global step.
        """
        if not self.can_log():
            return
        self.tb_writer.add_histogram(tag, np.asarray(values), step)

    def log_object(
        self,
        name: str, 
        object: Any, 
        path: Optional[str]=None, 
        protocol: str="torch"
    ):
        """Save a Python object to the given path.
        
        name :  the identifier of the object.
        object :  the object to save.
        path :  the path to save the object, will be created if not exist; will be set to `self.tb_dir` if None.
        """
        if not self.can_log():
            return None
        if path is None:
            path = self.output_dir
        else:
            os.makedirs(path, exist_ok=True)
        save_path = os.path.join(path, name)
        save_fn(protocol)(object, save_path)
        return save_path
    
    def load_object(
        self, 
        name: str, 
        path: Optional[str]=None, 
        protocol="torch"
    ):
        """Load a Python object from the given path.
        
        name :  the identifier of the object.
        path :  the path from which to load the object, default to `self.tb_dir` if None.
        """
        if not self.can_log():
            return None
        if path is None:
            path = self.output_dir
        return load_fn(protocol)(os.path.join(path, name))
    
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.tb_writer.close()
        
