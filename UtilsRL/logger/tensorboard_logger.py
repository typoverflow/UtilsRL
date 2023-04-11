import os
import pickle
import numpy as np
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict as DictLike
from typing import Optional, Sequence, Union, Any, List

from UtilsRL.logger.base_logger import LogLevel, BaseLogger, make_unique_name, save_fn, load_fn

numpy_compatible = np.ndarray
try:
    import torch
    numpy_compatible = torch.Tensor
except ImportError:
    pass

class TensorboardLogger(BaseLogger):
    """Create a Tensorboard logger.
    
    :param log_path: the base path where the log lies.
    :param name: the name of the log / experiment, will be used to construct the event file name. 
    :param bool terminal: whether messages will be printed to terminal.
    :param bool txt: whether messages will be printed to a text file.
    :param int warning_level: the level of warning messages, not used for now.
    """
    cmap = {
        "error": "\033[1;31m", 
        "debug": "\033[0m", 
        "warning": "\033[1;33m",
        "info": "\033[1;34m", 
        "reset": "\033[0m", 
    }
    
    def __init__(self, 
                 log_path: str, 
                 name: str, 
                 unique_name: Optional[str]=None, 
                 activate: bool=True, 
                 level=LogLevel.WARNING, 
                 *args, **kwargs):
        super().__init__(activate, level)
        
        from torch.utils.tensorboard.writer import SummaryWriter
        if unique_name:
            self.unique_name = unique_name
        else:
            self.unique_name = make_unique_name(name)
        self.log_path = os.path.join(log_path, self.unique_name, "tb")
        self.output_path = self.log_path
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)
        
        self.tb_writer = SummaryWriter(self.log_path)
        
    def log_str(
        self, 
        s: str, 
        type: Optional[str] = "LOG",
        *args, **kwargs):
        """Print the ``msg`` to stdout or txt file. This is for compatability use. 
        
        :param s: the message to print. 
        :param type: the type of the message, which also controls the color:
        
            * `None` or `"debug"`: no color
            * `"error"`: red
            * `"warning"`: yellow
            * `"log"`: blue
        :param terminal: whether s should be printed to terminal, masks `self.terminal`.
        :param txt: whether s should be printed to txt file, masks `self.txt`.
        """
        if type: type = type.lower()
        level = {
            None: LogLevel.DEBUG, 
            "error": LogLevel.ERROR, 
            "log": LogLevel.INFO, 
            "warning": LogLevel.WARNING, 
            "debug": LogLevel.DEBUG
        }.get(type)
        if not self.activate or level < self.level:
            return
        time_fmt = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print("{}[{}]{}\t{}".format(self.cmap[type], time_fmt, self.cmap["RESET"], s))

    def log_scalar(
        self, 
        tag: str, 
        value: Union[float, numpy_compatible], 
        step: Optional[int] = None):
        """Add scalar to tensorboard summary.
        
        :param tag: the identifier of the scalar.
        :param value: value to record.
        :param step: global timestep of the scalar. 
        """
        if not self.activate:
            return
        self.tb_writer.add_scalar(tag, value, step)
        
    def log_scalars(
        self,
        main_tag: str, 
        tag_scalar_dict: DictLike[str, Union[float, numpy_compatible]], 
        step: Optional[int] = None):
        """Add scalars which share the main tag to tensorboard summary.
        
        :param main_tag: the shared main tag of the scalars, can be a null string.
        :param tag_scalar_dict: a dictionary of tag and value.
        :param step: global timestep of the scalars.
        """
        if not self.activate:
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
        img_tensor: numpy_compatible, 
        step: Optional[int] = None, 
        dataformat: str = "CHW"):
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
        self.tb_writer.add_image(tag, img_tensor, step, dataformats=dataformat)
        
    def log_video(
        self, 
        tag: str, 
        vid_tensor: numpy_compatible, 
        step: Optional[int] = None, 
        fps: Optional[Union[int, float]] = 4, 
        dataformat: Optional[str] = "NTCHW"):
        """Add a piece of video to tensorboard summary. Note that this requires ``moviepy`` package.

        :param tag: the identifier of the video.
        :param vid_tensor: video data. 
        :param global_step: global step.
        :param fps: frames per second.
        :param dataformat: specify different permutation of the video tensor.
        """
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
        self.tb_writer.add_histogram(tag, np.asarray(values), step)

    def log_object(
        self,
        name: str, 
        object: Any, 
        type: Optional[str]="model", 
        path: Optional[str] = None, 
        protocol: str="torch"):
        """Save a Python object to the given path.
        
        :param name: the identifier of the object.
        :param object: the object to save.
        :param path: the path to save the object, will be created if not exist; will be set to `self.log_path` if None.
        """
        name = name.replace("/", "_")
        if path is None:
            path = self.output_path
        else:
            if not os.path.exists(path):
                os.makedirs(path)
        save_path = os.path.join(path, name)
        save_fn(protocol)(object, save_path)
        return save_path
    
    def load_object(self, 
                    name: str, 
                    path: Optional[str]=None, 
                    protocol="torch"):
        name = name.replace("/", "_")
        if path is None:
            path = self.output_path
        return load_fn(protocol)(os.path.join(path, name))
    
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.tb_writer.close()