import os
import json
import pickle
import numpy as np

from datetime import datetime as datetime
from torch.utils.tensorboard.writer import SummaryWriter

from abc import ABC, abstractmethod
from typing import Optional, Sequence, Union, Dict, Any

numpy_compatible = np.ndarray
try:
    import torch
    numpy_compatible = torch.Tensor
except ImportError:
    pass

class BaseLogger(ABC):
    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def log_str(self, msg, *args, **kwargs):
        raise NotImplementedError

class DummyLogger(BaseLogger):
    def __init__(self, *args, **kwargs):
        pass

    def log_str(self, msg, *args, **kwargs):
        print(msg)

class ColoredLogger(BaseLogger):
    def __init__(self, *args, **kwargs):
        pass

    def log_str(
        self, 
        s: str, 
        type: Optional[str] = None,
        *args, **kwargs):
        """Log string to terminal (if self.terminal is True) and txt file (if self.txt is True).
        
        Args: 
            s: String to log. The  string can be formatted with `{}`, 
                and use the *args to fill in the blanks.
            type: Type of the log. When logging to terminal, this will determine the ansi color;
        """
        cmap = {
            None: "\033[0m", 
            "ERROR": "\033[1;31m", 
            "LOG": "\033[1;34m", 
            "SUCCESS": "\033[1;32m",
            "WARNING": "\033[1;33m",
            "RESET": "\033[0m"
        }
        time_fmt = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print("{}[{}]{}\t{}".format(cmap[type], time_fmt, cmap["RESET"], s))

class TensorboardLogger(BaseLogger):
    """TensorBoard Logger with full data-type support.
    """

    def __init__(self, log_path, name, terminal=True, txt=False, warning_level=3, *args, **kwargs):
        super(TensorboardLogger, self).__init__()

        if not (terminal or txt):
            raise ValueError("At least one of the terminal and log file should be enabled.")
        self.unique_name = self.make_unique_name(name)
        
        self.log_path = os.path.join(log_path, self.unique_name)
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        
        self.tb_writer = SummaryWriter(self.log_path)
        self.txt = txt
        if self.txt:
            self.txt_path = os.path.join(self.log_path, "logs.txt")
        self.terminal = terminal
        self.warning_level = warning_level

        self.log_str("logging to {}".format(self.log_path))
        
    def make_unique_name(self, name):
        now = datetime.now()
        suffix = now.strftime("%m-%d-%H-%M")
        pid_str = os.getpid()
        if name == "":
            return f"{suffix}-{pid_str}"
        else:
            return f"{name}-{suffix}-{pid_str}"

    @property
    def log_dir(self):
        return self.log_path

    def log_str(
        self, 
        s: str, 
        type: Optional[str] = None,
        terminal: bool = True, 
        txt: bool = True, 
        level: int = 4, 
        *args, **kwargs):
        """Log string to terminal (if self.terminal is True) and txt file (if self.txt is True).
        
        Args: 
            s: String to log. The  string can be formatted with `{}`, 
                and use the *args to fill in the blanks.
            type: Type of the log. When logging to terminal, this will determine the ansi color;
        """
        if level < self.warning_level:
            return
        cmap = {
            None: "\033[0m", 
            "ERROR": "\033[1;31m", 
            "LOG": "\033[1;34m", 
            "SUCCESS": "\033[1;32m",
            "WARNING": "\033[1;33m",
            "RESET": "\033[0m"
        }
        time_fmt = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if self.terminal and terminal:
            print("{}[{}]{}\t{}".format(cmap[type], time_fmt, cmap["RESET"], s))
        if self.txt and txt:
            with open(self.txt_path, "a+") as f:
                f.write("[{}]\t{}\n".format(time_fmt, s))

    def log_scalar(
        self, 
        tag: str, 
        value: Union[float, numpy_compatible], 
        step: Optional[int] = None):
        """Add scalar data to summary. 
        
        Args: 
            tag: Identifier of the data.
            value: Valur to save. If string passed, it will be treated as a caffe blob. 
            step: Global step value to record.
        """
        self.tb_writer.add_scalar(tag, value, step)

    def log_scalars(
        self,
        main_tag: str, 
        tag_scalar_dict: Dict[str, Union[float, numpy_compatible]], 
        step: Optional[int] = None):
        """Log many scalar data. 
        Note that this function also keeps logged scalars in memory. In extreme case it explodes your RAM.

        Args:
            main_tag: The parent name for the tags
            tag_scalar_dict: Key-value pair storing the tag and corresponding values
            step: Global step value to record
        """
        if main_tag is None:
            main_tag = ""
        for tag, value in tag_scalar_dict.items():
            self.tb_writer.add_scalar("/".join([main_tag, tag]), value, step)

    def log_dict(
        self, 
        tag: str, 
        data: dict):
        """Log dict data. """

        def pretty(d, indent=0):
            ret = ""
            for key, value in d.items():
                ret = ret + "\t"*indent + str(key) + ": "
                if isinstance(value, dict):
                    ret += "\n"+pretty(value, indent+2)
                else:
                    ret += str(value) + "\n"
            return ret
        formatted_str = pretty(data)
        self.log_str(tag+"\n"+formatted_str, type="LOG", level=99999)

    def log_image(
        self, 
        tag: str, 
        img_tensor: numpy_compatible, 
        step: Optional[int] = None, 
        dataformat: Optional[str] = "CHW"):
        """Log image.
        Note that this requires the ``pillow`` package.

        Args:
            tag: Data identifier
            img_tensor: An `uint8` or `float` Tensor of shape `
                [channel, height, width]` where `channel` is 1, 3, or 4.
                The elements in img_tensor can either have values
                in [0, 1] (float32) or [0, 255] (uint8).
                Users are responsible to scale the data in the correct range/type.
            global_step: Global step value to record
            walltime: Optional override default walltime (time.time()) of event.
            dataformats: This parameter specifies the meaning of each dimension of the input tensor.
        Shape:
            img_tensor: Default is :math:`(3, H, W)`. You can use ``torchvision.utils.make_grid()`` to
            convert a batch of tensor into 3xHxW format or use ``add_images()`` and let us do the job.
            Tensor with :math:`(1, H, W)`, :math:`(H, W)`, :math:`(H, W, 3)` is also suitible as long as
            corresponding ``dataformats`` argument is passed. e.g. CHW, HWC, HW.
        """
        self.tb_writer.add_image(tag, img_tensor, step, dataformats=dataformat)

    def log_video(
        self, 
        tag: str, 
        vid_tensor: numpy_compatible, 
        step: Optional[int] = None, 
        fps: Optional[Union[int, float]] = 4, 
        dataformat: Optional[str] = "NTCHW"):
        """Log video data.

        Note that this requires the ``moviepy`` package.

        Args:
            tag: Data identifier
            vid_tensor: Video data
            global_step: Global step value to record
            fps: Frames per second
            walltime: Optional override default walltime (time.time()) of event
            dataformats: Specify different permutation of the video tensor
        Shape:
            vid_tensor: :math:`(N, T, C, H, W)`. The values should lie in [0, 255]
            for type `uint8` or [0, 1] for type `float`.
        """
        self.tb_writer.add_video(tag, vid_tensor, step, fps)

    def log_object(
        self,
        name: str, 
        object: Any, 
        path: Optional[str] = None):
        """Log python object to a given path.
        
        Args: 
            name: name of the object.
            object: The python object to save.
            path: The directory to save the object. If the value None, then 
                the logger will use self.log_path.
        """

        if path is None:
            path = self.log_path
        if not os.path.exists(path):
            os.makedirs(path)
        with open(os.path.join(path, name), "wb") as fp:
            pickle.dump(object, fp)
        self.log_str(f"Saved object to {os.path.join(path, name)}", type="LOG", level=2)
    
    def load_object(
        self,
        name: str, 
        path: Optional[str] = None):
        """Restore saved object.

        Args: 
            name: name of the object, used to index the binary file in path. 
            path: The directory to search for the file. If None, then logger 
                will look up in self.log_path.
        """

        if path is None:
            path = self.log_path
        with open(os.path.join(path, name), "rb") as fp:
            obj = pickle.load(fp)
        self.log_str(f"Load object from {os.path.join(path, name)}", type="LOG", level=2)
        return obj
          

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.tb_writer.close()

    
    

        

