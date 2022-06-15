import os
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
    """Create a dummy logger which just calls `print`. This is helpful when dubugging output issues. 
    """
    
    def __init__(self, *args, **kwargs):
        pass

    def log_str(self, msg, *args, **kwargs):
        """Print the msg to stdout. 

        :param msg: the message to print. 
        """
        print(msg)

class ColoredLogger(BaseLogger):
    """Create a logger which prints msg to terminal with ansi colors. 
    """
    
    def __init__(self, *args, **kwargs):
        pass

    def log_str(
        self, 
        s: str, 
        type: Optional[str] = None,
        *args, **kwargs):
        """Print the `msg` to stdout with ansi colors. 
        
        :param s: the message to print. 
        :param type: the type of the message, which also controls the color:
        
            * `None`: no color
            * `"Error"`: red
            * `"LOG"`: blue
            * `"SUCCESS"`: green
            * `"WARNING"`: yellow
            * `"RESET"`: reset all the colors
        """
        if type:
            type = type.upper()
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
    """Create a Tensorboard logger.
    
    :param log_path: the base path where the log lies.
    :param name: the name of the log / experiment, will be used to construct the event file name. 
    :param bool terminal: whether messages will be printed to terminal.
    :param bool txt: whether messages will be printed to a text file.
    :param int warning_level: the level of warning messages, not used for now.
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
        type: Optional[str] = "LOG",
        terminal: bool = True, 
        txt: bool = True, 
        level: int = 4, 
        *args, **kwargs):
        """Print the ``msg`` to stdout or txt file. 
        
        :param s: the message to print. 
        :param type: the type of the message, which also controls the color:
        
            * `None`: no color
            * `"Error"`: red
            * `"LOG"`: blue
            * `"SUCCESS"`: green
            * `"WARNING"`: yellow
            * `"RESET"`: reset all the colors
        :param terminal: whether s should be printed to terminal, masks `self.terminal`.
        :param txt: whether s should be printed to txt file, masks `self.txt`.
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
        """Add scalar to tensorboard summary.
        
        :param tag: the identifier of the scalar.
        :param value: value to record.
        :param step: global timestep of the scalar. 
        """
        self.tb_writer.add_scalar(tag, value, step)

    def log_scalars(
        self,
        main_tag: str, 
        tag_scalar_dict: Dict[str, Union[float, numpy_compatible]], 
        step: Optional[int] = None):
        """Add scalars which share the main tag to tensorboard summary.
        
        :param main_tag: the shared main tag of the scalars, can be a null string.
        :param tag_scalar_dict: a dictionary of tag and value.
        :param step: global timestep of the scalars.
        """
        
        if main_tag is None or main_tag == "":
            main_tag = ""
        else:
            main_tag = main_tag+"/"
            
        for tag, value in tag_scalar_dict.items():
            self.tb_writer.add_scalar(main_tag+tag, value, step)

    def log_dict(
        self, 
        tag: str, 
        data: dict):
        """Print a Python dict object to terminal. The dict will be formatted. 
        
        :param tag: the identifier of the dict.
        :param data: a dict-like object.
        """
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

    def log_object(
        self,
        name: str, 
        object: Any, 
        path: Optional[str] = None):
        """Save a Python object to the given path.
        
        :param name: the identifier of the object.
        :param object: the object to save.
        :param path: the path to save the object, will be created if not exist; will be set to `self.log_path` if None.
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
        """Restore a previously saved Python object.
        
        :param name: the identifier of the object.
        :param path: the path to load the object, will be set to `self.log_path` if None.
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

    
logger = ColoredLogger() 