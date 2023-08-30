from UtilsRL.exp.argparse import parse_args, argparse_callbacks, register_argparse_callback
from UtilsRL.exp._seed import *
from UtilsRL.exp._device import *

from UtilsRL.logger import BaseLogger
from UtilsRL.logger import logger as url_internal_logger

from typing import Any, Optional

device = None
seed = None
logger = None

def setup(args, 
          _logger: Optional[BaseLogger] = None, 
          _device: Optional[torch.device] = None, 
          _seed: Optional[int] = None):
    """Setup the args for logger, device and seed. The choice sequence is \
        ``given parameter -> args -> None (default)``, for example, if `_device` is not `None`, \
        then `_device` will be selected as the device-to-use during this experiment; otherwise, `args.device` will be used, \
        if `args.device` is also `None`, then the return object of :func:`~select_device` will be used. 
        
    :param args: arguments of this experiment, should be a dict-like or `NameSpace` class.
    :param _logger: specify the logger to use. 
    :param _device: specify the device to use.
    :param _seed: specify the seed to use. 
    """
    
    if not _logger is None:
        _logger = _logger
    elif "logger" in args:
        _logger = args["logger"]
    else:
        _logger = url_internal_logger
    args["logger"] = _logger
        
    if not _device is None:
        _device = select_device(_device)
    elif "device" in args:
        _device = select_device(args["device"])
    else:
        _device = select_device(None)
    args["device"] = _device
    
    if not _seed is None:
        _seed = set_seed(_seed)
    elif "seed" in args:
        _seed = set_seed(args["seed"])
    else:
        _seed = set_seed(None)
    args["seed"] = _seed
    
    global logger
    logger = _logger
    
    global seed
    seed = _seed
    
    global device
    device = _device
    
    return args

    