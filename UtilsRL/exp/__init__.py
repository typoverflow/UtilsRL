from UtilsRL.exp.argparse import parse_args, argparse_callbacks, register_argparse_callback
from UtilsRL.exp._seed import *
from UtilsRL.exp._device import *
from UtilsRL.exp.snapshot import make_snapshot

from UtilsRL.logger import BaseLogger, DummyLogger, logger

from typing import Any, Optional

device = None
seed = None

def setup(args, 
          _logger: Optional[BaseLogger] = None, 
          _device: Optional[torch.device] = None, 
          _seed: Optional[int] = None):
    """Setup the args for logger, device and seed. The sequence is:
        given parameter -> args -> None
    """
    if not _logger is None:
        _logger = _logger
    elif "logger" in args:
        _logger = args["logger"]
    else:
        _logger = DummyLogger()
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

    
register_argparse_callback("UtilsRL.snapshot", make_snapshot)