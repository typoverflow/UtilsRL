from UtilsRL.exp.argparse import parse_args, argparse_callbacks, register_argparse_callback
from UtilsRL.exp.seed import *
from UtilsRL.exp.device import *
from UtilsRL.exp.snapshot import make_snapshot

from UtilsRL.logger import BaseLogger, DummyLogger, logger

from typing import Any, Optional

def setup(args, 
          _logger: Optional[BaseLogger] = None, 
          device: Optional[torch.device] = None, 
          seed: Optional[int] = None):
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
        
    if not device is None:
        device = select_device(device)
    elif "device" in args:
        device = select_device(args["device"])
    else:
        device = select_device(None)
    args["device"] = device
    
    if not seed is None:
        seed = set_seed(seed)
    elif "seed" in args:
        seed = set_seed(args["seed"])
    else:
        seed = set_seed(None)
    args["seed"] = seed
    
    global logger
    logger = _logger
        
    return args

    
register_argparse_callback("UtilsRL.snapshot", make_snapshot)