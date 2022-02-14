from UtilsRL.misc.seed import *
from UtilsRL.misc.device import *
from UtilsRL.misc.namespace import *
from UtilsRL.misc.chore import *
from UtilsRL.logger import BaseLogger, DummyLogger

from typing import Any, Optional

def setup(args, 
          logger: Optional[BaseLogger] = None, 
          device: Optional[torch.device] = None, 
          seed: Optional[int] = None):
    """Setup the args for logger, device and seed. The sequence is:
        given parameter -> args -> None
    """
    if not logger is None:
        logger = logger
    elif "logger" in args:
        logger = args["logger"]
    else:
        logger = DummyLogger()
    args["logger"] = logger
        
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
        
    return args