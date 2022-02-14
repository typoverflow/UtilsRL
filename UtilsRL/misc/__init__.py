from UtilsRL.misc.seed import *
from UtilsRL.misc.device import *
from UtilsRL.misc.namespace import *
from UtilsRL.logger import BaseLogger, DummyLogger

from typing import Any, Optional

def setup(args, 
          logger: Optional[BaseLogger] = None, 
          device: torch.device = "cpu", 
          seed: int = None):
    if logger is None:
        logger = DummyLogger()
    device = select_device(device)
    seed = set_seed(seed)
    args["logger"] = logger
    args["seed"] = seed
    args["device"] = device
    return args