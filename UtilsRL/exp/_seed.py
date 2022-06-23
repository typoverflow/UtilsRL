import torch
import random
import numpy as np
from typing import Optional, Union

def set_seed(seed: Optional[Union[str, int]] = None):
    if seed is None:
        seed = np.random.randint(0, 2**32)
    seed = int(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    return seed

        