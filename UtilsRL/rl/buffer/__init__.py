from collections import OrderedDict
from gym.spaces import (
    Space, 
    Box, 
    Discrete, 
    MultiDiscrete, 
    MultiBinary, 
    Tuple, 
    Dict, 
)

from .base import Replay, SimpleReplay, FlexReplay
from .transition_replay import TransitionSimpleReplay, TransitionFlexReplay
from .prioritized_replay import PrioritizedSimpleReplay, PrioritizedFlexReplay

__SUPPORT_CONVERTING_SPACES__ = {
    Box, Discrete, Dict
}

def convert_space_to_spec(space, onehot=False, dtype=None):
    if not type(space) in __SUPPORT_CONVERTING_SPACES__:
        raise TypeError(f"convert space to spec receives non-supported type of space: {type(space)}.")
    if isinstance(space, Box):
        return {
            "shape": list(space.shape), 
            "dtype": dtype or space.dtype, 
            "space": space, 
        }
    if isinstance(space, Discrete):
        if onehot: 
            # if use onehot to encode each discrete action (often used when storing log probs)
            return {
                "shape": [space.n, ], 
                "dtype": dtype or space.dtype, 
                "space": space, 
            }
        else:
            return {
                "shape": [1, ], 
                "dtype": dtype or space.dtype, 
                "space": space, 
            }
    if isinstance(space, Dict):
        return OrderedDict([(_key, convert_space_to_spec(_space, onehot, dtype)) for (_key, _space) in space.items()])
