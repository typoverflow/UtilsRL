from typing import Optional, Union, Any, Sequence
from typing import Dict as DictLike

import random
import numpy as np

from UtilsRL.logger import logger
from .base import Replay, SimpleReplay, FlexReplay
from .transition_replay import TransitionSimpleReplay, TransitionFlexReplay


class PrioritizedSimpleReplay(TransitionSimpleReplay):
    def __init__(self, 
                 max_size:int, 
                 field_specs: Optional[DictLike]=None, 
                 metric="propotional", 
                 alpha=None, 
                 *args, **kwargs):
        super().__init__(max_size, field_specs, *args, **kwargs)
        self.metric = metric
        if metric == "propotional":
            from UtilsRL.data_structure.data_structure import SumTree as CSumTree
            self.credential_cls = CSumTree
            self.credential = self.credential_cls(self._max_size)
            self.metric_fn = self._propotional
            self.alpha = alpha
        elif metric == "rank":
            raise NotImplementedError
        else:
            raise ValueError(f"PER only supports propotional or rank-based")
        
    def reset(self):
        super().reset()
        self.credential = self.credential_cls(self._max_size)
        
    def add_sample(self, data_dict: DictLike, metric_value: Union[Sequence, float]):
        metric_value = np.asarray(metric_value)
        if metric_value.shape == ():
            metric_value = np.asarray([metric_value, ])
        data_len = len(metric_value)
        index_to_go = np.arange(self._pointer, self._pointer+data_len) % self._max_size
        for (_key, _data)in data_dict.items():
            self.fields[_key][index_to_go] = _data
        self._pointer = (self._pointer + data_len) % self._max_size
        self._size = min(self._size + data_len, self._max_size)

        # update credential
        if self.metric == "propotional":
            for m in metric_value:
                self.credential.add(m)
        elif self.metric == "rank":
            raise NotImplementedError
    
    def random_batch(self, batch_size: Optional[int] = None, beta: float=0, fields: Optional[Sequence[str]] = None, return_idx: bool = True):
        if len(self) == 0:
            batch_data, batch_is, batch_idx = None, None, None
        else:
            if batch_size is None:
                raise NotImplementedError(f"you must specify a batch size for PER for now.")
            else:
                batch_is = []
                batch_idx = []
                if self.metric == "propotional":
                    total_p = self.credential.total()
                    segment = 1 / batch_size
                    for i_seg in range(batch_size):
                        target = (random.random() + i_seg) * segment
                        idx, p = self.credential.find(target)
                        batch_is.append(np.power((len(self)*p/total_p), beta))
                        batch_idx.append(idx)
            if fields is None:
                fields = self.field_specs.keys()
            batch_idx = np.asarray(batch_idx)
            batch_is = np.asarray(batch_is)
            batch_data = {
                _key: self.fields[_key][batch_idx] for _key in fields
            }
        return (batch_data, batch_is, batch_idx) if return_idx else (batch_data, batch_is)
    
    def batch_update(self, batch_idx, metric_value):
        batch_idx = np.asarray(batch_idx)
        if batch_idx.shape == ():
            batch_idx = np.asarray([batch_idx, ])
        metric_value = np.asarray(metric_value)
        if metric_value.shape == ():
            metric_value = np.asarray([metric_value, ])
        # update crendential
        for idx, metric_v in zip(batch_idx, metric_value):
            self.credential.update(idx, self.metric_fn(metric_v))
                        