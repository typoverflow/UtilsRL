from typing import Optional, Union, Any, Sequence
from typing import Dict as DictLike

import random
import numpy as np

from UtilsRL.logger import logger
from .base import Replay, SimpleReplay, FlexReplay
from .transition_replay import TransitionSimpleReplay, TransitionFlexReplay

def proportional(metric_value, alpha):
    return np.abs(metric_value) ** alpha

class PrioritizedSimpleReplay(TransitionSimpleReplay):
    def __init__(self, 
                 max_size: int, 
                 field_specs: Optional[DictLike]=None, 
                 metric: str="proportional", 
                 alpha: float=0.2, 
                 *args, **kwargs):
        super().__init__(max_size, field_specs, *args, **kwargs)
        self.metric = metric
        if metric == "proportional":
            from UtilsRL.data_structure.data_structure import SumTree as CSumTree
            self.credential_cls = CSumTree
            self.credential = self.credential_cls(self._max_size)
            self.metric_fn = lambda x: proportional(x, alpha)
            self.alpha = alpha
        elif metric == "rank":
            raise NotImplementedError
        else:
            raise ValueError(f"PER only supports proportional or rank-based.")
        
    def reset(self):
        self.credential.reset()
        super().reset()
        
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
        if self.metric == "proportional":
            self.credential.add(metric_value)
        elif self.metric == "rank":
            raise NotImplementedError
    
    def random_batch(self, batch_size: Optional[int]=None, beta: float=0, fields: Optional[Sequence[str]]=None, return_idx: bool=True):
        if len(self) == 0:
            batch_data, batch_is, batch_idx = None, None, None
        else:
            if batch_size is None:
                raise NotImplementedError(f"you must specify a batch size for PER for now.")
            else:
                batch_is = []
                batch_idx = []
                if self.metric == "proportional":
                    total_p = self.credential.total()
                    segment = 1 / batch_size
                    batch_target = (np.random.random(size=[batch_size, ]) + np.arange(0, batch_size))*segment
                    batch_idx, batch_p = self.credential.find(batch_target)
                    batch_idx = np.asarray(batch_idx)
                    batch_p = np.asarray(batch_p)
                    batch_is = np.power((len(self)*batch_p/total_p))
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
        self.credential.update(batch_idx, metric_value)

        
class PrioritizedFlexReplay(TransitionFlexReplay):
    def __init__(self, 
                 max_size: int, 
                 field_specs: Optional[DictLike]=None, 
                 metric: str="proportional", 
                 alpha: float=0.2, 
                 cache_max_size: Optional[int]=None, 
                 metric_key: str="metric_value", 
                 *args, **kwargs):
        field_specs = field_specs or {}
        if not metric_key in field_specs:
            field_specs[metric_key] = {
                "shape": [1, ], 
                "dtype": np.float32
            }
        super().__init__(self, max_size, field_specs, cache_max_size, *args, **kwargs)
        self.metric = metric
        self.metric_key = metric
        if metric == "proportional":
            from UtilsRL.data_structure.data_structure import SumTree as CSumTree
            self.credential_cls = CSumTree
            self.credential = self.credential_cls(self._committed_max_size)
            self.metric_fn = lambda x: proportional(x, alpha)
            self.alpha = alpha
        elif metric == "rank":
            raise NotImplementedError
        else:
            raise ValueError(f"PER only supports proportinal or rank-based.")
        
    def __len__(self):
        return self._committed_size
    
    def reset_committed(self):
        self.credential.reset()
        super().reset_committed()
        
    def commit(self, commit_num: Optional[int]=None):
        can_commit_num = (self._cache_start[self.metric_key] - self._cache_start) % self._cache_max_size
        commit_num = commit_num or can_commit_num
        if commit_num > can_commit_num:
            raise ValueError(f"cannot commit {commit_num} samples, cache size is only {can_commit_num}.")
        index1 = np.arange(self._committed_pointer, self._committed_pointer + commit_num) % self._committed_max_size
        index2 = np.arange(self._cache_start, self._cache_start + commit_num) % self._cache_max_size

        # update pointers except for _cache_pointer
        self._committed_size = min(self._committed_size+commit_num, self._committed_max_size)
        self._committed_pointer = (self._committed_pointer + commit_num) % self._committed_max_size
        self._cache_start = (self._cache_start + commit_num) % self._cache_max_size 

        # make the commit really happen
        for _key in self.field_specs:
            self.committed_fields[_key][index1] = self.cache_fields[_key][index2]
            self._cache_pointer[_key] = self._cache_start
        # update credential
        if self.metric == "proportional":
            self.credential.add(self.cache_fields[index2])
        elif self.metric == "rank":
            raise NotImplementedError
        
    def random_batch(self, batch_size: Optional[int]=None, beta: float=0, fields: Optional[Sequence[str]]=None, return_idx: bool=True):
        if len(self) == 0:
            batch_data = batch_is = batch_idx = None
        else:
            if batch_size is None:
                raise NotImplementedError(f"you must specify a batch size for PER for now.")
            else:
                batch_is = []
                batch_idx = []
                if self.metric == "proportional":
                    total_p = self.credential.total()
                    segment = 1 / batch_size
                    batch_target = (np.random.random(size=[batch_size, ]) + np.arange(0, batch_size))*segment
                    batch_idx, batch_p = self.credential.find(batch_target)
                    batch_idx = np.asarray(batch_idx)
                    batch_p = np.asarray(batch_p)
                    batch_is = np.power((len(self)*batch_p/total_p))
            if fields is None:
                fields = self.field_specs.keys()
            batch_idx = np.asarray(batch_idx)
            batch_is = np.asarray(batch_is)
            batch_data = {
                _key: self.committed_fields[_key][batch_idx] for _key in fields
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
        self.credential.update(batch_idx, metric_value)