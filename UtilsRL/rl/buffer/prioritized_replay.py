from typing import Optional, Union, Any, Sequence
from typing import Dict as DictLike

import random
import numpy as np

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
            from UtilsRL.data_structure import SumTree as CSumTree
            from UtilsRL.data_structure import MinTree as CMinTree
            self.sum_tree = CSumTree(self._max_size)
            self.min_tree = CMinTree(self._max_size)
            self.metric_fn = lambda x: proportional(x, alpha)
            self.alpha = alpha
            self.max_metric_value = 1
        elif metric == "rank":
            raise NotImplementedError
        else:
            raise ValueError(f"PER only supports proportional or rank-based.")

        self.reset()
        
    def reset(self):
        if self.metric == "proportional":
            self.sum_tree.reset()
            self.min_tree.reset()
        else:
            raise NotImplementedError
        super().reset()
        
    def add_sample(self, data_dict: DictLike):
        data_len = None
        index_to_go = None
        for (_key, _data)in data_dict.items():
            if data_len is None:
                data_len = _data.reshape([-1, ]+list(self.field_specs[_key]["shape"])).shape[0]
                index_to_go = np.arange(self._pointer, self._pointer+data_len) % self._max_size
            self.fields[_key][index_to_go] = _data
        self._pointer = (self._pointer + data_len) % self._max_size
        self._size = min(self._size + data_len, self._max_size)

        # update credential
        if self.metric == "proportional":
            self.sum_tree.add(np.full(shape=[data_len, ], fill_value=self.metric_fn(self.max_metric_value)))
            self.min_tree.add(np.full(shape=[data_len, ], fill_value=self.metric_fn(self.max_metric_value)))
        elif self.metric == "rank":
            raise NotImplementedError
    
    def random_batch(self, batch_size: Optional[int]=None, beta: float=0, fields: Optional[Sequence[str]]=None, return_idx: bool=True):
        if len(self) == 0:
            batch_data, batch_is, batch_idx = None, None, None
        else:
            if batch_size is None:
                raise NotImplementedError(f"you must specify a batch size for PER for now.")
            else:
                batch_idx = []
                if self.metric == "proportional":
                    min_p = self.min_tree.min() + 1e-6
                    segment = 1 / batch_size
                    batch_target = (np.random.random(size=[batch_size, ]) + np.arange(0, batch_size))*segment
                    batch_idx, batch_p = self.sum_tree.find(batch_target)
                    batch_idx = np.asarray(batch_idx)
                    batch_p = np.asarray(batch_p)
                    # Simplified form for IS weight, ref: https://github.com/nuance1979/tianshou/blob/104d47655299c2d386caf73ecb011a688b3384df/tianshou/data/buffer/prio.py#L73
                    batch_is = (batch_p / min_p) ** (-beta)
                    batch_is = batch_is / batch_is.max()
            if fields is None:
                fields = self.field_specs.keys()
            batch_data = {
                _key: self.fields[_key][batch_idx] for _key in fields
            }
        return (batch_data, batch_is, batch_idx) if return_idx else (batch_data, batch_is)
    
    def batch_update(self, batch_idx, metric_value):
        batch_idx = np.asarray(batch_idx)
        if len(batch_idx.shape) == 0:
            batch_idx = np.asarray([batch_idx, ])
        metric_value = np.asarray(metric_value)
        if len(metric_value.shape) == 0:
            metric_value = np.asarray([metric_value, ])
        # update crendential
        self.max_metric_value = max(np.max(metric_value), self.max_metric_value)
        self.sum_tree.update(batch_idx, self.metric_fn(metric_value))
        self.min_tree.update(batch_idx, self.metric_fn(metric_value))

        
class PrioritizedFlexReplay(TransitionFlexReplay):
    def __init__(self, 
                 max_size: int, 
                 field_specs: Optional[DictLike]=None, 
                 metric: str="proportional", 
                 alpha: float=0.2, 
                 cache_max_size: Optional[int]=None, 
                 *args, **kwargs):
        field_specs = field_specs or {}
        super().__init__(max_size, field_specs, cache_max_size, *args, **kwargs)
        self.metric = metric
        if metric == "proportional":
            from UtilsRL.data_structure import SumTree as CSumTree
            from UtilsRL.data_structure import MinTree as CMinTree
            self.sum_tree = CSumTree(self._committed_max_size)
            self.min_tree = CMinTree(self._committed_max_size)
            self.metric_fn = lambda x: proportional(x, alpha)
            self.alpha = alpha
            self.max_metric_value = 1
        elif metric == "rank":
            raise NotImplementedError
        else:
            raise ValueError(f"PER only supports proportinal or rank-based.")
        
        self.reset()
        
    def __len__(self):
        return self._committed_size
    
    def reset_committed(self):
        if self.metric == "proportional":
            self.sum_tree.reset()
            self.min_tree.reset()
        else:
            raise NotImplementedError
        super().reset_committed()
        
    def commit(self, commit_num: Optional[int]=None):
        commit_num = super().commit(commit_num)
        # update credential
        if self.metric == "proportional":
            self.sum_tree.add(np.full(shape=[commit_num, ], fill_value=self.metric_fn(self.max_metric_value)))
            self.min_tree.add(np.full(shape=[commit_num, ], fill_value=self.metric_fn(self.max_metric_value)))
        elif self.metric == "rank":
            raise NotImplementedError
        
    def random_batch(self, batch_size: Optional[int]=None, beta: float=0, fields: Optional[Sequence[str]]=None, return_idx: bool=True):
        if len(self) == 0:
            batch_data = batch_is = batch_idx = None
        else:
            if batch_size is None:
                raise NotImplementedError(f"you must specify a batch size for PER for now.")
            else:
                batch_idx = []
                if self.metric == "proportional":
                    min_p = self.min_tree.min() + 1e-6
                    segment = 1 / batch_size
                    batch_target = (np.random.random(size=[batch_size, ]) + np.arange(0, batch_size))*segment
                    batch_idx, batch_p = self.sum_tree.find(batch_target)
                    batch_idx = np.asarray(batch_idx)
                    batch_p = np.asarray(batch_p)
                    # Simplified form for IS weight, ref: https://github.com/nuance1979/tianshou/blob/104d47655299c2d386caf73ecb011a688b3384df/tianshou/data/buffer/prio.py#L73
                    batch_is = (batch_p / min_p) ** (-beta)
                    batch_is = batch_is / batch_is.max()
            if fields is None:
                fields = self.field_specs.keys()
            batch_data = {
                _key: self.committed_fields[_key][batch_idx] for _key in fields
            }
        return (batch_data, batch_is, batch_idx) if return_idx else (batch_data, batch_is)
    
    def batch_update(self, batch_idx, metric_value):
        batch_idx = np.asarray(batch_idx)
        if len(batch_idx.shape) == 0:
            batch_idx = np.asarray([batch_idx, ])
        metric_value = np.asarray(metric_value)
        if len(metric_value.shape) == 0:
            metric_value = np.asarray([metric_value, ])
        # update crendential
        self.max_metric_value = max(np.max(metric_value), self.max_metric_value)
        self.sum_tree.update(batch_idx, self.metric_fn(metric_value))
        self.min_tree.update(batch_idx, self.metric_fn(metric_value))
