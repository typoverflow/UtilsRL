from typing import Optional, Union, Any, Sequence
from typing import Dict as DictLike

import numpy as np

from UtilsRL.logger import logger
from .base import Replay, SimpleReplay, FlexReplay

class TransitionSimpleReplay(SimpleReplay):
    def __init__(self, max_size: int, field_specs: Optional[DictLike]=None, *args, **kwargs):
        super().__init__(max_size, field_specs, *args, **kwargs)
        self._size = 0
        
    def reset(self):
        self._pointer = self._size = 0
        self.fields = self.fields or {}
        for _key, _specs in self.field_specs.items():
            initializer = _specs.get("initialzier", np.zeros)
            self.fields[_key] = initializer(shape=[self._max_size, ]+list(_specs["shape"]), dtype=_specs["dtype"])
            
    def add_fields(self, new_field_specs: Optional[DictLike]=None):
        new_field_specs = new_field_specs or {}
        self.fields = self.fields or {}
        for _key, _specs in new_field_specs.items():
            _old_specs = self.field_specs.get(_key, None)
            if _old_specs is None or _old_specs != _specs:
                self.field_specs[_key] = _specs
                initializer = _specs.get("initializer", np.zeros)
                self.fields[_key] = initializer(shape=[self._max_size, ]+list(_specs["shape"]), dtype=_specs["dtype"])
                
    def add_sample(self, key_or_dict: Union[str, DictLike], data: Optional[Any]=None):
        if isinstance(key_or_dict, str):
            key_or_dict = {key_or_dict: data}
        unsqueeze = None
        data_len = None
        for _key, _data in key_or_dict.items():
            if unsqueeze is None:
                unsqueeze = len(_data.shape) == len(self.field_specs[_key]["shape"])
                data_len = 1 if unsqueeze else _data.shape[0]
                index_to_go = np.arange(self._pointer, self._pointer + data_len) % self._max_size
            self.fields[_key][index_to_go] = _data
        self._pointer = (self._pointer + data_len) % self._max_size
        self._size = min(self._size + data_len, self._max_size)
        
    def random_batch(self, batch_size: Optional[int]=None, fields: Optional[Sequence[str]]=None, return_idx: bool=False):
        if len(self) == 0:
            batch_data, batch_idx = None, None
        else:
            if batch_size is None:
                batch_idx = np.arange(0, len(self))
                np.random.shuffle(batch_idx)
            else:
                batch_idx = np.random.randint(0, len(self), batch_size)
            if fields is None:
                fields = self.field_specs.keys()
            batch_data = {
                _key:self.fields[_key][batch_idx] for _key in fields
            }
        return (batch_data, batch_idx) if return_idx else batch_data
    
    
class TransitionFlexReplay(FlexReplay):
    def __init__(
        self, 
        max_size: int, 
        field_specs: Optional[DictLike]=None, 
        cache_max_size: Optional[int]=None, 
        *args, **kwargs
    ):
        super().__init__(max_size, field_specs, cache_max_size, *args, **kwargs)
        self._committed_size = 0

    def __len__(self):
        return self._committed_size
    
    def reset_committed(self):
        self._committed_size = 0
        self._committed_pointer = 0
        for _key, _spec in self.field_specs.items():
            initializer = _spec.get("initializer", np.zeros)
            self.committed_fields[_key] = initializer(shape=[self._committed_max_size, ]+list(_spec["shape"]), dtype=_spec["dtype"])
    
    def reset_cache(self):
        self._cache_start = 0
        self._cache_pointer = {_key: 0 for _key in self.field_specs}
        for _key, _spec in self.field_specs.items():
            initializer = _spec.get("initializer", np.zeros)
            self.cache_fields[_key] = initializer(shape=[self._cache_max_size, ]+list(_spec["shape"]), dtype=_spec["dtype"])  
    
    def add_fields(self, new_field_specs):
        new_field_specs = new_field_specs or {}
        self.cache_fields = self.cache_fields or {}
        self.committed_fields = self.committed_fields or {}
        for _key, _specs in new_field_specs.items():
            _old_spec = self.field_specs.get(_key, None)
            if _old_spec is None or _old_spec != _specs:
                self.field_specs[_key] = _specs
                initializer = _specs.get("initializer", np.zeros)
                self.cache_fields[_key] = initializer(shape=[self._committed_max_size, ]+list(_specs["shape"]), dtype=_specs["dtype"])
                self.committed_fields[_key] = initializer(shape=[self._cache_max_size, ]+list(_specs["shape"]), dtype=_specs["dtype"])

    def add_sample(self, key_or_dict: Union[str, DictLike], data: Optional[Any]=None):
        if isinstance(key_or_dict, str):
            key_or_dict = {key_or_dict: data}
        _data_len = None
        for _key, _data in key_or_dict.items():
            if _data_len is None:
                _data_arr = np.asarray(_data)
                if _data_arr.shape == ():
                    _data_arr = np.asarray([_data, ])
                _data_len = _data_arr.reshape([-1, ]+self.field_specs[_key]["shape"]).shape[0]
            _key_pointer = self._cache_pointer[_key]
            index = np.arange(_key_pointer, _key_pointer+_data_len) % self._cache_max_size
            self.cache_fields[_key][index] = _data
            self._cache_pointer[_key] = (_key_pointer + _data_len) % self._cache_max_size


    def commit(self, commit_num: Optional[int]=None):
        can_commit_num = np.unique(np.asarray([(p-self._cache_start)%self._cache_max_size for _, p in self._cache_pointer.items()]))
        # if len(can_commit_num) != 1:
            # logger.log_str(f"buffer: commit different size of samples", type="warning")
        commit_num = commit_num or can_commit_num.min()
        if commit_num > can_commit_num.min():
            raise ValueError(f"cannot commit {commit_num} samples, cache size is only {can_commit_num.min()}.")
        index1 = np.arange(self._committed_pointer, self._committed_pointer + commit_num) % self._committed_max_size
        index2 = np.arange(self._cache_start, self._cache_start + commit_num) % self._cache_max_size

        # update pointers except for _cache_pointer
        self._committed_size = min(self._committed_size+commit_num, self._committed_max_size)
        self._committed_pointer = (self._committed_pointer + commit_num) % self._committed_max_size
        self._cache_start = (self._cache_start + commit_num) % self._cache_max_size 

        # make the commit really happen
        for _key in self.field_specs:
            self.committed_fields[_key][index1] = self.cache_fields[_key][index2]
            # self._cache_pointer[_key] = self._cache_start
        return commit_num
                
    def random_batch(self, batch_size: Optional[int]=None, fields: Optional[Sequence[str]]=None, return_idx: bool=False):
        if len(self) == 0:
            batch_data, batch_idx = None, None
        else:
            if batch_size is None:
                batch_idx = np.arange(0, len(self))
                np.random.shuffle(batch_idx)
            else:
                batch_idx = np.random.randint(0, len(self), batch_size)
            if fields is None:
                fields = self.field_specs.keys()
            batch_data = {
                _key:self.committed_fields[_key][batch_idx] for _key in fields
            }
        return (batch_data, batch_idx) if return_idx else batch_data
    
    def get_cache_data(self, num: Optional[int]=None):
        ret = {}
        num = num or np.inf
        for _key, _pointer in self._cache_pointer.items():
            can_get_num = (_pointer - self._cache_start) % self._cache_max_size
            get_num = min(num, can_get_num)
            ret[_key] = self.cache_fields[_key][
                np.arange(self._cache_start, self._cache_start + get_num) % self._cache_max_size
            ]
        return ret
    
