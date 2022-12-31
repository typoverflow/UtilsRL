# from typing import Optional, Union, Any, Sequence
# from typing import Dict as DictLike

# import numpy as np

# from UtilsRL.logger import logger
# from .base import Replay, SimpleReplay, FlexReplay

# class TrajectorySimpleReplay(SimpleReplay):
#     def __init__(self, max_size: int, max_traj_len: int, field_specs: Optional[DictLike]=None, *args, **kwargs):
#         field_specs["valid"] = {
#             "shape": [1, ], 
#             "dtype": np.float32, 
#         }
#         super().__init__(max_size, field_specs, *args, **kwargs)
#         self._max_traj_len = max_traj_len
#         self._size = 0
        
#     def reset(self):
#         self._pointer = self._size = 0
#         self.fields = self.fields or {}
#         for _key, _specs in self.field_specs.items():
#             initializer = _specs.get("initializer", np.zeros)
#             self.fields[_key] = initializer(shape=[self._max_size, self._max_traj_len, ]+list(_specs["shape"]), dtype=_specs["stype"])
            
#     def add_fields(self, new_field_specs: Optional[DictLike]=None):
#         new_field_specs = new_field_specs or {}
#         self.fields = self.fields or {}
#         for _key, _specs in new_field_specs.items():
#             _old_specs = self.field_specs.get(_key, None)
#             if _old_specs is None or _old_specs != _specs:
#                 self.field_specs[_key] = _specs
#                 initializer = _specs.get("initializer", np.zeros)
#                 self.fields[_key] = initializer(shape=[self._max_size, self._max_traj_len, ]+list(_specs["shape"]), dtype=_specs["stype"])
    
#     def add_sample(self, key_or_dict: Union[str, DictLike], data: Optional[Any]=None):
#         # we force data to be [Batch, Length, ...]
#         def pad_or_trunc(name, values, max_len):
#             if values.shape
        
#         if isinstance(key_or_dict, str):
#             key_or_dict = {key_or_dict: data}
        