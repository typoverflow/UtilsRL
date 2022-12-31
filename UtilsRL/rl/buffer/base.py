from typing import Optional, Sequence, Any, Union
from typing import Dict as DictLike
from abc import ABC, abstractmethod


class Replay(ABC):
    def __init__(self, max_size: int, field_specs: Optional[DictLike]=None, *args, **kwargs):
        super().__init__()
        self._max_size = int(max_size)
        self.field_specs = field_specs
        
    @property
    def field_names(self):
        return list(self.field_specs.keys())
    
    @abstractmethod
    def reset(self):
        raise NotImplementedError

    @abstractmethod
    def __len__(self):
        raise NotImplementedError
    
    @abstractmethod
    def add_fields(self):
        raise NotImplementedError
    
    @abstractmethod
    def add_sample(self):
        raise NotImplementedError
    
    @abstractmethod
    def random_batch(self):
        raise NotImplementedError
    
    
class SimpleReplay(ABC):
    def __init__(self, max_size: int, field_specs: Optional[DictLike]=None, *args, **kwargs):
        super().__init__()
        self.field_specs = {}
        self.fields = {}
        
        self._max_size = int(max_size)
        self._size = 0
        self.add_fields(field_specs)
                
    def __len__(self):
        return self._size
        
    @abstractmethod
    def reset(self):
        raise NotImplementedError
    
    @abstractmethod
    def add_fields(self):
        raise NotImplementedError
    
    @abstractmethod
    def add_sample(self):
        raise NotImplementedError
    
    @abstractmethod
    def random_batch(self):
        raise NotImplementedError


class FlexReplay(ABC):
    def __init__(self, max_size: int, field_specs: Optional[DictLike]=None, cache_max_size: Optional[int]=None, *args, **kwargs):
        super().__init__()
        self.field_specs = {}
        self.committed_fields = {}
        self.cache_fields = {}
        
        self._committed_max_size = int(max_size)
        self._cache_max_size = cache_max_size or int(self._committed_max_size * 0.1)
        self._committed_pointer = 0
        self._cache_pointer = 0
        self._cache_start = 0
        
        self.add_fields(field_specs)
        
    def reset(self):
        self.reset_committed()
        self.reset_cache()
        
    @abstractmethod
    def __len__(self):
        raise NotImplementedError
        
    @abstractmethod
    def add_fields(self):
        raise NotImplementedError
    
    @abstractmethod
    def add_sample(self):
        raise NotImplementedError
    
    @abstractmethod
    def random_batch(self):
        raise NotImplementedError
    
    @abstractmethod
    def reset_committed(self):
        raise NotImplementedError
    
    @abstractmethod
    def reset_cache(self):
        raise NotImplementedError
    
    @abstractmethod
    def commit(self):
        raise NotImplementedError
    
    @abstractmethod
    def get_cache_data(self):
        raise NotImplementedError
    