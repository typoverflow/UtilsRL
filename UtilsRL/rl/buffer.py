import numpy as np

from abc import ABC, abstractmethod

class ReplayPool(ABC):
    def __init__(self, max_size, field_attrs, *args, **kwargs):
        super().__init__()
        
        self._max_size = int(max_size)
        self.fields = {}
        self.fields_attrs = {}
        self.add_fields(field_attrs)
        
        self._pointer = 0
        self._size = 0
        self._samples_since_save = 0
        
    @property
    def size(self):
        return self._size
    
    @property
    def field_names(self):
        return list(self.fields.keys())
    
    def clear(self):
        self._pointer = 0
        self._size = 0
        self._samples_since_save = 0
    
    def _advance(self, count=1):
        self._pointer = (self._pointer + count) % self._max_size
        self._size = min(self._size + count, self._max_size)
        self._samples_since_save += count
        
    def add_sample(self, sample):
        samples = {
            key: value[None, ...] for key, value in sample.items()
        }
        self.add_samples(samples)
        
    def add_fields(self, extra_fields, *args, **kwargs):
        self.fields_attrs.update(extra_fields)
        for field_name, field_attrs in extra_fields.items():
            field_shape = (self._max_size, *field_attrs["shape"])
            initializer = field_attrs.get("initializer", np.zeros)
            self.fields[field_name] = initializer(field_shape, dtype=field_attrs["dtype"])
    
    @abstractmethod
    def add_samples(self, samples, *args, **kwargs):
        raise NotImplementedError
    
    @abstractmethod
    def random_batch(self, batch_size, *args, **kwargs):
        raise NotImplementedError
    
    def random_indices(self, batch_size):
        if self._size == 0: 
            return np.arange(0, 0)
        elif batch_size == 0:
            idx = np.arange(0, self._size)
            np.random.shuffle(idx)
            return idx
        return np.random.randint(0, self._size, batch_size)
    
    def __getstate__(self):
        state = self.__dict__.copy()
        state["fields"] = {
            field_name: self.fields[field_name][:self._size] 
            for field_name in self.field_names
        }
        
    def __setstate__(self, state):
        if state['_size'] < state['_max_size']:
            pad_size = state['_max_size'] - state['_size']
            for field_name in state['fields'].keys():
                field_shape = state['fields_attrs'][field_name]['shape']
                state['fields'][field_name] = np.concatenate((
                    state['fields'][field_name],
                    np.zeros((pad_size, *field_shape))
                ), axis=0)

        self.__dict__ = state

    def get_placeholder(self, batch_size):
        samples = {}
        for field_name, field_attrs in self.fields_attrs.items():
            field_shape = (batch_size, *field_attrs["shape"])
            initializer = field_attrs.get("initializer", np.zeros)
            samples[field_name] = initializer(field_shape, dtype=field_attrs["dtype"])
        return samples

    
class TransitionReplayPool(ReplayPool):
    def __init__(self, observation_space, action_space, max_size, extra_fields={}, *args, **kwargs):
        self._observation_space = observation_space
        self._action_space = action_space
        
        fields = {
            "obs": {
                "shape": observation_space.shape, 
                "dtype": "float32"
            }, 
            "action": {
                "shape": action_space.shape, 
                "dtype": "float32"
            }, 
            "next_obs": {
                "shape": observation_space.shape, 
                "dtype": "float32"
            }, 
            "reward": {
                "shape": (1, ), 
                "dtype": "float32"
            }, 
            "done": {
                "shape": (1, ), 
                "dtype": "float32"
            }, 
        }
        for k, v in extra_fields.items():
            fields[k] = {
                "shape": v["shape"], 
                "dtype": v["dtype"]
            }
        
        super().__init__(max_size=max_size, field_attrs=fields, **kwargs)
        
    def add_fields(self, extra_fields):
        super().add_fields(extra_fields)

    def add_samples(self, samples):
        sample_fields = list(samples.keys())
        num_samples = samples[sample_fields[0]].shape[0]
        index_togo = np.arange(self._pointer, self._pointer + num_samples) % self._max_size
        for field_name in self.field_names:
            if field_name not in sample_fields:
                continue
            values = samples[field_name]
            self.fields[field_name][index_togo] = values
        
        self._advance(num_samples)
        
    def random_batch(self, batch_size, fields=None):
        idx = self.random_indices(batch_size)
        if fields is None:
            fields = self.field_names
        return {
            field_name: self.fields[field_name][idx] for field_name in fields
        }

class TrajectoryReplayPool(ReplayPool):
    def __init__(self, observation_space, action_space, max_size, max_traj_len, extra_fields={}, *args, **kwargs):
        self._observation_space = observation_space
        self._action_space = action_space
        self.max_traj_len = max_traj_len
        
        fields = {
            "obs": {
                "shape": tuple(self._observation_space.shape), 
                "dtype": "float32"
            }, 
            "action": {
                "shape": tuple(self._action_space.shape), 
                "dtype": "float32"
            }, 
            "next_obs": {
                "shape": tuple(self._observation_space.shape), 
                "dtype": "float32"
            }, 
            "reward": {
                "shape": (1, ), 
                "dtype": "float32"
            }, 
            "done": {
                "shape": (1, ), 
                "dtype": "float32"
            }, 
            "valid": {
                "shape": (1, ), 
                "dtype": "float32"
            }, 
        }
        for k, v in extra_fields.items():
            fields[k] = {
                "shape": tuple(v["shape"]), 
                "dtype": v["dtype"]
            }
        
        super().__init__(max_size=max_size, field_attrs=fields, **kwargs)
        
    def add_fields(self, extra_fields):
        for field_name, field_attr in extra_fields.items():
            extra_fields[field_name]["shape"] = (self.max_traj_len, *extra_fields[field_name]["shape"])
        super().add_fields(extra_fields)
        
    def add_samples(self, samples):
        def pad_or_trunc(name, values, max_len):
            if values.shape[1] >= max_len:
                return values[:, :max_len]
            else:
                initializer = self.fields_attrs.get(name, self.fields_attrs["obs"]).get("initializer", np.zeros)
                dtype = self.fields_attrs.get(name, self.fields_attrs["obs"]).get("dtype", "float32")
                padding = initializer((values.shape[0], max_len - values.shape[1], *values.shape[2:]), dtype=dtype)
                return np.concatenate([samples, padding], axis=1)
            
        sample_fields = list(samples.keys())
        num_samples = samples[sample_fields[0]].shape[0]
        index_togo = np.arange(self._pointer, self._pointer + num_samples) % self._max_size
        
        # check if we should pad or truncate the trajectories
        for field_name in self.field_names:
            if field_name not in sample_fields:
                continue
            values = samples[field_name]
            values = pad_or_trunc(field_name, values, self.max_traj_len)
            self.fields[field_name][index_togo] = values
            
        self._advance(num_samples)
        
    def random_batch(self, batch_size, fields=None):
        idx = self.random_indices(batch_size)
        if fields is None:
            fields = list(self.field_names)
        return {
            field_name: self.fields[field_name][idx] for field_name in fields
        }
        
    def random_batch_for_initial(self, batch_size):
        valids = np.sum(self.fields["valid"], axis=1).squeeze()[:self._size]
        first_idx = np.random.choice(np.arange(self._size), p=valids/np.sum(valids), size=(batch_size, ))
        second_idx = []
        for ind, item in enumerate(first_idx):
            second_idx.append(np.random.randint(valids[item]))
        indices = [(a, b) for a, b in zip(first_idx, second_idx)]
    
        return self.batch_by_double_index(indices)

    def batch_by_double_index(self, indices):
        batch = {}
        for field in self.field_names:
            shapes = self.fields[field].shape
            shapes = (len(indices), shapes[-1])
            data = np.zeros(shapes, dtype=np.float32)
            for ind, item in enumerate(indices):
                # print(self.fields[field].shape, data[ind].shape, self.fields[field][item[0], item[1]].shape, item)
                data[ind] = self.fields[field][item[0], item[1]]
            batch[field] = data
        return batch
    
        