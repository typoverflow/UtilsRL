from typing import Dict
import copy
import gym
from gym.spaces import Box
import numpy as np

from UtilsRL.env.wrapper.compat import _parse_reset_result, _format_reset_result, _parse_step_result, _format_step_result

class SinglePrecision(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

        if isinstance(self.observation_space, Box):
            obs_space = self.observation_space
            self.observation_space = Box(obs_space.low, obs_space.high,
                                         obs_space.shape)
        elif isinstance(self.observation_space, Dict):
            obs_spaces = copy.copy(self.observation_space.spaces)
            for k, v in obs_spaces.items():
                obs_spaces[k] = Box(v.low, v.high, v.shape)
            self.observation_space = Dict(obs_spaces)
        else:
            raise NotImplementedError

    def observation(self, observation: np.ndarray) -> np.ndarray:
        if isinstance(observation, np.ndarray):
            return observation.astype(np.float32)
        elif isinstance(observation, dict):
            observation = copy.copy(observation)
            for k, v in observation.items():
                observation[k] = v.astype(np.float32)
            return observation


class BatchObservationWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        
    def reset(self, **kwargs):
        res, info, contains_info = _parse_reset_result(self.env.reset(**kwargs))
        if isinstance(res, np.ndarray):
            res = res[None, :]
        elif isinstance(res, dict):
            for _key, _value in res.items():
                res[_key] = _value[None, :]
        return _format_reset_result(res, info, contains_info)

    def step(self, action):
        action = np.squeeze(action)
        obs, *res, new_api = _parse_step_result(self.env.step(action))
        if isinstance(obs, np.ndarray):
            obs = obs[None, :]
        elif isinstance(obs, dict):
            for _key, _value in obs.items():
                obs[_key] = _value[None, :]
        return _format_reset_result(obs, *res, new_api)