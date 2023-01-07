import gym
import numpy as np

class MujocoParamOverWrite(gym.Wrapper):
    def __init__(self, env, overwrite_args: dict={}, do_scale: bool=False):
        super().__init__(env)
        for _key, _value in overwrite_args.items():
            if _key == "gravity":
                if isinstance(_value, float):
                    new_value = np.asarray([0, 0, _value])
                else:
                    new_value = np.asarray(_value)
                if do_scale:
                    new_value = new_value * self.unwrapped.model.opt.gravity
                self.unwrapped.model.opt.gravity[:] = new_value
            elif _key == "dof_damping":
                new_value = np.asarray(_value)
                if do_scale:
                    new_value = new_value * self.unwrapped.model.dof_damping
                self.unwrapped.model.dof_damping[:] = new_value
            elif _key == "body_mass":
                new_value = _value
                if do_scale:
                    new_value = new_value * self.unwrapped.model.body_mass
                self.unwrapped.model.body_mass[:] = new_value
            elif _key == "wind":
                new_value = _value
                if do_scale:
                    new_value = new_value * self.unwrapped.model.opt.wind
                self.unwrapped.model.opt.wind[:] = new_value
            else:
                raise ValueError(f"Unsupported type for mujoco param overwriting: {_key}")
            
