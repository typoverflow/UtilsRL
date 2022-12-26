import gym

class RewardClip(gym.Wrapper):
    def __init__(self, env: gym.Env, reward_min=None, reward_max=None):
        super().__init__(env)
        self.reward_min = reward_min
        self.reward_max = reward_max
        
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        if abs(reward) > 1:
            pass
        if self.reward_min is not None:
            reward = max(self.reward_min, reward)
        if self.reward_max is not None:
            reward = min(self.reward_max, reward)
        return obs, reward, done, info