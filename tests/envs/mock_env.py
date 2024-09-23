import numpy as np
from gymnasium import spaces
from neroRL.environments.env import Env

class MockEnv(Env):
    def __init__(self):
        self._unwrapped = self
        self._observation_space = spaces.Dict({
            "vec_obs": spaces.Box(low=0, high=1, shape=(10,), dtype=np.float32),
            "vis_obs": spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8)
        })
        self._action_space = spaces.Discrete(4)
        self._max_episode_steps = 10
        self._seed = 42
        self._action_names = ["up", "down", "left", "right"]
        self._trajectory = {"states": [], "actions": [], "rewards": []}
        self._mocked_rewards = list(range(10))
        self.t = 0

    @property
    def unwrapped(self):
        return self

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    @property
    def max_episode_steps(self):
        return self._max_episode_steps

    @property
    def seed(self):
        return self._seed

    @property
    def action_names(self):
        return self._action_names

    @property
    def get_episode_trajectory(self):
        return self._trajectory

    def reset(self, reset_params=None):
        self.t = 0
        self.rewards = []
        observation = {
            "vec_obs": np.zeros((10,), dtype=np.float32),
            "vis_obs": np.zeros((64, 64, 3), dtype=np.uint8)
        }
        return observation, {}

    def step(self, action):
        self.t += 1
        observation = {
            "vec_obs": np.ones((10,), dtype=np.float32) * self.t,
            "vis_obs": np.ones((64, 64, 3), dtype=np.uint8) * self.t
        }
        reward = self._mocked_rewards[self.t - 1]
        self.rewards.append(reward)
        done = self.t >= self._max_episode_steps
        if done:
            info = {"reward": sum(self._mocked_rewards),
                    "length": len(self._mocked_rewards)}
        else:
            info = {}
        
        return observation, reward, done, info

    def close(self):
        pass