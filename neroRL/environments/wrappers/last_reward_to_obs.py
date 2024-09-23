import numpy as np
import gymnasium as gym
from gymnasium import spaces
from neroRL.environments.env import Env

class LastRewardToObs(Env):
    """This wrapper adds the last received reward to the agent's observation space.
    If "last_action" is present, it concatenates the last reward to it and renames the key to "last_action_last_reward".
    Otherwise, it adds "last_reward" as a separate entry in the observation space dictionary.
    """

    def __init__(self, env):
        """        
        Arguments:
            env {Env} -- The environment to be wrapped, which is derived from the Env class.
        """
        self._env = env

        new_spaces = self._env.observation_space.spaces.copy()
        if "last_action" in new_spaces:
            last_action_space = new_spaces.pop("last_action")
            last_action_last_reward_space = spaces.Box(
                low=0,
                high=1,
                shape= (last_action_space.shape[0] + 1,),
                dtype=last_action_space.dtype
            )
            new_spaces["last_action_last_reward"] = last_action_last_reward_space
        else:
            new_spaces["last_reward"] = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(1,),
                dtype=np.float32
            )
        self._observation_space = spaces.Dict(new_spaces)

    @property
    def unwrapped(self):
        """Return this environment in its vanilla (i.e., unwrapped) state."""
        return self._env.unwrapped

    @property
    def observation_space(self):
        """Returns the updated observation space."""
        return self._observation_space

    @property
    def ground_truth_space(self):
        """Returns the ground truth info space if available."""
        return self._env.ground_truth_space

    @property
    def action_space(self):
        """Returns the action space of the agent."""
        return self._env.action_space

    @property
    def max_episode_steps(self):
        """Returns the maximum number of steps per episode."""
        return self._env.max_episode_steps

    @property
    def seed(self):
        """Returns the current episode seed."""
        return self._env.seed

    @property
    def action_names(self):
        """Returns a list of action names."""
        return self._env.action_names

    @property
    def get_episode_trajectory(self):
        """Returns the episode trajectory."""
        return self._env.get_episode_trajectory

    def reset(self, reset_params=None):
        """Reset the environment and initialize "last_reward"."""
        obs, info = self._env.reset(reset_params=reset_params)
        self._last_reward = 0.0
        obs = self._process_observation(obs, self._last_reward)
        return obs, info

    def step(self, action):
        """Take a step in the environment and update "last_reward"."""
        obs, reward, done, info = self._env.step(action)
        obs = self._process_observation(obs, self._last_reward)
        self._last_reward = reward
        return obs, reward, done, info

    def _process_observation(self, obs, last_reward):
        """Processes the observation to include "last_reward" appropriately.
        
        Arguments:
            obs {dict} -- The observation dictionary.
            last_reward {float} -- The last reward received.
            
        Returns:
            {dict} -- The updated observation dictionary.
        """
        if "last_action" in obs:
            last_action = obs.pop("last_action")
            last_action_last_reward = np.concatenate([last_action, [last_reward]], axis=0)
            obs["last_action_last_reward"] = last_action_last_reward
        else:
            obs["last_reward"] = np.array([last_reward], dtype=np.float32)
        return obs

    def close(self):
        """Shuts down the environment."""
        self._env.close()
