import numpy as np
import gymnasium as gym
from gymnasium import spaces
from neroRL.environments.env import Env

class LastActionToObs(Env):
    """This wrapper adds the last taken action (one-hot) to the agent's observation space as a separate entry."""

    def __init__(self, env):
        """        
        Arguments:
            env {Env} -- The environment to be wrapped, which is derived from the Env class.
        """
        self._env = env

        # Calculate the total number of actions for one-hot encoding
        self._action_space = self._env.action_space
        if isinstance(self._action_space, spaces.Discrete):
            self._action_space_shape = (self._action_space.n,)
        elif isinstance(self._action_space, spaces.MultiDiscrete):
            self._action_space_shape = tuple(self._action_space.nvec)
        else:
            raise TypeError("Unsupported action space type")
        self._num_actions = sum(self._action_space_shape)

        # Add the "last_action" observation space
        self._last_action_space = spaces.Box(
            low=0.0, high=1.0, shape=(self._num_actions,), dtype=np.float32
        )
        new_spaces = self._env.observation_space.spaces.copy()
        new_spaces["last_action"] = self._last_action_space
        self._observation_space = spaces.Dict(new_spaces)

    @property
    def unwrapped(self):
        """Return this environment in its vanilla (i.e., unwrapped) state."""
        return self._env.unwrapped

    @property
    def observation_space(self):
        """Returns the updated observation space of the wrapped environment."""
        return self._observation_space

    @property
    def ground_truth_space(self):
        """Returns the space of the ground truth info space if available."""
        return self._env.ground_truth_space

    @property
    def action_space(self):
        """Returns the action space of the agent."""
        return self._env.action_space

    @property
    def max_episode_steps(self):
        """Returns the maximum number of steps that an episode can last."""
        return self._env.max_episode_steps

    @property
    def seed(self):
        """Returns the seed of the current episode."""
        return self._env.seed

    @property
    def action_names(self):
        """Returns a list of action names."""
        return self._env.action_names

    @property
    def get_episode_trajectory(self):
        """Returns the trajectory of an entire episode as a dictionary."""
        return self._env.get_episode_trajectory

    def reset(self, reset_params=None):
        """Reset the environment and initialize "last_action" to zeros."""
        obs, info = self._env.reset(reset_params=reset_params)
        obs["last_action"] = np.zeros(self._num_actions, dtype=np.float32)
        return obs, info

    def step(self, action):
        """Execute a step in the environment and update "last_action"."""
        obs, reward, done, info = self._env.step(action)
        one_hot_action = self._action_to_one_hot(action)
        obs["last_action"] = one_hot_action
        return obs, reward, done, info

    def _action_to_one_hot(self, action):
        """Converts the action to a one-hot encoded vector.
        
        Arguments:
            action {int or list} -- The action to be converted.
            
        Returns:
            np.ndarray -- The one-hot encoded action.
        """
        one_hot_action = np.zeros(self._num_actions, dtype=np.float32)
        if isinstance(self._action_space, spaces.Discrete):
            one_hot_action[action] = 1.0
        elif isinstance(self._action_space, spaces.MultiDiscrete):
            index = 0
            for i, action_size in enumerate(self._action_space_shape):
                one_hot_action[index + action[i]] = 1.0
                index += action_size
        return one_hot_action

    def close(self):
        """Shuts down the environment."""
        self._env.close()
