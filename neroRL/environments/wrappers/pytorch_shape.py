import numpy as np
from gymnasium import spaces
from neroRL.environments.env import Env

class PyTorchEnv(Env):
    """This wrapper reshapes visual observations in the observation dictionary to match PyTorch's channels-first format."""

    def __init__(self, env):
        """Initializes the wrapper and modifies the observation space to reflect the changes in observation shapes.

        Arguments:
            env {Env} -- The environment to be wrapped.
        """
        self._env = env

        # Whenever the observation space is an image, swap the axes to pytorch format: (H, W, C) -> (C, H, W)
        original_observation_space = self._env.observation_space
        new_spaces = {}
        for key, space in original_observation_space.spaces.items():
            if isinstance(space, spaces.Box) and len(space.shape) == 3:
                old_shape = space.shape
                new_shape = (old_shape[2], old_shape[0], old_shape[1])
                new_spaces[key] = spaces.Box(
                    low=space.low.min(),
                    high=space.high.max(),
                    shape=new_shape,
                    dtype=space.dtype
                )
            else:
                new_spaces[key] = space
        self._observation_space = spaces.Dict(new_spaces)

    @property
    def unwrapped(self):
        """Returns the unwrapped environment."""
        return self._env.unwrapped

    @property
    def observation_space(self):
        """Returns the modified observation space of the environment."""
        return self._observation_space

    @property
    def ground_truth_space(self):
        """Returns the ground truth space if available."""
        return self._env.ground_truth_space

    @property
    def action_space(self):
        """Returns the action space of the agent."""
        return self._env.action_space

    @property
    def max_episode_steps(self):
        """Returns the maximum number of steps in an episode."""
        return self._env.max_episode_steps

    @property
    def seed(self):
        """Returns the seed of the current episode."""
        return self._env._seed

    @property
    def action_names(self):
        """Returns a list of action names."""
        return self._env.action_names

    @property
    def get_episode_trajectory(self):
        """Returns the trajectory of an entire episode as a dictionary."""
        return self._env.get_episode_trajectory

    def reset(self, reset_params=None):
        """Resets the environment.

        Arguments:
            reset_params {dict} -- Reset parameters of the environment such as the seed.

        Returns:
            obs {dict} -- Processed observation dictionary.
            info {dict} -- Additional information.
        """
        obs, info = self._env.reset(reset_params=reset_params)
        return self._process_observation(obs), info

    def step(self, action):
        """Executes one step in the environment. Ensures that image observations come in the right shape.
        If the action is a list of length 1, it is unpacked to allow for compatibility with environments that expect a single action.

        Arguments:
            action {List} -- Actions to be executed by the agent.

        Returns:
            obs {dict} -- Processed observation dictionary.
            reward {float} -- Scalar reward from the environment.
            done {bool} -- Whether the episode has terminated.
            info {dict} -- Additional information.
        """
        if isinstance(action, list):
            if len(action) == 1:
                action = action[0]
        obs, reward, done, info = self._env.step(action)
        return self._process_observation(obs), reward, done, info

    def _process_observation(self, obs):
        """Processes the observation dictionary to swap axes for image data.

        Arguments:
            obs {dict} -- The observation returned by the environment.

        Returns:
            obs {dict} -- The processed observation.
        """
        new_obs = {}
        for key, value in obs.items():
            if isinstance(value, np.ndarray) and value.ndim == 3:
                value = np.transpose(value, (2, 0, 1)) # Swap axes to make channels first
            new_obs[key] = value
        return new_obs

    def close(self):
        """Closes the environment."""
        self._env.close()
