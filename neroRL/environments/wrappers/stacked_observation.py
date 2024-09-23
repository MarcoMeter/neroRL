import numpy as np
from collections import deque
from gymnasium import spaces
from neroRL.environments.env import Env

class StackedObservationEnv(Env):
    """This wrapper stacks observations over a specified number of time steps.
    It supports environments where observations are dictionaries (spaces.Dict).
    For image observations, it stacks them along the channel dimension.
    For 1D vector observations, it concatenates them along the feature dimension.
    """

    def __init__(self, env, num_stacks):
        """Initializes the wrapper by defining the new shapes of the observation spaces.
        
        Arguments:
            env {Env} -- The environment to be wrapped, which is derived from the Env class.
            num_stacks {int} -- Number of observations to stack over "num_stacks" time steps.
        """
        self._env = env
        self._num_stacks = num_stacks

        assert self._num_stacks > 0, "Number of stacks must be positive."

        # Get the keys to stack (modalities present at the time of wrapping)
        self._stacking_keys = set(self._env.observation_space.spaces.keys())

        # Create the stacked observation space for the stacking keys
        self._observation_space = self._create_stacked_obs_space(self._env.observation_space)

        # Initialize the observation stacks for each stacking key
        self._obs_stacks = {
            key: deque(maxlen=self._num_stacks) for key in self._stacking_keys
        }

    def _create_stacked_obs_space(self, original_obs_space):
        """Creates the stacked observation space based on the original observation space.

        Arguments:
            original_obs_space {spaces.Dict} -- The original observation space.

        Returns:
            spaces.Dict -- The new stacked observation space.
        """
        stacked_spaces = {}
        for key in original_obs_space.spaces.keys():
            space = original_obs_space.spaces[key]
            if isinstance(space, spaces.Box):
                if len(space.shape) == 3:
                    # For image observations (height, width, channels)
                    new_shape = (space.shape[0], space.shape[1], space.shape[2] * self._num_stacks)
                    # Repeat low and high along the channel dimension
                    low = np.repeat(space.low, self._num_stacks, axis=2)
                    high = np.repeat(space.high, self._num_stacks, axis=2)
                elif len(space.shape) == 1:
                    # For 1D vector observations
                    new_shape = (space.shape[0] * self._num_stacks,)
                    # Repeat low and high along the feature dimension
                    low = np.repeat(space.low, self._num_stacks, axis=0)
                    high = np.repeat(space.high, self._num_stacks, axis=0)
                else:
                    raise ValueError(f"Unsupported space shape {space.shape} for key '{key}'.")

                stacked_spaces[key] = spaces.Box(low=low, high=high, shape=new_shape, dtype=space.dtype)
            else:
                raise TypeError(
                    f"Unsupported space type {type(space)} for key '{key}'. Only spaces.Box is supported."
                )

        return spaces.Dict(stacked_spaces)

    @property
    def unwrapped(self):
        """Return the unwrapped environment."""
        return self._env.unwrapped

    @property
    def observation_space(self):
        """Returns the stacked observation space of the environment."""
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
        """Returns the seed of the current episode."""
        return self._env.seed

    @property
    def action_names(self):
        """Returns a list of action names."""
        return self._env.action_names

    @property
    def get_episode_trajectory(self):
        """Returns the trajectory of an entire episode."""
        return self._env.get_episode_trajectory

    def reset(self, reset_params=None):
        """Resets the environment and initializes the observation stacks.
        
        Arguments:
            reset_params {dict} -- Reset parameters to configure the environment (default: {None})
        
        Returns:
            obs {dict} -- The initial stacked observation.
            info {dict} -- Additional information from the environment.
        """
        obs, info = self._env.reset(reset_params=reset_params)

        # Initialize the observation stacks with the initial observation repeated
        for key in self._stacking_keys:
            self._obs_stacks[key].clear()
            for _ in range(self._num_stacks):
                self._obs_stacks[key].append(obs[key])

        # Build the stacked observations
        stacked_obs = {}
        for key in self._stacking_keys:
            stacked_obs[key] = self._get_stacked_obs_for_key(key)

        # Include other keys as-is (e.g., positional encoding)
        for key in obs:
            if key not in self._stacking_keys:
                stacked_obs[key] = obs[key]

        return stacked_obs, info

    def step(self, action):
        """Executes one step in the environment and updates the observation stacks.
        
        Arguments:
            action -- The action to be executed by the agent.
        
        Returns:
            obs {dict} -- The stacked observation after the step.
            reward {float} -- The reward received from the environment.
            done {bool} -- Whether the episode has terminated.
            info {dict} -- Additional information from the environment.
        """
        obs, reward, done, info = self._env.step(action)

        # Update the observation stacks with the new observation
        for key in self._stacking_keys:
            self._obs_stacks[key].append(obs[key])

        # Build the stacked observations
        stacked_obs = {}
        for key in self._stacking_keys:
            stacked_obs[key] = self._get_stacked_obs_for_key(key)

        # Include other keys as-is (e.g., positional encoding)
        for key in obs:
            if key not in self._stacking_keys:
                stacked_obs[key] = obs[key]

        return stacked_obs, reward, done, info

    def _get_stacked_obs_for_key(self, key):
        """Constructs the stacked observation for a specific key."""
        stack = self._obs_stacks[key]
        space = self._observation_space.spaces[key]
        if len(space.shape) == 3:
            # Stack images along the channel dimension (axis=2)
            stacked = np.concatenate(list(stack), axis=2)
        elif len(space.shape) == 1:
            # Concatenate vectors along the feature dimension (axis=0)
            stacked = np.concatenate(list(stack), axis=0)
        else:
            raise ValueError(f"Unsupported space shape {space.shape} for key '{key}'.")
        return stacked

    def close(self):
        """Closes the environment."""
        self._env.close()
