import numpy as np
from gymnasium import spaces
from neroRL.environments.env import Env

class PositionalEncodingEnv(Env):
    """This wrapper adds positional encoding as a new modality to the observation space."""

    def __init__(self, env):
        """Initializes the positional encoding and updates the observation space.
        
        Arguments:
            env {Env} -- The environment to be wrapped, which is derived from the Env class.
        """
        self._env = env
        self.t = 0

        # Prepare absolute positional encoding
        sequence_length = 512
        n = 10000
        d = 16 
        self.pos_encoding = np.zeros((sequence_length, d), dtype=np.float32)
        for k in range(sequence_length):
            for i in range(d // 2):
                denominator = np.power(n, 2 * i / d)
                self.pos_encoding[k, 2 * i] = np.sin(k / denominator)
                self.pos_encoding[k, 2 * i + 1] = np.cos(k / denominator)

        if isinstance(self._env.observation_space, spaces.Dict):
            new_spaces = self._env.observation_space.spaces.copy()
            new_spaces["positional_encoding"] = spaces.Box(low=-np.inf, high=np.inf, shape=(d,), dtype=np.float32)
            self._observation_space = spaces.Dict(new_spaces)
        else:
            raise TypeError("The environment's observation space must be spaces.Dict.")
        
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
        """Returns the ground truth info space if available."""
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
        """Returns the trajectory of an entire episode."""
        return self._env.get_episode_trajectory

    def reset(self, reset_params=None):
        """Reset the environment and initialize the positional encoding.
        
        Keyword Arguments:
            reset_params {dict} -- Reset parameters to configure the environment (default: {None})
        
        Returns:
            obs {dict} -- Observation dictionary including "positional_encoding".
            info {dict} -- Additional information from the environment.
        """
        obs, info = self._env.reset(reset_params=reset_params)
        self.t = 0
        obs["positional_encoding"] = self.pos_encoding[self.t]
        return obs, info

    def step(self, action):
        """Executes one step in the environment and updates the positional encoding.
        
        Arguments:
            action -- The action to be executed by the agent.
        
        Returns:
            obs {dict} -- Observation dictionary including "positional_encoding".
            reward {float} -- Reward received from the environment.
            done {bool} -- Whether the episode has terminated.
            info {dict} -- Additional information from the environment.
        """
        obs, reward, done, info = self._env.step(action)
        self.t += 1
        obs["positional_encoding"] = self.pos_encoding[self.t]
        return obs, reward, done, info

    def close(self):
        """Shuts down the environment."""
        self._env.close()
