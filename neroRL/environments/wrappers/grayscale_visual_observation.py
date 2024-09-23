import cv2
import numpy as np
from gymnasium import spaces
from neroRL.environments.env import Env

class GrayscaleVisualObsEnv(Env):
    """This wrapper converts all RGB (or multi-channel) visual observations to grayscale."""

    def __init__(self, env):
        """Initializes the wrapper by identifying image modalities and updating the observation space.
        
        Arguments:
            env {Env} -- The environment to be wrapped, which is derived from the Env class.
        """
        self._env = env

        # Identify keys that are images with more than 1 channel
        self._gray_keys = []
        new_spaces = {}
        for key, space in self._env.observation_space.spaces.items():
            if isinstance(space, spaces.Box):
                if len(space.shape) == 3 and space.shape[2] > 1:
                    self._gray_keys.append(key)
                    new_shape = (space.shape[0], space.shape[1], 1)
                    new_spaces[key] = spaces.Box(
                        low=0.0, high=1.0, shape=new_shape, dtype=np.float32
                    )
                else:
                    new_spaces[key] = space
            else:
                new_spaces[key] = space

        # Update the observation space
        self._observation_space = spaces.Dict(new_spaces)

    @property
    def unwrapped(self):
        """Return the unwrapped environment."""
        return self._env.unwrapped

    @property
    def observation_space(self):
        """Returns the updated observation space of the environment."""
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
        """Returns the trajectory of an entire episode as a dictionary (vis_obs, vec_obs, rewards, actions)."""
        return self._env.get_episode_trajectory

    def reset(self, reset_params=None):
        """Resets the environment and converts relevant visual observations to grayscale.
        
        Arguments:
            reset_params {dict} -- Reset parameters to configure the environment (default: {None})
        
        Returns:
            obs {dict} -- The updated observation dictionary with grayscale images.
            info {dict} -- Additional information from the environment.
        """
        obs, info = self._env.reset(reset_params=reset_params)
        for key in self._gray_keys:
            if key in obs:
                obs[key] = self._vis_obs_to_gray(obs[key])

        return obs, info

    def step(self, action):
        """Executes one step in the environment and converts relevant visual observations to grayscale.
        
        Arguments:
            action -- The action to be executed by the agent.
        
        Returns:
            obs {dict} -- The updated observation dictionary with grayscale images.
            reward {float} -- The reward received from the environment.
            done {bool} -- Whether the episode has terminated.
            info {dict} -- Additional information from the environment.
        """
        obs, reward, done, info = self._env.step(action)
        for key in self._gray_keys:
            if key in obs:
                obs[key] = self._vis_obs_to_gray(obs[key])
        return obs, reward, done, info

    def close(self):
        """Closes the environment."""
        self._env.close()

    def _vis_obs_to_gray(self, vis_obs):
        """Converts a multi-channel visual observation to grayscale.
        
        Arguments:
            vis_obs {numpy.ndarray} -- The visual observation to be converted.
        
        Returns:
            {numpy.ndarray} -- The grayscale visual observation with a single channel.
        """
        vis_obs = vis_obs.astype(np.float32)
        vis_obs = cv2.cvtColor(vis_obs, cv2.COLOR_RGB2GRAY)
        vis_obs = np.expand_dims(vis_obs, axis=2)
        return vis_obs
