import cv2
import numpy as np
from gym import spaces
from neroRL.environments.env import Env

class GrayscaleVisualObsEnv(Env):
    """This wrapper turns RGB visual observations to grayscale visual observations."""

    def __init__(self, env):
        """Initializes the wrapped by defining the new shape of the visual observation space.
        
        Arguments:
            env {Env} -- The to be wrapper environment, which is derived from the Env class
        """
        self._env = env

        # Check if visual observations are available
        assert (self._env.visual_observation_space is not None), "Visual observations of the environment have to be available."

        # Modify visual observation space
        if self._env.visual_observation_space is not None:
            old_shape = self._env.visual_observation_space.shape
            self._visual_observation_space = spaces.Box(
                    low = 0,
                    high = 1.0,
                    shape = (old_shape[0], old_shape[1], old_shape[2] // 3),
                    dtype = np.float32)
        else:
            self._visual_observation_space = None

    @property
    def unwrapped(self):
        """Return this environment in its vanilla (i.e. unwrapped) state."""
        return self._env.unwrapped

    @property
    def visual_observation_space(self):
        """Returns the shape of the visual component of the observation space as a tuple."""
        return self._visual_observation_space

    @property
    def vector_observation_space(self):
        """Returns the shape of the vector component of the observation space as a tuple."""
        return self._env.vector_observation_space

    @property
    def action_space(self):
        """Returns the shape of the action space of the agent."""
        return self._env.action_space

    @property
    def action_names(self):
        """Returns a list of action names. It has to be noted that only the names of action branches are provided and not the actions themselves!"""
        return self._env.action_names

    def reset(self, reset_params = None):
        """Reset the environment. The provided config is a dictionary featuring reset parameters of the environment such as the seed.
        
        Keyword Arguments:
            reset_params {dict} -- Reset parameters to configure the environment (default: {None})
        
        Returns:
            {numpy.ndarray} -- Resized visual observation
            {numpy.ndarray} -- Vector observation
        """
        vis_obs, vec_obs = self._env.reset(reset_params = reset_params)

        # Convert RGB to Grayscale
        vis_obs = self._vis_obs_to_gray(vis_obs)

        return vis_obs, vec_obs

    def step(self, action):
        """Executes one step of the agent.
        
        Arguments:
            action {List} -- A list of at least one discrete action to be executed by the agent
        
        Returns:
            {numpy.ndarray} -- Stacked visual observation
            {numpy.ndarray} -- Stacked vector observation
            {float} -- Scalar reward signaled by the environment
            {bool} -- Whether the episode of the environment terminated
            {dict} -- Further episode information retrieved from the environment
        """
        vis_obs, vec_obs, reward, done, info = self._env.step(action)
        
        # Convert RGB to Grayscale
        vis_obs = self._vis_obs_to_gray(vis_obs)

        return vis_obs, vec_obs, reward, done, info

    def close(self):
        """Shuts down the environment."""
        self._env.close()

    def _vis_obs_to_gray(self, vis_obs):
        """Converts an RGB visual observsation to grayscale.
        
        Arguments:
            vis_obs {numpy.ndarray} -- The to be converted RGB visual observation
        
        Returns:
            {numpy.ndarray} -- Grayscale visual observation
        """
        vis_obs = vis_obs.astype(np.float32)
        vis_obs = cv2.cvtColor(vis_obs, cv2.COLOR_RGB2GRAY)
        vis_obs = np.expand_dims(vis_obs, axis = 2)
        return vis_obs