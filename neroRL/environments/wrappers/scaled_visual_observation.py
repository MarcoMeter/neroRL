import cv2
import numpy as np
from gym import spaces
from neroRL.environments.env import Env

class ScaledVisualObsEnv(Env):
    """This wrapper resizes visual observations."""

    def __init__(self, env, width, height):
        """Initializes the wrapper by defining the new shape of the visual observation space.
        
        Arguments:
            env {Env} -- The to be wrapped environment, which is derived from the Env class
            width {int} -- The width of the resized visual observation
            height {int} -- The height of the resized visual observation
        """
        self._env = env
        self._width = width
        self._height = height

        # Check if visual observations are available
        assert (self._env.visual_observation_space is not None), "Visual observations of the environment have to be available."
        # Check inputs
        assert (self._width > 0 and self._height > 0), "Image dimensions have to be greater than 0."

        # Modify visual observation space
        if self._env.visual_observation_space is not None:
            old_shape = self._env.visual_observation_space.shape
            self._visual_observation_space = spaces.Box(
                    low = 0,
                    high = 1.0,
                    shape = (self._width, self._height, old_shape[2]),
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

        # Process visual observation
        vis_obs = self._resize_vis_obs(vis_obs)

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
        
        # Process visual observation
        vis_obs = self._resize_vis_obs(vis_obs)

        return vis_obs, vec_obs, reward, done, info

    def close(self):
        """Shuts down the environment."""
        self._env.close()

    def _resize_vis_obs(self, visual_observation):
        """Resizes the visual observation based on member vairables.
        
        Arguments:
            visual_observation {numpy.ndarray} -- The to be resized visual observation
        
        Returns:
            {numpy.ndarray} -- The resized visual observation
        """
        visual_observation = cv2.resize(visual_observation, (self._width, self._height), interpolation=cv2.INTER_AREA)
        # Ensure that the visual observation has a channel dimension
        if len(visual_observation.shape) == 2:
            visual_observation = np.expand_dims(visual_observation, axis = 2)
        return visual_observation 