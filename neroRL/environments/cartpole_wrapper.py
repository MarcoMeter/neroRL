import numpy as np
import gym
import time
from gym import error, spaces
from neroRL.environments.env import Env

class CartPoleWrapper(Env):
    """This class wraps Gym CartPole environments.
    https://gym.openai.com/docs/#environments
    Available Environments:
        CartPole-v0
        CartPole-v1
    """

    def __init__(self, env_name, reset_params = None, realtime_mode = False):
        """Instantiates the CartPole environment.

        Arguments:
            env_name {string} -- Name of the CartPole environment

        Keyword Arguments:
            reset_params {dict} -- Provides parameters, like if the velocity, which is part of the observation, should be masked. (default: {None})
        """
        # Set default reset parameters if none were provided
        if reset_params is None:
            self._default_reset_params = {"mask-velocity": False}
        else:
            self._default_reset_params = reset_params
        self._realtime_mode = realtime_mode

        # Initialize environment
        self._env_name = env_name
        self._env = gym.make(self._env_name)

        # Prepare observation space
        self._vector_observation_space = self._env.observation_space.shape
        # Create mask to hide the velocity of the cart and the pole if requested by the reset params
        self._obs_mask = np.ones(4, dtype=np.float32) if not self._default_reset_params["mask-velocity"] else np.asarray([1,0,1,0], dtype=np.float32)

    @property
    def unwrapped(self):
        """Return this environment in its vanilla (i.e. unwrapped) state."""
        return self

    @property
    def visual_observation_space(self):
        """Returns the shape of the visual component of the observation space as a tuple."""
        return None

    @property
    def vector_observation_space(self):
        """Returns the shape of the vector component of the observation space as a tuple."""
        return self._vector_observation_space

    @property
    def action_space(self):
        """Returns the shape of the action space of the agent."""
        return self._env.action_space

    @property
    def action_names(self):
        """Returns a list of action names."""
        return ["move right", "move left"]

    def reset(self, reset_params = None):
        """Resets the environment.
        
        Keyword Arguments:
            reset_params {dict} -- Provides parameters, like if the observed velocity should be masked. (default: {None})
        
        Returns:
            {numpy.ndarray} -- Visual observation
            {numpy.ndarray} -- Vector observation
        """
        # Set default reset parameters if none were provided
        if reset_params is None:
            reset_params = self._default_reset_params
        # Create mask to hide the velocity of the cart and the pole if requested by the reset params
        self._obs_mask = np.ones(4, dtype=np.float32) if not self._default_reset_params["mask-velocity"] else np.asarray([1,0,1,0], dtype=np.float32)

        # Track rewards of an entire episode
        self._rewards = []

        # Retrieve the agent's initial observation
        vis_obs = None
        vec_obs = self._env.reset()

        return vis_obs, vec_obs * self._obs_mask

    def step(self, action):
        """Runs one timestep of the environment's dynamics.
        
        Arguments:
            action {int} -- The to be executed action
        
        Returns:
            {numpy.ndarray} -- Visual observation
            {numpy.ndarray} -- Vector observation
            {float} -- (Total) Scalar reward signaled by the environment
            {bool} -- Whether the episode of the environment terminated
            {dict} -- Further information (e.g. episode length) retrieved from the environment once an episode completed
        """
        # Render environment?
        if self._realtime_mode:
            self._env.render()
            time.sleep(0.033)

        # Execute action
        obs, reward, done, info = self._env.step(action[0])
        self._rewards.append(reward)
        # Retrieve the agent's current observation
        vis_obs = None
        vec_obs = obs

        # Wrap up episode information once completed (i.e. done)
        if done:
            info = {"reward": sum(self._rewards),
                    "length": len(self._rewards)}
        else:
            info = None

        return vis_obs, vec_obs * self._obs_mask, reward / 100.0, done, info

    def close(self):
        """Shuts down the environment."""
        self._env.close()
