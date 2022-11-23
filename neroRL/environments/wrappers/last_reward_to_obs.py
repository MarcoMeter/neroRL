import numpy as np
import gymnasium as gym
from gymnasium import spaces
from neroRL.environments.env import Env

class LastRewardToObs(Env):
    """This wrapper adds the last received reward to the agent's vector observation space."""

    def __init__(self, env):
        """        
        Arguments:
            env {Env} -- The to be wrapped environment, which is derived from the Env class
        """
        self._env = env

        # Derive the new vector observation space shape
        if self._env.vector_observation_space is None:
            self._vector_observation_space = (1,)
        else:
            self._vector_observation_space = (self._env.vector_observation_space[0] + 1,)

    @property
    def unwrapped(self):
        """Return this environment in its vanilla (i.e. unwrapped) state."""
        return self._env.unwrapped

    @property
    def visual_observation_space(self):
        """Returns the shape of the visual component of the observation space as a tuple."""
        return self._env.visual_observation_space

    @property
    def vector_observation_space(self):
        """Returns the shape of the vector component of the observation space as a tuple."""
        return self._vector_observation_space

    @property
    def action_space(self):
        """Returns the shape of the action space of the agent."""
        return self._env.action_space

    @property
    def seed(self):
        """Returns the seed of the current episode."""
        return self._env._seed

    @property
    def action_names(self):
        """Returns a list of action names. It has to be noted that only the names of action branches are provided and not the actions themselves!"""
        return self._env.action_names

    @property
    def get_episode_trajectory(self):
        """Returns the trajectory of an entire episode as dictionary (vis_obs, vec_obs, rewards, actions). 
        """
        return self._env.get_episode_trajectory

    def reset(self, reset_params = None):
        """Reset the environment. The provided reset_params is a dictionary featuring reset parameters of the environment such as the seed."""
        vis_obs, vec_obs = self._env.reset(reset_params = reset_params)

        # Concatenate the reward signal to the vector observation space
        if vec_obs is None:
            vec_obs = np.zeros(1, dtype=np.float32)
        else:
            vec_obs = np.concatenate((vec_obs, np.zeros(1, dtype=np.float32)), axis=0)

        return vis_obs, vec_obs

    def step(self, action):
        """Executes steps of the agent in the environment untill the "skip"-th frame is reached.
        
        Arguments:
            action {List} -- A list of at least one discrete action to be executed by the agent
        
        Returns:
                {numpy.ndarray} -- Visual observation
                {numpy.ndarray} -- Vector observation
                {float} -- (Total) Scalar reward signaled by the environment
                {bool} -- Whether the episode of the environment terminated
                {dict} -- Further episode information retrieved from the environment
        """
        vis_obs, vec_obs, reward, done, info = self._env.step(action)

        # Concatenate the reward signal to the vector observation space
        if vec_obs is None:
            vec_obs = np.array([reward], dtype=np.float32)
        else:
            vec_obs = np.concatenate((vec_obs, np.array([reward], dtype=np.float32)), axis=0)

        return vis_obs, vec_obs, reward, done, info

    def close(self):
        """Shuts down the environment."""
        self._env.close()