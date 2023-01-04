import numpy as np
from gymnasium import spaces
from neroRL.environments.env import Env

class RewardNormalizer(Env):
    """This wrapper normalizes the reward of the agent by dividing through max_reward."""

    def __init__(self, env, max_reward = 1.0):
        """        
        Arguments:
            env {Env} -- The to be wrapped environment, which is derived from the Env class
        """
        self._env = env
        self._max_reward = max_reward

    @property
    def unwrapped(self):
        """Return this environment in its vanilla (i.e. unwrapped) state."""
        return self._env.unwrapped

    @property
    def visual_observation_space(self):
        """Returns the shape of the visual component of the observation space as a tuple."""
        if self._env.visual_observation_space is not None:
            return self._env.visual_observation_space
        else:
            return None

    @property
    def vector_observation_space(self):
        """Returns the shape of the vector component of the observation space as a tuple."""
        if self._env.vector_observation_space is not None:
            return self._env.vector_observation_space
        return None

    @property
    def action_space(self):
        """Returns the shape of the action space of the agent."""
        return self._env.action_space

    @property
    def max_episode_steps(self):
        """Returns the maximum number of steps that an episode can last."""
        return self._env.max_episode_steps

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

        return vis_obs, vec_obs

    def step(self, action):
        """Executes steps of the agent in the environment and normalizes the reward by dividing through its maximum reward.
        
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
        # Normalize reward
        reward /= self._max_reward

        return vis_obs, vec_obs, reward, done, info

    def close(self):
        """Shuts down the environment."""
        self._env.close()