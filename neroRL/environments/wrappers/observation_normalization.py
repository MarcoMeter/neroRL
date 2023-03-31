from neroRL.environments.env import Env
from gymnasium.spaces import Box, MultiDiscrete, Discrete, Tuple
import numpy as np

class ObservationNorm(Env):
    """This wrapper normalizes the observation to the range [0, 1].

    Arguments:
        Env {Env} -- The to be wrapped environment that needs normalized observations.
    """
    def __init__(self, env):
        self._env = env
        self.observation_space = env.observation_space

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
        return self._env.vector_observation_space

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
        vec_obs = self.normalize(vec_obs, self._env.observation_space)
        
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
        vec_obs = self.normalize(vec_obs, self._env.observation_space)

        return vis_obs, vec_obs, reward, done, info

    def normalize(self, vec_obs, observation_space):
        """Normalizes the observation to the range [0, 1].

        Arguments:
            vec_obs {np.ndarray} -- Vector observation
            observation_space {gymnasium.spaces} -- Observation space of the environment

        Returns:
            {np.ndarray} -- Normalized vector observation
        """
        if isinstance(observation_space, Box):
            vec_obs = self.box_normalize(vec_obs, observation_space)
        elif isinstance(observation_space, MultiDiscrete):
            vec_obs = self.multi_discrete_normalize(vec_obs, observation_space)
        elif isinstance(observation_space, Discrete):
            vec_obs = self.discrete_normalize(vec_obs, observation_space)
        elif isinstance(observation_space, Tuple):
            vec_obs = self.tuple_normalize(vec_obs, observation_space)
        return vec_obs
    
    def box_normalize(self, vec_obs):
        """Normalize Box observation space to [0, 1].
        
        Arguments:
            vec_obs {np.ndarray} -- Vector observation
            observation_space {gymnasium.spaces.Box} -- Box observation space
        Returns:
            {np.ndarray} -- Normalized vector observation
        """
        lower, upper = self._env.observation_space.low, self._env.observation_space.high
        
        norm_vec_obs = []
        for low, up, val in zip(lower, upper, vec_obs):
            norm_vec_obs.append((val - low) / (up - low))
            
        return np.array(norm_vec_obs)
            
    def multi_discrete_normalize(self, vec_obs, observation_space):
        """Normalize MultiDiscrete observation space to [0, 1].
            
        Arguments:
            vec_obs {np.ndarray} -- Vector observation
            observation_space {gymnasium.spaces.MultiDiscrete} -- MultiDiscrete observation space
        
        Returns:
            {np.ndarray} -- Normalized vector observation
        """
        new_vec_obs = []
        for val, obs_space in zip(observation_space, vec_obs):
            new_vec_obs.append(self.discrete_normalize(val, obs_space))
        return np.array(new_vec_obs)
    
    def discrete_normalize(self, vec_obs, observation_space):
        """Normalize Discrete observation space to [0, 1].
        
        Arguments:
            vec_obs {np.ndarray} -- Vector observation
            observation_space {gymnasium.spaces.Discrete} -- Discrete observation space
        
        Returns:
            {np.ndarray} -- Normalized vector observation
        """
        num_obs = observation_space.n
        vec_obs /= num_obs
        return vec_obs
    
    def tuple_normalize(self, vec_obs, observation_space):
        """Normalize Tuple observation space to [0, 1].
        
        Arguments:
            vec_obs {np.ndarray} -- Vector observation
            observation_space {gymnasium.spaces.Tuple} -- Tuple observation space
        
        Returns:
            {np.ndarray} -- Normalized vector observation
        """
        new_vec_obs = []
        for val, obs_space in zip(observation_space, vec_obs):
            new_vec_obs.append(self.normalize(val, obs_space))
        return np.array(new_vec_obs)

    def close(self):
        """Shuts down the environment."""
        self._env.close()