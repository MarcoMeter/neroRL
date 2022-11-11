import numpy as np
from neroRL.environments.env import Env

class PositionalEncodingEnv(Env):
    """This wrapper adds positional encoding to visual observation or concatenates positional encoding to vector observations."""

    def __init__(self, env):
        """Initializes the positoinal encoding.
        
        Arguments:
            env {Env} -- The to be wrapped environment, which is derived from the Env class
        """
        self._env = env

        # Prepare positional encoding
        self._vector_observation_space = self._env.vector_observation_space
        sequence_length = 512
        n = 10000
        if self._env.visual_observation_space is not None:
            d = self._env.visual_observation_space.shape[0] * self._env.visual_observation_space.shape[0]
        else:
            d = 32
            # Udate the shape of the vector observation space
            self._vector_observation_space = (self._env.vector_observation_space[0] + d,)
        self.pos_encoding = np.zeros((512, d))
        for k in range(sequence_length):
            for i in np.arange(int(d/2)):
                denominator = np.power(n, 2*i/d)
                self.pos_encoding[k, 2*i] = np.sin(k/denominator)
                self.pos_encoding[k, 2*i+1] = np.cos(k/denominator)

    @property
    def unwrapped(self):
        """Return this environment in its vanilla (i.e. unwrapped) state."""
        return self._env.unwrapped

    @property
    def visual_observation_space(self):
        """Returns the shape of the visual component of the observation space as a tuple."""
        return self._env._visual_observation_space

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
        """Reset the environment. The provided config is a dictionary featuring reset parameters of the environment such as the seed.
        
        Keyword Arguments:
            reset_params {dict} -- Reset parameters to configure the environment (default: {None})
        
        Returns:
            {numpy.ndarray} -- Resized visual observation
            {numpy.ndarray} -- Vector observation
        """
        # Track the current step of the episode
        self.t = 0
        vis_obs, vec_obs = self._env.reset(reset_params = reset_params)

        # Apply positional encoding
        if self._env.visual_observation_space is not None:
            vis_obs = self._add_visual_positional_encoding(vis_obs, self.t)
        else:
            vec_obs = self._cat_positional_encoding(vec_obs, self.t)

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
        
        # Increment the current step of the episode
        self.t += 1

        # Apply positional encoding
        if self._env.visual_observation_space is not None:
            vis_obs = self._add_visual_positional_encoding(vis_obs, self.t)
        else:
            vec_obs = self._cat_positional_encoding(vec_obs, self.t)

        return vis_obs, vec_obs, reward, done, info

    def close(self):
        """Shuts down the environment."""
        self._env.close()

    def _add_visual_positional_encoding(self, vis_obs, t):
        """Adds positional encoding to to the visual observation.
        
        Arguments:
            vis_obs {numpy.ndarray} -- The source visual observation
            t {int} -- The current step of the episode
        
        Returns:
            {numpy.ndarray} -- The visual observation with the positional encoding
        """
        encoding = self.pos_encoding[t].reshape(self._env.visual_observation_space.shape[:2])
        vis_obs = vis_obs + np.expand_dims(encoding, axis=2)
        return vis_obs

    def _cat_positional_encoding(self, vec_obs, t):
        """Concatenate positional encoding to the vector observation

        Arguments:
            vec_obs {numpy.ndarray} -- The source visual observation
            t {int} -- The current step of the episode

        Returns:
            {numpy.ndarray} -- The vector observation with the positional encoding
        """
        return np.concatenate((vec_obs, self.pos_encoding[t]))
