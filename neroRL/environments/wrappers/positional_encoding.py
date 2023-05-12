import numpy as np
from neroRL.environments.env import Env

class PositionalEncodingEnv(Env):
    """This wrapper concatenates positional encoding to vector observations."""

    def __init__(self, env):
        """Initializes the positoinal encoding.
        
        Arguments:
            env {Env} -- The to be wrapped environment, which is derived from the Env class
        """
        self._env = env

        # Prepare relative positional encoding
        sequence_length = 512
        n = 10000
        d = 16
        self.pos_encoding = np.zeros((512, d), dtype=np.float32)
        for k in range(sequence_length):
            for i in np.arange(int(d/2)):
                denominator = np.power(n, 2*i/d)
                self.pos_encoding[k, 2*i] = np.sin(k/denominator)
                self.pos_encoding[k, 2*i+1] = np.cos(k/denominator)

        # Udate the shape of the vector observation space
        if self._env.vector_observation_space is not None:
            self._vector_observation_space = (self._env.vector_observation_space[0] + d,)
        else:
            self._vector_observation_space = (d,)

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
    def ground_truth_space(self):
        """Returns the space of the ground truth info space if available."""
        return self._env.ground_truth_space

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
        """Reset the environment. The provided config is a dictionary featuring reset parameters of the environment such as the seed.
        
        Keyword Arguments:
            reset_params {dict} -- Reset parameters to configure the environment (default: {None})
        
        Returns:
            {numpy.ndarray} -- Visual observation
            {numpy.ndarray} -- Vector observation
        """
        # Track the current step of the episode
        self.t = 0
        vis_obs, vec_obs, info = self._env.reset(reset_params = reset_params)

        # Concatenate positional encoding
        if vec_obs is not None:
            vec_obs = np.concatenate((vec_obs, self.pos_encoding[self.t]))
        else:
            vec_obs = self.pos_encoding[self.t]

        return vis_obs, vec_obs, info

    def step(self, action):
        """Executes one step of the agent.
        
        Arguments:
            action {List} -- A list of at least one discrete action to be executed by the agent
        
        Returns:
            {numpy.ndarray} -- Visual observation
            {numpy.ndarray} -- Vector observation
            {float} -- Scalar reward signaled by the environment
            {bool} -- Whether the episode of the environment terminated
            {dict} -- Further episode information retrieved from the environment
        """
        vis_obs, vec_obs, reward, done, info = self._env.step(action)
        
        # Increment the current step of the episode
        self.t += 1

        # Concatenate positional encoding
        if vec_obs is not None:
            vec_obs = np.concatenate((vec_obs, self.pos_encoding[self.t]))
        else:
            vec_obs = self.pos_encoding[self.t]

        return vis_obs, vec_obs, reward, done, info

    def close(self):
        """Shuts down the environment."""
        self._env.close()
