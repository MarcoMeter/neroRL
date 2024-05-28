import numpy as np
from collections import deque
from gymnasium import spaces
from neroRL.environments.env import Env

class StackedObservationEnv(Env):
    """This wrapper stacks visual and vector observations "num_stacks" times."""

    def __init__(self, env, num_stacks):
        """Initializes the wrapper by defining the new shapes of the observation spaces.
        
        Arguments:
            env {Env} -- The to be wrapped environment, which is derived from the Env class
            num_stacks {int} -- Number of observations that are to be stacked across "num_stacks" frames
        """
        self._env = env
        self._num_stacks = num_stacks

        assert (self._num_stacks > 0), "Number of stacks cannot be negative."

        # Modify visual observation space
        if self._env.visual_observation_space is not None:
            old_shape = self._env.visual_observation_space.shape
            self._visual_observation_space = spaces.Box(
                    low = 0,
                    high = 1.0,
                    shape = (old_shape[0], old_shape[1], old_shape[2] * num_stacks),
                    dtype = np.float32)
        else:
            self._visual_observation_space = None

        # Modify vector observation space
        if self._env.vector_observation_space is not None:
            self._vector_observation_space = (self._env.vector_observation_space[0] * num_stacks,)
        else:
            self._vector_observation_space = None

        # Visual observation stack
        self._vis_obs_stack = deque(maxlen = self._num_stacks) 
        # Vector observation stack
        self._vec_obs_stack = deque(maxlen = self._num_stacks)

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
            {numpy.ndarray} -- Stacked visual observation
            {numpy.ndarray} -- Stacked vector observation
        """
        vis_obs, vec_obs, info = self._env.reset(reset_params = reset_params)

        for _ in range(self._num_stacks):
            self._vis_obs_stack.append(vis_obs)
            self._vec_obs_stack.append(vec_obs)

        # Convert the stacks to numpy arrays
        vis_obs = self._process_vis_obs_stack(self._vis_obs_stack)
        vec_obs = self._process_vec_obs_stack(self._vec_obs_stack)

        return vis_obs, vec_obs, info

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
        
        self._vis_obs_stack.append(vis_obs)
        self._vec_obs_stack.append(vec_obs)

        # Convert the stacks to numpy arrays
        vis_obs = self._process_vis_obs_stack(self._vis_obs_stack)
        vec_obs = self._process_vec_obs_stack(self._vec_obs_stack)

        return vis_obs, vec_obs, reward, done, info

    def close(self):
        """Shuts down the environment."""
        self._env.close()

    def _process_vis_obs_stack(self, stacked_vis_obs):
        """Converts the list of observations to a numpy array that matches the defined shape of the stacked visual observation.
        
        Arguments:
            stacked_vis_obs {collections.deque} -- Visual observations stored in a deque list
        
        Returns:
            {numpy.ndarray} -- Returns the final numpy array. If there are no visual observations then return None.
        """
        if self._env.visual_observation_space is not None:
            return np.concatenate(stacked_vis_obs, axis=2)
        else:
            return None

    def _process_vec_obs_stack(self, stacked_vec_obs):
        """Converts the list of observations to a numpy array that matches the defined shape of the stacked vector observation.
        
        Arguments:
            stacked_vec_obs {collections.deque} -- Vector observations stored in a deque list
        
        Returns:
            {numpy.ndarray} -- Returns the final numpy array. If there are no vector observations then return None.
        """
        if self._env.vector_observation_space is not None:
            return np.concatenate(stacked_vec_obs, axis=0)
        else:
            return None