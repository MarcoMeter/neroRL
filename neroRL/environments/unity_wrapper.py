import numpy as np
import logging
import random

from gym import error, spaces
from mlagents.envs.environment import UnityEnvironment
from neroRL.environments.env import Env

class UnityWrapper(Env):
    """This class wraps Unity environments.

    This wrapper has notable constraints:
        - Only one agent (no multi-agent environments).
        - Only one visual observation
        - Only discrete and multi-discrete action spaces (no continuous action space)"""

    def __init__(self, env_path, worker_id = 1, no_graphis = False, realtime_mode = False, config = None):
        """Instantiates the Unity Environment from a specified executable.
        
        Arguments:
            env_path {string} -- Path to the executable of the environment
        
        Keyword Arguments:
            worker_id {int} -- Port of the environment"s instance (default: {1})
            no_graphis {bool} -- Whether to allow the executable to render or not (default: {False})
            realtime_mode {bool} -- Whether to run the environment in real time or as fast as possible (default: {False})
            config {dict} -- Specifies the reset parameters of the environment (default: {None})
        """
        # Disable logging
        logging.disable(logging.INFO)

        self._config = config
        # Launch the environment"s executable
        self._env = UnityEnvironment(file_name = env_path, worker_id = worker_id, no_graphics = no_graphis)

        # Verify the environment
        # Verify brain count == 1
        if  len(self._env.brains) <= 0:
            raise UnityEnvironmentException("The unity environment {} does not provide any brains.".format(self._env.academy_name))
        elif len(self._env.brains) > 1:
            raise UnityEnvironmentException("The unity environment {} has more than one brain.".format(self._env.academy_name))
        # The number of agents is verified within reset()

        self._default_brain = self._env.brain_names[0]
        self._brain = self._env.brains[self._default_brain]
        self._realtime_mode = realtime_mode

        # Set action space property
        if self._brain.vector_action_space_type == "discrete":
            if len(self._brain.vector_action_space_size) == 1:
                self._action_space = spaces.Discrete(self._brain.vector_action_space_size[0])
            else:
                self._action_space = spaces.MultiDiscrete(self._brain.vector_action_space_size)
        else:
            raise UnityEnvironmentException("Action space type: {}. This wrapper supports discrete and multidiscrete Unity environments only!".format(self._brain.vector_action_space_type))
        self._action_names = self._brain.vector_action_descriptions
        
        # Set visual observation space property
        if self._brain.number_visual_observations == 1:
            width = self._brain.camera_resolutions[0]["width"]
            height = self._brain.camera_resolutions[0]["height"]
            depth = 1 if self._brain.camera_resolutions[0]["blackAndWhite"] else 3
            self._visual_observation_space = spaces.Box(
                low = 0,
                high = 1.0,
                shape = (height, width, depth),
                dtype = np.float32)
        elif self._brain.number_visual_observations > 1:
            raise UnityEnvironmentException("Only one visual observation is supported: ".format(self._brain.number_visual_observations))
        else:
            self._visual_observation_space = None

        # Set vector observation space property
        if self._brain.vector_observation_space_size > 0:
            self._vector_observatoin_space = (self._brain.vector_observation_space_size, )
        else:
            self._vector_observatoin_space = None

    @property
    def unwrapped(self):
        """        
        Returns:
            {UnityWrapper} -- Environment in its vanilla (i.e. unwrapped) state
        """
        return self
    
    @property
    def action_space(self):
        """Returns the shape of the action space of the agent."""
        return self._action_space

    @property
    def action_names(self):
        return self._action_names

    @property
    def visual_observation_space(self):
        return self._visual_observation_space

    @property
    def vector_observation_space(self):
        return self._vector_observatoin_space

    def reset(self, reset_params = None):
        """Resets the environment based on a global or just specified config.
        
        Keyword Arguments:
            config {dict} -- Reset parameters to configure the environment (default: {None})
        
        Returns:
            {numpy.ndarray} -- Visual observation
            {numpy.ndarray} -- Vector observation
        """
        # Track rewards of an entire episode
        self._rewards = []

        # Process config: Either load global or new config (if specified)
        if reset_params is None:
            reset_params = {}
            if self._config is not None:
                reset_params = self._config
        else:
            reset_params = reset_params

        # Reset the environment and retrieve the initial observation
        # env_info = self._env.reset(config = reset_params, train_mode = not self._realtime_mode)[self._default_brain]
        env_info = self._env.reset(config = {"tower-seed": random.randint(0, 99)}, train_mode = not self._realtime_mode)[self._default_brain]

        # Verify if only one agent is present
        if len(env_info.agents) != 1:
            raise UnityEnvironmentException("Only one agent is supported: " + str(len(env_info.agents)))

        # Seperate visual and vector observations
        if env_info.visual_observations:
            vis_obs = env_info.visual_observations[0][0][:, :, :]
        else:
            vis_obs = None

        if len(env_info.vector_observations[0]) > 0:
            vec_obs = env_info.vector_observations[0]
        else:
            vec_obs = None
        
        return vis_obs, vec_obs

    def step(self, action):
        """Runs one timestep of the environment"s dynamics.
        Once an episode is done, reset() has to be called manually.
                
        Arguments:
            action {List} -- A list of at least one discrete action to be executed by the agent

        Returns:
            {numpy.ndarray} -- Visual observation
            {numpy.ndarray} -- Vector observation
            {float} -- (Total) Scalar reward signaled by the environment
            {bool} -- Whether the episode of the environment terminated
            {dict} -- Further episode information (e.g. cumulated reward) retrieved from the environment once an episode completed
        """
        # Carry out the agent"s action
        env_info = self._env.step(action)[self._default_brain]
        # Separate visual and vector observations
        if env_info.visual_observations:
            vis_obs = env_info.visual_observations[0][0][:, :, :]
        else:
            vis_obs = None
        if len(env_info.vector_observations[0]) > 0:
            vec_obs = env_info.vector_observations[0]
        else:
            vec_obs = None
        # Retrieve reward
        reward = env_info.rewards[0]
        self._rewards.append(reward)
        # Episode done?
        done = env_info.local_done[0]

        # Episode information
        if done:
            info = {"reward": sum(self._rewards),
                    "length": len(self._rewards)}
        else:
            info = None

        return vis_obs, vec_obs, reward, done, info

    def close(self):
        """Shut down the environment."""
        self._env.close()
    
class UnityEnvironmentException(error.Error):
    """Any error related to running the Unity environment."""
    pass