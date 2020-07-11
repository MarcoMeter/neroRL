import numpy as np
import logging
import random

from gym import error, spaces
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.environment_parameters_channel import (
    EnvironmentParametersChannel,
)
from mlagents_envs.side_channel.engine_configuration_channel import (
    EngineConfigurationChannel,
)
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

        # Initialize channels
        self.reset_parameters = EnvironmentParametersChannel()
        self.engine_config = EngineConfigurationChannel()

        self._config = config
        self._realtime_mode = realtime_mode
        if realtime_mode:
            self.engine_config.set_configuration_parameters(time_scale=1.0, width=1280, height=720)
        else:
            self.engine_config.set_configuration_parameters(time_scale=20.0, width=128, height=128)

        # Launch the environment's executable
        self._env = UnityEnvironment(file_name = env_path, worker_id = worker_id, no_graphics = no_graphis, side_channels=[self.reset_parameters, self.engine_config])
        # Reset the environment
        self._env.reset()
        # Retrieve behavior configuration
        self._behavior_name = list(self._env.behavior_specs)[0]
        self._behavior_spec = self._env.behavior_specs[self._behavior_name]

        # Set action space properties
        if len(self._behavior_spec.action_shape) == 1:
            self._action_space = spaces.Discrete(self._behavior_spec.action_shape[0])
        else:
            self._action_space = spaces.MultiDiscrete(self._behavior_spec.action_shape)
        self._action_names = ["Not available"]
        
        # Count visual and vector observations
        self._num_vis_obs, self._num_vec_obs = 0, 0
        self._vec_obs_indices = []
        for index, obs in enumerate(self._behavior_spec.observation_shapes):
            if len(obs) > 1:
                self._num_vis_obs = self._num_vis_obs + 1
                self._vis_obs_index = index
            else:
                self._num_vec_obs = self._num_vec_obs + 1
                self._vec_obs_indices.append(index)

        # Verify the environment
        self._verify_environment()

        # Set visual observation space property
        if self._num_vis_obs == 1:
            height = self._behavior_spec.observation_shapes[self._vis_obs_index][0]
            width = self._behavior_spec.observation_shapes[self._vis_obs_index][1]
            depth = self._behavior_spec.observation_shapes[self._vis_obs_index][2]
            self._visual_observation_space = spaces.Box(
                low = 0,
                high = 1.0,
                shape = (height, width, depth),
                dtype = np.float32)
        else:
            self._visual_observation_space = None

        # Set vector observation space property
        if self._num_vec_obs > 0:
            # Determine the length of vec obs by summing the length of each distinct one
            vec_obs_length = sum([self._behavior_spec.observation_shapes[i][0] for i in self._vec_obs_indices])
            self._vector_observatoin_space = (vec_obs_length, )
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

        # Apply reset parameters
        for key, value in reset_params.items():
            self.reset_parameters.set_float_parameter(key, value)

        # Reset and verify the environment
        self._env.reset()
        info, terminal_info = self._env.get_steps(self._behavior_name)
        self._verify_environment(len(info))
        
        # Retrieve initial observations
        vis_obs, vec_obs, _, _ = self._process_agent_info(info, terminal_info)
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
        # Carry out the agent's action
        self._env.set_actions(self._behavior_name, action.reshape([1, -1]))
        self._env.step()
        info, terminal_info = self._env.get_steps(self._behavior_name)

        # Process step results
        vis_obs, vec_obs, reward, done = self._process_agent_info(info, terminal_info)
        self._rewards.append(reward)

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

    def _process_agent_info(self, info, terminal_info):
        """Extracts the observations, rewards, dones, and episode infos.

        Args:
            info {DecisionSteps}: Current state
            terminal_info {TerminalSteps}: Terminal state

        Returns:
            vis_obs {ndarray} -- Visual observation if available, else None
            vec_obs {ndarray} -- Vector observation if available, else None
            reward {float} -- Reward signal from the environment
            done {bool} -- Whether the episode terminated or not
        """
        # Determine if the episode terminated or not
        if len(terminal_info) == 0:
            done = False
            use_info = info
        else:
            done = True
            use_info = terminal_info

        # Process visual observations
        if self.visual_observation_space is not None:
            vis_obs = use_info.obs[self._vis_obs_index][0]
        else:
            vis_obs = None

        # Process vector observations
        if self.vector_observation_space is not None:
            for i, dim in enumerate(self._vec_obs_indices):
                if i == 0:
                    vec_obs = use_info.obs[dim][0]
                else:
                    vec_obs = np.concatenate((vec_obs, use_info.obs[dim][0]))
        else:
            vec_obs = None

        return vis_obs, vec_obs, use_info.reward[0], done

    def _verify_environment(self, num_agents = None):
        """Checks if the environment meets the requirements of this wrapper.
        Only one agent and at maximum one visual observation is allowed.
        Only Discrete and MultiDiscrete action spaces are supported.

        Arguments:
            num_agents {int} -- Number of agents (default: {None})
        """
        # Verify number of agent types
        if len(self._env.behavior_specs) != 1:
            raise UnityEnvironmentException("The unity environment containts more than one agent type.")
        # Verify action space type
        if int(self._behavior_spec.action_type.value) == 1:
            raise UnityEnvironmentException("Continuous action spaces are not supported. Only discrete and MultiDiscrete spaces are supported.")
        # Verify number of visual observations
        if self._num_vis_obs > 1:
            raise UnityEnvironmentException("The unity environment contains more than one visual observation.")
        # Verify agent count
        if num_agents is not None and num_agents > 1:
            raise UnityEnvironmentException("The unity environment contains more than one agent.")
        


class UnityEnvironmentException(error.Error):
    """Any error related to running the Unity environment."""
    pass