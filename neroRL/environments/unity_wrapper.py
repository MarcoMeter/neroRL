import numpy as np

from gymnasium import error, spaces
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple
from mlagents_envs.side_channel.environment_parameters_channel import (EnvironmentParametersChannel,)
from mlagents_envs.side_channel.engine_configuration_channel import (EngineConfigurationChannel,)
from neroRL.environments.env import Env
from random import randint

class UnityWrapper(Env):
    """This class wraps Unity environments.

    This wrapper has notable constraints:
        - Only one agent (no multi-agent environments).
        - Only one visual observation
        - Only discrete and multi-discrete action spaces (no continuous action space)"""

    def __init__(self, env_path, reset_params, worker_id = 1, no_graphis = False, realtime_mode = False,  record_trajectory = False):
        """Instantiates the Unity Environment from a specified executable.
        
        Arguments:
            env_path {string} -- Path to the executable of the environment
            reset_params {dict} -- Reset parameters of the environment such as the seed
        
        Keyword Arguments:
            worker_id {int} -- Port of the environment"s instance (default: {1})
            no_graphis {bool} -- Whether to allow the executable to render or not (default: {False})
            realtime_mode {bool} -- Whether to run the environment in real time or as fast as possible (default: {False})
            record_trajectory {bool} -- Whether to record the trajectory of an entire episode. This can be used for video recording. (default: {False})
        """
        # Initialize channels
        self.reset_parameters = EnvironmentParametersChannel()
        self.engine_config = EngineConfigurationChannel()

        # Prepare default reset parameters
        self._default_reset_parameters = {}
        for key, value in reset_params.items():
            self._default_reset_parameters[key] = value
            if key != "start-seed" or key != "num-seeds":
                self.reset_parameters.set_float_parameter(key, value)

        self._realtime_mode = realtime_mode
        if realtime_mode:
            self.engine_config.set_configuration_parameters(time_scale=1.0, width=1280, height=720)
        else:
            self.engine_config.set_configuration_parameters(time_scale=30.0, width=256, height=256)

        # Whether to record the trajectory of an entire episode
        self._record = record_trajectory

        # Launch the environment's executable
        self._env = UnityEnvironment(file_name = env_path, worker_id = worker_id, no_graphics = no_graphis, side_channels=[self.reset_parameters, self.engine_config], timeout_wait=300)
        # If the Unity Editor chould be used instead of a build
        # self._env = UnityEnvironment(file_name = None, worker_id = 0, no_graphics = no_graphis, side_channels=[self.reset_parameters, self.engine_config])

        # Reset the environment
        self._env.reset()
        # Retrieve behavior configuration
        self._behavior_name = list(self._env.behavior_specs)[0]
        self._behavior_spec = self._env.behavior_specs[self._behavior_name]

        # Check whether this Unity environment is supported
        self._verify_environment()

        # Set action space properties
        if self._behavior_spec.action_spec.is_discrete():
            num_action_branches = self._behavior_spec.action_spec.discrete_size
            action_branch_dimensions = self._behavior_spec.action_spec.discrete_branches
            if num_action_branches == 1:
                self._action_space = spaces.Discrete(action_branch_dimensions[0])
            else:
                self._action_space = spaces.MultiDiscrete(action_branch_dimensions)

        # Count visual and vector observations
        self._num_vis_obs, self._num_vec_obs = 0, 0
        self._vec_obs_indices = []
        for index, obs in enumerate(self._behavior_spec.observation_specs):
            if len(obs[0]) > 1:
                self._num_vis_obs = self._num_vis_obs + 1
                self._vis_obs_index = index
            else:
                self._num_vec_obs = self._num_vec_obs + 1
                self._vec_obs_indices.append(index)

        # Set visual observation space property
        if self._num_vis_obs == 1:
            vis_obs_shape = self._behavior_spec.observation_specs[self._vis_obs_index].shape

            self._visual_observation_space = spaces.Box(
                low = 0,
                high = 1.0,
                shape = vis_obs_shape,
                dtype = np.float32)
        else:
            self._visual_observation_space = None

        # Set vector observation space property
        if self._num_vec_obs > 0:
            # Determine the length of vec obs by summing the length of each distinct one
            vec_obs_length = sum([self._behavior_spec.observation_specs[i][0][0] for i in self._vec_obs_indices])
            self._vector_observatoin_space = (vec_obs_length, )
        else:
            self._vector_observatoin_space = None

        # Videos can only be recorded if the environment provides visual observations
        if self._record and self._visual_observation_space is None:
            UnityEnvironmentException("Videos cannot be rendered for a Unity environment that does not provide visual observations.")

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
    def seed(self):
        """Returns the seed of the current episode."""
        return self._seed

    @property
    def action_names(self):
        return None

    @property
    def max_episode_steps(self):
        """Returns the maximum number of steps that an episode can last."""
        return 2048

    @property
    def get_episode_trajectory(self):
        """Returns the trajectory of an entire episode as dictionary (vis_obs, vec_obs, rewards, actions). 
        """
        self._trajectory["action_names"] = self.action_names
        return self._trajectory if self._trajectory else None

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

        # Use initial or new reset parameters
        if reset_params is None:
            reset_params = self._default_reset_parameters
        else:
            reset_params = reset_params

        # Apply reset parameters
        for key, value in reset_params.items():
            # Skip reset parameters that are not used by the Unity environment
            if key != "start-seed" or key != "num-seeds":
                self.reset_parameters.set_float_parameter(key, value)

        # Sample the to be used seed
        if reset_params["start-seed"] > -1:
            self._seed = randint(reset_params["start-seed"], reset_params["start-seed"] + reset_params["num-seeds"] - 1)
        else:
            # Use unlimited seeds
            self._seed = -1
        self.reset_parameters.set_float_parameter("seed", self._seed)

        # Reset and verify the environment
        self._env.reset()
        info, terminal_info = self._env.get_steps(self._behavior_name)
        self._verify_environment()
        
        # Retrieve initial observations
        vis_obs, vec_obs, _, _ = self._process_agent_info(info, terminal_info)

        # Prepare trajectory recording
        self._trajectory = {
            "vis_obs": [vis_obs * 255] if vis_obs is not None else [vis_obs],
            "vec_obs": [vec_obs],
            "rewards": [0.0], "actions": []
        }

        return vis_obs, vec_obs, {}

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
        action_tuple = ActionTuple()
        action_tuple.add_discrete(np.asarray(action).reshape([1, -1]))
        self._env.set_actions(self._behavior_name, action_tuple)
        self._env.step()
        info, terminal_info = self._env.get_steps(self._behavior_name)

        # Process step results
        vis_obs, vec_obs, reward, done = self._process_agent_info(info, terminal_info)
        self._rewards.append(reward)

        # Record trajectory data
        if self._record:
            if vis_obs is not None:
                self._trajectory["vis_obs"].append(vis_obs * 255)
            else:
                self._trajectory["vis_obs"].append(vis_obs)
            self._trajectory["vec_obs"].append(vec_obs)
            self._trajectory["rewards"].append(reward)
            self._trajectory["actions"].append(action)

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

    def _verify_environment(self):
        # Verify number of agent behavior types
        if len(self._env.behavior_specs) != 1:
            raise UnityEnvironmentException("The unity environment containts more than one agent type.")
        # Verify number of agents
        decision_steps, _ = self._env.get_steps(self._behavior_name)
        if len(decision_steps) > 1:
            raise UnityEnvironmentException("The unity environment contains more than one agent, which is not supported.")
        # Verify action space type
        if not self._behavior_spec.action_spec.is_discrete() or self._behavior_spec.action_spec.is_continuous():
            raise UnityEnvironmentException("Continuous action spaces are not supported. " 
                                            "Only discrete and MultiDiscrete spaces are supported.")
        # Verify that at least one observation is provided
        num_vis_obs = 0
        num_vec_obs = 0
        for obs_spec in self._behavior_spec.observation_specs:
            if len(obs_spec.shape) == 3:
                num_vis_obs += 1
            elif(len(obs_spec.shape)) == 1:
                num_vec_obs += 1
        if num_vis_obs == 0 and num_vec_obs == 0:
            raise UnityEnvironmentException("The unity environment does not contain any observations.")
        # Verify number of visual observations
        if num_vis_obs > 1:
            raise UnityEnvironmentException("The unity environment contains more than one visual observation.")
        
class UnityEnvironmentException(error.Error):
    """Any error related to running the Unity environment."""
    pass