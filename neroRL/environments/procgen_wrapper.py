import numpy as np
import procgen
import gym
import time
from gym import error, spaces
from neroRL.environments.env import Env

class ProcgenWrapper(Env):
    """This class wraps Gym Procgen environments.
    https://github.com/openai/procgen
    Available Environments:
        procgen:procgen-bigfish-v0
        procgen:procgen-bossfight-v0
        procgen:procgen-caveflyer-v0
        procgen:procgen-chaser-v0
        procgen:procgen-climber-v0
        procgen:procgen-coinrun-v0
        procgen:procgen-dodgeball-v0
        procgen:procgen-fruitbot-v0
        procgen:procgen-heist-v0
        procgen:procgen-jumper-v0
        procgen:procgen-leaper-v0
        procgen:procgen-maze-v0
        procgen:procgen-miner-v0
        procgen:procgen-ninja-v0
        procgen:procgen-plunder-v0
        procgen:procgen-starpilot-v0
    """

    def __init__(self, env_name, reset_params = None,  realtime_mode = False):
        """Instantiates the Procgen environment.

        Arguments:
            env_name {string} -- Name of the Procgen environment

        Keyword Arguments:
            reset_params {dict} -- Provides parameters, like a seed, to configure the environment. (default: {None})
            realtime_mode {bool} -- Whether the environment should run in realtime or as fast as possible (default: {False})
        """
        # Set default reset parameters if none were provided
        if reset_params is None:
            self._default_reset_params = {"start-seed": 0, "num-seeds": 100}
        else:
            self._default_reset_params = reset_params
        self._realtime_mode = realtime_mode

        # Initialize environment
        self._env_name = env_name
        self._env = gym.make(self._env_name, start_level = self._default_reset_params["start-seed"], num_levels = self._default_reset_params["num-seeds"])

        # Prepare observation space
        self._visual_observation_space = self._env.observation_space

    @property
    def unwrapped(self):
        """Return this environment in its vanilla (i.e. unwrapped) state."""
        return self

    @property
    def visual_observation_space(self):
        """Returns the shape of the visual component of the observation space as a tuple."""
        return self._visual_observation_space

    @property
    def vector_observation_space(self):
        """Returns the shape of the vector component of the observation space as a tuple."""
        return None

    @property
    def action_space(self):
        """Returns the shape of the action space of the agent."""
        return self._env.action_space

    @property
    def action_names(self):
        """Returns a list of action names."""
        return ["left down", "left", "left up", "down", "No-op", "up", "right down", "right", "right up", "D", "A", "W", "S", "Q", "E"]

    def reset(self, reset_params = None):
        """Resets the environment.
        
        Keyword Arguments:
            reset_params {dict} -- Provides parameters, like a seed, to configure the environment. (default: {None})
        
        Returns:
            {numpy.ndarray} -- Visual observation
            {numpy.ndarray} -- Vector observation
        """
        # Set default reset parameters if none were provided
        if reset_params is None:
            reset_params = self._default_reset_params

        # If new reset parameters were specified, Procgen has to be restarted
        if not self._default_reset_params == reset_params:
            self._env.close()
            self._env = gym.make(self._env_name, start_level = reset_params["start-seed"], num_levels = reset_params["num-seeds"])
        # Track rewards of an entire episode
        self._rewards = []
        # Reset the environment and retrieve the initial observation
        obs = self._env.reset()
        # Retrieve the RGB frame of the agent"s vision
        vis_obs = obs.astype(np.float32) / 255.

        return vis_obs, None

    def step(self, action):
        """Runs one timestep of the environment's dynamics.
        
        Arguments:
            action {int} -- The to be executed action
        
        Returns:
            {numpy.ndarray} -- Visual observation
            {numpy.ndarray} -- Vector observation
            {float} -- (Total) Scalar reward signaled by the environment
            {bool} -- Whether the episode of the environment terminated
            {dict} -- Further episode information (e.g. episode length) retrieved from the environment once an episode completed
        """
        # Render environment?
        if self._realtime_mode:
            self._env.render()
            time.sleep(0.033)

        # Execute action
        obs, reward, done, info = self._env.step(action)
        self._rewards.append(reward)
        # Retrieve the RGB frame of the agent's vision
        vis_obs = obs.astype(np.float32)  / 255.

        # Wrap up episode information once completed (i.e. done)
        if done:
            info = {"reward": sum(self._rewards),
                    "length": len(self._rewards)}
        else:
            info = None

        return vis_obs, None, reward, done, info

    def close(self):
        """Shuts down the environment."""
        self._env.close()
