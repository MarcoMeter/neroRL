import numpy as np
import gymnasium as gym
from gymnasium import spaces
import time
from random import randint
from neroRL.environments.env import Env

class PendulumWrapper(Env):
    """This class wraps Gym CartPole environments.
    https://gym.openai.com/docs/#environments
    Available Environments:
        Pendulum-v1
    """

    def __init__(self, env_name, reset_params = None, realtime_mode = False, record_trajectory = False):
        """Instantiates the Pendulum environment.

        Arguments:
            env_name {string} -- Name of the Pendulum environment

        Keyword Arguments:
            reset_params {dict} -- Provides parameters for configuring the environment (default: {None})
        """
        # Set default reset parameters if none were provided
        if reset_params is None:
            self._default_reset_params = {"start-seed": 0, "num-seeds": 100}
        else:
            self._default_reset_params = reset_params

        self._realtime_mode = realtime_mode
        self._record = record_trajectory
        render_mode = None
        if realtime_mode:
            render_mode = "human"
        if record_trajectory:
            render_mode = "rgb_array"

        # Initialize environment
        self._env_name = env_name
        self._env = gym.make(self._env_name, render_mode = render_mode)
        self._observation_space = spaces.Dict({"vec_obs": self._env.observation_space})

    @property
    def unwrapped(self):
        """Return this environment in its vanilla (i.e. unwrapped) state."""
        return self

    @property
    def observation_space(self):
        """Returns the observation space of the environment."""
        return self._observation_space

    @property
    def action_space(self):
        """Returns the action space of the agent."""
        return self._env.action_space

    @property
    def max_episode_steps(self):
        """Returns the maximum number of steps that an episode can last."""
        return self._env._max_episode_steps

    @property
    def seed(self):
        """Returns the seed of the current episode."""
        return self._seed

    @property
    def action_names(self):
        """Returns a list of action names."""
        return ["force"]

    @property
    def get_episode_trajectory(self):
        """Returns the trajectory of an entire episode as dictionary (vis_obs, vec_obs, rewards, actions)."""
        self._trajectory["action_names"] = self.action_names
        return self._trajectory if self._trajectory else None

    def reset(self, reset_params = None):
        """Resets the environment.
        
        Keyword Arguments:
            reset_params {dict} -- Provides parameters, like if the observed velocity should be masked. (default: {None})
        
        Returns:
            {dict} -- Observation of the environment
            {dict} -- Empty info
        """
        # Set default reset parameters if none were provided
        if reset_params is None:
            reset_params = self._default_reset_params

        # Sample seed
        self._seed = randint(reset_params["start-seed"], reset_params["start-seed"] + reset_params["num-seeds"] - 1)

        # Track rewards of an entire episode
        self._rewards = []

        # Retrieve the agent's initial observation
        vec_obs, _ = self._env.reset(seed=self._seed)
        obs = {"vec_obs": vec_obs.squeeze()}

        # Render environment?
        if self._realtime_mode:
            self._env.render()

        # Prepare trajectory recording
        if self._record:
            self._trajectory = {
                "vis_obs": [self._env.render()], "vec_obs": [vec_obs],
                "rewards": [0.0], "actions": [], "frame_rate": 20
            }

        return obs, {}

    def step(self, action):
        """Runs one timestep of the environment's dynamics.
        
        Arguments:
            action {int} -- The to be executed action
        
        Returns:
            {dict} -- Observation of the environment
            {float} -- Scalar reward signaled by the environment
            {bool} -- Whether the episode of the environment terminated
            {dict} -- Further information (e.g. episode length) retrieved from the environment once an episode completed
        """
        # Execute action
        if not isinstance(action, (list, np.ndarray)):
            action = [action]
        vec_obs, reward, done, truncation, info = self._env.step(action)
        obs = {"vec_obs": vec_obs.squeeze()}
        self._rewards.append(reward)

        # Render environment?
        if self._realtime_mode:
            self._env.render()
            time.sleep(0.033)

        # Record trajectory data
        if self._record:
            self._trajectory["vis_obs"].append(self._env.render())
            self._trajectory["vec_obs"].append(vec_obs)
            self._trajectory["rewards"].append(reward)
            self._trajectory["actions"].append(action)

        # Wrap up episode information once completed (i.e. done)
        if done or truncation:
            info = {"reward": sum(self._rewards),
                    "length": len(self._rewards)}
        else:
            info = None

        return obs, reward, done or truncation, info

    def close(self):
        """Shuts down the environment."""
        self._env.close()
