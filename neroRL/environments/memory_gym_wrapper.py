from cgitb import reset
import gym
import memory_gym
import numpy as np

from random import randint

from neroRL.environments.env import Env

class MemoryGymWrapper(Env):
    """
    This abstract class helps to implement new environments that fit this implementation.
    It is to some extend inspired by OpenAI gym.
    The key difference is that an environment can have a visual and a vector observation space at the same time.
    The provided arguments of the to be implemented functions are not fixed, but are recommended at minimum.
    """
    def __init__(self, env_name, reset_params = None, realtime_mode = False, record_trajectory = False) -> None:
        if reset_params is None:
            self._default_reset_params = {"start-seed": 0, "num-seeds": 100}
        else:
            self._default_reset_params = reset_params

        self._env = gym.make(env_name, headless = not realtime_mode)

        self._realtime_mode = realtime_mode
        self._record = record_trajectory

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
        """Returns a list of action names. It has to be noted that only the names of action branches are provided and not the actions themselves!"""
        return [["no-op", "left", "right"], ["no-op", "up", "down"]] # TODO check

    @property
    def get_episode_trajectory(self):
        """Returns a dictionary containing a complete trajectory of data comprising an entire episode. This is only used for recording a video."""
        self._trajectory["action_names"] = self.action_names
        return self._trajectory if self._trajectory else None

    def reset(self, reset_params = None):
        """Reset the environment. The provided config is a dictionary featuring reset parameters for the environment (e.g. seed)."""
        # Process reset parameters
        if reset_params is None:
            reset_params = self._default_reset_params
        else:
            reset_params = reset_params

        # Sample seed
        seed = randint(reset_params["start-seed"], reset_params["start-seed"] + reset_params["num-seeds"] - 1)

        # Remove reset params that are not processed directly by the environment
        options = reset_params.copy()
        options.pop("start-seed", None)
        options.pop("num-seeds", None)
        options.pop("seed", None)

        # Track rewards of an entire episode
        self._rewards = []

        # Reset the environment to retrieve the visual observation
        vis_obs = self._env.reset(seed=seed, options=options)

        # Render environment?
        if self._realtime_mode:
            self._env.render()

        # Prepare trajectory recording
        self._trajectory = {
            "vis_obs": [self._env.render()], "vec_obs": [None],
            "rewards": [0.0], "actions": []
        } if self._record else None

        return vis_obs, None

    def step(self, action):
        """Executes one step of the agent."""
        vis_obs, reward, done, info = self._env.step(action)
        self._rewards.append(reward)
        # Retrieve the RGB frame of the agent's vision

        # Render the environment in realtime
        if self._realtime_mode:
            self._env.render()

        # Record trajectory data
        if self._record:
            self._trajectory["vis_obs"].append(self._env.render())
            self._trajectory["vec_obs"].append(None)
            self._trajectory["rewards"].append(reward)
            self._trajectory["actions"].append(action)

        # Wrap up episode information once completed (i.e. done)
        if done:
            info = {"reward": sum(self._rewards),
                    "length": len(self._rewards),}
        else:
            info = None

        return vis_obs, None, reward, done, info

    def close(self):
        """Shuts down the environment."""
        self._env.close()