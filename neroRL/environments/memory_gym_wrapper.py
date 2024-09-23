import gymnasium as gym
import os
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"
import memory_gym

from random import randint
from gymnasium import spaces

from neroRL.environments.env import Env

class MemoryGymWrapper(Env):
    """
    This class wraps memory-gym environments.
    https://github.com/MarcoMeter/drl-memory-gym
    Available Environments:
        SearingSpotlights-v0
        MortarMayhem-v0
        MortarMayhem-Grid-v0
        MortarMayhemB-v0
        MortarMayhemB-Grid-v0
        MysteryPath-v0
        MysteryPath-Grid-v0
    """
    
    def __init__(self, env_name, reset_params = None, realtime_mode = False, record_trajectory = False) -> None:
        """Instantiates the memory-gym environment.
        
        Arguments:
            env_name {string} -- Name of the memory-gym environment
            reset_params {dict} -- Provides parameters, like a seed, to configure the environment. (default: {None})
            realtime_mode {bool} -- Whether to render the environment in realtime. (default: {False})
            record_trajectory {bool} -- Whether to record the trajectory of an entire episode. This can be used for video recording. (default: {False})
        """
        if reset_params is None:
            self._default_reset_params = {"start-seed": 0, "num-seeds": 100}
        else:
            self._default_reset_params = reset_params

        render_mode = "debug_rgb_array" if not realtime_mode else "human"
        self._env_name = env_name
        self._env = gym.make(env_name, disable_env_checker = True, render_mode = render_mode)

        # Update the visual observation space to low=0.0 and high=1.0
        env_obs_space = self._env.observation_space
        if isinstance(env_obs_space, spaces.Dict):
            new_spaces = {}
            for key, space in env_obs_space.spaces.items():
                if key == "visual_observation":
                    new_spaces[key] = spaces.Box(low=0.0, high=1.0, shape=space.shape, dtype=space.dtype)
                else:
                    new_spaces[key] = space
            self._observation_space = spaces.Dict(new_spaces)
        else:
            new_box = spaces.Box(low=0.0, high=1.0, shape=env_obs_space.shape, dtype=env_obs_space.dtype)
            self._observation_space = spaces.Dict({"visual_observation": new_box})

        self._realtime_mode = realtime_mode
        self._record = record_trajectory

        # Ensure the environment is reset once to calculate max_episode_steps correctly
        _, _ = self.reset(reset_params)

    @property
    def unwrapped(self):
        """Return this environment in its vanilla (i.e. unwrapped) state."""
        return self

    @property
    def observation_space(self):
        """Returns the observation space of the environment."""
        return self._observation_space

    @property
    def ground_truth_space(self):
        """Returns the space of the ground truth info space if available."""
        if self._env.has_ground_truth_info:
            return self._env.ground_truth_space
        else:
            return None

    @property
    def action_space(self):
        """Returns the action space of the agent."""
        return self._env.action_space

    @property
    def max_episode_steps(self):
        """Returns the maximum number of steps that an episode can last."""
        return self._env.max_episode_steps

    @property
    def seed(self):
        """Returns the seed of the current episode."""
        return self._seed

    @property
    def action_names(self):
        """Returns a list of action names. It has to be noted that only the names of action branches are provided and not the actions themselves!"""
        if self._env_name == "Endless-MysteryPath-v0":
            return [["no-op", "move forward", "rotate left", "rotate right"]]
        elif isinstance(self.action_space, spaces.MultiDiscrete):
            return [["no-op", "left", "right"], ["no-op", "up", "down"]]
        else:
            return [["no-op", "rotate left", "rotate right", "move forward"]]

    @property
    def get_episode_trajectory(self):
        """Returns the trajectory of an entire episode as dictionary (vis_obs, vec_obs, rewards, actions)."""
        self._trajectory["action_names"] = self.action_names
        return self._trajectory if self._trajectory else None

    def reset(self, reset_params = None):
        """Resets the environment.
        
        Keyword Arguments:
            reset_params {dict} -- Provides parameters, like a seed, to configure the environment. (default: {None})
        
        Returns:
            {dict} -- Observation of the environment
            {dict} -- Empty info
        """
        # Process reset parameters
        if reset_params is None:
            reset_params = self._default_reset_params
        else:
            reset_params = reset_params

        # Sample seed
        self._seed = randint(reset_params["start-seed"], reset_params["start-seed"] + reset_params["num-seeds"] - 1)

        # Remove reset params that are not processed directly by the environment
        options = reset_params.copy()
        options.pop("start-seed", None)
        options.pop("num-seeds", None)
        options.pop("seed", None)

        # Reset the environment to retrieve the initial observation
        obs, info = self._env.reset(seed=self._seed, options=options)
        if isinstance(obs, dict):
            obs["visual_observation"] = obs["visual_observation"] / 255.0
            vec_obs = obs["vector_observation"]
        else:
            obs = {"visual_observation": obs / 255.0}
            vec_obs = None
    
        if self._realtime_mode:
            self._env.render()

        # Prepare trajectory recording
        self._trajectory = {
            "vis_obs": [self._env.render()], "vec_obs": [vec_obs],
            "rewards": [0.0], "actions": []
        } if self._record else None

        return obs, info

    def step(self, action):
        """Runs one timestep of the environment's dynamics.
        
        Arguments:
            action {int} -- The to be executed action
        
        Returns:
            {dict} -- Observation of the environment
            {float} -- Reward signaled by the environment
            {bool} -- Whether the episode of the environment terminated
            {dict} -- Further episode information (e.g. cumulated reward) retrieved from the environment once an episode completed
        """
        # Flatten action in single discrete action space
        if isinstance(action, list):
            if len(action) == 1:
                action = action[0]
        
        # Step the environment
        obs, reward, done, truncation, info = self._env.step(action)
        if isinstance(obs, dict):
            obs["visual_observation"] = obs["visual_observation"] / 255.0
            vec_obs = obs["vector_observation"]
        else:
            obs = {"visual_observation": obs / 255.0}
            vec_obs = None

        if self._realtime_mode or self._record:
            img = self._env.render()

        # Record trajectory data
        if self._record:
            self._trajectory["vis_obs"].append(img)
            self._trajectory["vec_obs"].append(vec_obs)
            self._trajectory["rewards"].append(reward)
            self._trajectory["actions"].append(action)

        return obs, reward, done, info

    def close(self):
        """Shuts down the environment."""
        self._env.close()
