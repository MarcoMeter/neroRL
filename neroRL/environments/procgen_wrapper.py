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

    def __init__(self, env_name, reset_params = None,  realtime_mode = False, record_trajectory = False):
        """Instantiates the Procgen environment.

        Arguments:
            env_name {string} -- Name of the Procgen environment

        Keyword Arguments:
            reset_params {dict} -- Provides parameters, like a seed, to configure the environment. (default: {None})
            realtime_mode {bool} -- Whether the environment should run in realtime or as fast as possible (default: {False})
            record_trajectory {bool} -- Whether to record the trajectory of an entire episode. This can be used for video recording. (default: {False})
        """
        # Set default reset parameters if none were provided
        self._default_reset_params = {"start-seed": 0, "num-seeds": 100, "paint_vel_info": False,
                                        "use_generated_assets": False, "center_agent": False, "use_sequential_levels": False,
                                        "distribution_mode": "hard", "use_backgrounds": True, "restrict_themes": False,
                                        "use_monochrome_assets": False}

        # Set default reset parameters if none were provided
        if reset_params is None:
            reset_params = self._default_reset_params
        else:
            reset_params = reset_params

        self._realtime_mode = realtime_mode
        self._record = record_trajectory

        # Initialize environment
        self._env_name = env_name
        self._env = gym.make(self._env_name,
                            render_mode = "human" if self._realtime_mode else None,
                            start_level = reset_params["start-seed"],
                            num_levels = reset_params["num-seeds"],
                            paint_vel_info = reset_params["paint_vel_info"],
                            use_generated_assets = reset_params["use_generated_assets"],
                            center_agent = reset_params["center_agent"],
                            use_sequential_levels = reset_params["use_sequential_levels"],
                            distribution_mode = reset_params["distribution_mode"],
                            use_backgrounds = reset_params["use_backgrounds"],
                            restrict_themes = reset_params["restrict_themes"],
                            use_monochrome_assets = reset_params["use_monochrome_assets"])

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
        return [["left down", "left", "left up", "down", "No-op", "up", "right down", "right", "right up", "D", "A", "W", "S", "Q", "E"]]

    @property
    def get_episode_trajectory(self):
        """Returns the trajectory of an entire episode as dictionary (vis_obs, vec_obs, rewards, actions). 
        """
        self._trajectory["action_names"] = self.action_names
        return self._trajectory if self._trajectory else None

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
            self._env = gym.make(self._env_name,
                            render_mode = "human" if self._realtime_mode else None,
                            start_level = reset_params["start-seed"],
                            num_levels = reset_params["num-seeds"],
                            paint_vel_info = reset_params["paint_vel_info"],
                            use_generated_assets = reset_params["use_generated_assets"],
                            center_agent = reset_params["center_agent"],
                            use_sequential_levels = reset_params["use_sequential_levels"],
                            distribution_mode = reset_params["distribution_mode"],
                            use_backgrounds = reset_params["use_backgrounds"],
                            restrict_themes = reset_params["restrict_themes"],
                            use_monochrome_assets = reset_params["use_monochrome_assets"])
        # Track rewards of an entire episode
        self._rewards = []
        # Reset the environment and retrieve the initial observation
        obs = self._env.reset()
        # Retrieve the RGB frame of the agent"s vision
        vis_obs = obs.astype(np.float32) / 255.

        # Prepare trajectory recording
        self._trajectory = {
            "vis_obs": [self._env.render(mode = "rgb_array")], "vec_obs": [None],
            "rewards": [0.0], "actions": []
        }

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
        # Execute action
        obs, reward, done, info = self._env.step(action[0])
        self._rewards.append(reward)
        # Retrieve the RGB frame of the agent's vision
        vis_obs = obs.astype(np.float32)  / 255.

        # Record trajectory data
        if self._record:
            self._trajectory["vis_obs"].append(self._env.render(mode = "rgb_array"))
            self._trajectory["vec_obs"].append(None)
            self._trajectory["rewards"].append(reward)
            self._trajectory["actions"].append(action)

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
