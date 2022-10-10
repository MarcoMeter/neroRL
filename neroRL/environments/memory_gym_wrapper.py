import gym
import memory_gym

from random import randint
from gym import spaces

from neroRL.environments.env import Env

class MemoryGymWrapper(Env):
    """
    This class wraps memory-gym environments.
    https://github.com/MarcoMeter/drl-memory-gym
    Available Environments:
        SearingSpotlights-v0
        soon MortarMayhem-v0
        soon MysteryMaze-v0
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

        self._env = gym.make(env_name, disable_env_checker = True, headless = not realtime_mode)

        self._realtime_mode = realtime_mode
        self._record = record_trajectory

        if type(self._env.observation_space) is spaces.Dict:
            self._visual_observation_space = self._env.observation_space["visual_observation"]
            self._vector_observation_space = self._env.observation_space["vector_observation"].shape
        else:
            self._visual_observation_space = self._env.observation_space
            self._vector_observation_space = None

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
        return self._vector_observation_space

    @property
    def action_space(self):
        """Returns the shape of the action space of the agent."""
        return self._env.action_space

    @property
    def action_names(self):
        """Returns a list of action names. It has to be noted that only the names of action branches are provided and not the actions themselves!"""
        if isinstance(self.action_space, spaces.MultiDiscrete):
            return [["no-op", "left", "right"], ["no-op", "up", "down"]]
        else:
            return ["no-op", "rotate left", "rotate right", "move forward"]

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
            {numpy.ndarray} -- Visual observation
            {numpy.ndarray} -- Vector observation
        """
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

        # Reset the environment to retrieve the initial observation
        obs = self._env.reset(seed=seed, options=options)
        if type(self._env.observation_space) is spaces.Dict:
            vis_obs = obs["visual_observation"]
            vec_obs = obs["vector_observation"]
        else:
            vis_obs = obs
            vec_obs = None


        if self._realtime_mode:
            self._env.render(mode="debug_rgb_array")

        # Prepare trajectory recording
        self._trajectory = {
            "vis_obs": [self._env.render(mode="debug_rgb_array")], "vec_obs": [vec_obs],
            "rewards": [0.0], "actions": []
        } if self._record else None

        return vis_obs, vec_obs

    def step(self, action):
        """Runs one timestep of the environment's dynamics.
        
        Arguments:
            action {int} -- The to be executed action
        
        Returns:
            {numpy.ndarray} -- Visual observation
            {numpy.ndarray} -- Vector observation
            {float} -- (Total) Scalar reward signaled by the environment
            {bool} -- Whether the episode of the environment terminated
            {dict} -- Further episode information (e.g. cumulated reward) retrieved from the environment once an episode completed
        """
        obs, reward, done, info = self._env.step(action)

        if type(self._env.observation_space) is spaces.Dict:
            vis_obs = obs["visual_observation"]
            vec_obs = obs["vector_observation"]
        else:
            vis_obs = obs
            vec_obs = None

        if self._realtime_mode or self._record:
            img = self._env.render(mode="debug_rgb_array")

        # Record trajectory data
        if self._record:
            self._trajectory["vis_obs"].append(img)
            self._trajectory["vec_obs"].append(vec_obs)
            self._trajectory["rewards"].append(reward)
            self._trajectory["actions"].append(action)

        return vis_obs, vec_obs, reward, done, info

    def close(self):
        """Shuts down the environment."""
        self._env.close()