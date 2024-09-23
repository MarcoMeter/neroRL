import math
from neroRL.environments.env import Env

class FrameSkipEnv(Env):
    """This wrapper returns only every "skip"-th frame"""

    def __init__(self, env, skip):
        """        
        Arguments:
            env {Env} -- The to be wrapped environment, which is derived from the Env class
            skip {int} -- Number of frames to be skipped
        """
        self._env = env
        self._skip = skip
        self._max_episode_steps = None  # Initialize to None to calculate later

        # Reset the environment once to initialize and get the max episode steps
        _, _ = self._env.reset()

        # Set max_episode_steps based on the reset environment state
        if hasattr(self.unwrapped, "max_episode_steps"):
            self._max_episode_steps = math.ceil(self.unwrapped.max_episode_steps / self._skip)
        else:
            raise AttributeError("The environment does not have 'max_episode_steps' set after reset.")

    @property
    def unwrapped(self):
        """Return this environment in its vanilla (i.e. unwrapped) state."""
        return self._env.unwrapped

    @property
    def observation_space(self):
        """Returns the observation space of the environment."""
        return self._env.observation_space

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
        return self._max_episode_steps

    @property
    def seed(self):
        """Returns the seed of the current episode."""
        return self._env.seed

    @property
    def action_names(self):
        """Returns a list of action names. It has to be noted that only the names of action branches are provided and not the actions themselves!"""
        return self._env.action_names

    @property
    def get_episode_trajectory(self):
        """Returns the trajectory of an entire episode as dictionary (vis_obs, vec_obs, rewards, actions)."""
        return self._env.get_episode_trajectory

    def reset(self, reset_params = None):
        """Reset the environment. The provided reset_params is a dictionary featuring reset parameters of the environment such as the seed."""
        obs, info = self._env.reset(reset_params = reset_params)
        self._max_episode_steps = math.ceil(self.unwrapped.max_episode_steps / self._skip)
        return obs, info

    def step(self, action):
        """Executes steps of the agent in the environment untill the "skip"-th frame is reached.
        
        Arguments:
            action {List} -- A list of at least one discrete action to be executed by the agent
        
        Returns:
                {dict} -- Observation
                {float} -- Reward signaled by the environment
                {bool} -- Whether the episode of the environment terminated
                {dict} -- Further episode information retrieved from the environment
        """
        total_reward = 0.0

        # Repeat the same action for the to be skipped frames
        for _ in range(self._skip):
            obs, reward, done, info = self._env.step(action)
            total_reward += reward
            if done:
                info["length"] = math.ceil(info["length"] / self._skip)
                break

        return obs, total_reward, done, info

    def close(self):
        """Shuts down the environment."""
        self._env.close()