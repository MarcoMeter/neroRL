from collections import deque
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
        # Store skipped frames in the case of video recording (what if the environment does not provide visual obs?)
        self.skipped_frames = deque(maxlen = self._skip)

    @property
    def unwrapped(self):
        """Return this environment in its vanilla (i.e. unwrapped) state."""
        return self._env.unwrapped

    @property
    def visual_observation_space(self):
        """Returns the shape of the visual component of the observation space as a tuple."""
        return self._env.visual_observation_space

    @property
    def vector_observation_space(self):
        """Returns the shape of the vector component of the observation space as a tuple."""
        return self._env.vector_observation_space

    @property
    def action_space(self):
        """Returns the shape of the action space of the agent."""
        return self._env.action_space

    @property
    def action_names(self):
        """Returns a list of action names. It has to be noted that only the names of action branches are provided and not the actions themselves!"""
        return self._env.action_names

    @property
    def get_episode_trajectory(self):
        """Returns the trajectory of an entire episode as dictionary (vis_obs, vec_obs, rewards, actions). 
        """
        return self._env.get_episode_trajectory

    def reset(self, reset_params = None):
        """Reset the environment. The provided reset_params is a dictionary featuring reset parameters of the environment such as the seed."""
        vis_obs, vec_obs = self._env.reset(reset_params = reset_params)
        return vis_obs, vec_obs

    def step(self, action):
        """Executes steps of the agent in the environment untill the "skip"-th frame is reached.
        
        Arguments:
            action {List} -- A list of at least one discrete action to be executed by the agent
        
        Returns:
                {numpy.ndarray} -- Visual observation
                {numpy.ndarray} -- Vector observation
                {float} -- (Total) Scalar reward signaled by the environment
                {bool} -- Whether the episode of the environment terminated
                {dict} -- Further episode information retrieved from the environment
        """
        total_reward = 0.0

        # Repeat the same action for the to be skipped frames
        for _ in range(self._skip):
            vis_obs, vec_obs, reward, done, info = self._env.step(action)
            total_reward += reward
            self.skipped_frames.append(vis_obs)
            if done:
                info["length"] = info["length"] // self._skip
                break

        return vis_obs, vec_obs, total_reward, done, info

    def close(self):
        """Shuts down the environment."""
        self._env.close()