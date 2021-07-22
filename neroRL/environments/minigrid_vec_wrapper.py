import time
import numpy as np
import gym_minigrid
import gym
from gym import error, spaces
from random import randint
from neroRL.environments.env import Env
from gym_minigrid.wrappers import ViewSizeWrapper

class MinigridVecWrapper(Env):
    """This class wraps Gym Minigrid environments.
    https://github.com/maximecb/gym-minigrid
    """

    def __init__(self, env_name, reset_params = None, realtime_mode = False, record_trajectory = False):
        """Instantiates the Minigrid environment.
        
        Arguments:
            env_name {string} -- Name of the Minigrid environment
            reset_params {dict} -- Provides parameters, like a seed, to configure the environment. (default: {None})
            realtime_mode {bool} -- Whether to render the environment in realtime. (default: {False})
            record_trajectory {bool} -- Whether to record the trajectory of an entire episode. This can be used for video recording. (default: {False})
        """
        # Set default reset parameters if none were provided
        if reset_params is None:
            self._default_reset_params = {"start-seed": 0, "num-seeds": 100}
        else:
            self._default_reset_params = reset_params

        self._env = gym.make(env_name)
        self.agent_view_size = 3
        self._env = ViewSizeWrapper(self._env, self.agent_view_size)

        self._realtime_mode = realtime_mode
        self._record = record_trajectory

        # Prepare observation space
        self._vector_observation_space = (self.agent_view_size**2*5,)

    @property
    def unwrapped(self):
        """Return this environment in its vanilla (i.e. unwrapped) state."""
        return self

    @property
    def visual_observation_space(self):
        """Returns the shape of the visual component of the observation space as a tuple."""
        return None

    @property
    def vector_observation_space(self):
        """Returns the shape of the vector component of the observation space as a tuple."""
        return self._vector_observation_space

    @property
    def action_space(self):
        """Returns the shape of the action space of the agent."""
        return spaces.Discrete(3)
        # return self._env.action_space

    @property
    def action_names(self):
        """Returns a list of action names."""
        return [["left", "right", "forward"]]
        # return [["left", "right", "forward", "toggle", "pickup", "drop", "done"]]

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
        else:
            reset_params = reset_params
        # Set seed
        self._env.seed(randint(reset_params["start-seed"], reset_params["start-seed"] + reset_params["num-seeds"] - 1))
        # Track rewards of an entire episode
        self._rewards = []
        # Reset the environment and retrieve the initial observation
        obs = self._env.reset()

        # Vector observation
        vec_obs = self.process_obs(obs["image"])

        # Render environment?
        if self._realtime_mode:
            self._env.render(tile_size = 96)

        # Prepare trajectory recording
        self._trajectory = {
            "vis_obs": [self._env.render(tile_size = 96, mode = "rgb_array").astype(np.uint8)], "vec_obs": [None],
            "rewards": [0.0], "actions": [], "frame_rate": 1
        }
        return None, vec_obs

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
        obs, reward, done, info = self._env.step(action[0])
        self._rewards.append(reward)
        # Retrieve the RGB frame of the agent's vision
        # vis_obs = self._env.get_obs_render(obs["image"], tile_size=12)  / 255.
        # Vector observation
        vec_obs = self.process_obs(obs["image"])

        # Render the environment in realtime
        if self._realtime_mode:
            self._env.render(tile_size = 96)
            time.sleep(0.5)

        # Record trajectory data
        if self._record:
            self._trajectory["vis_obs"].append(self._env.render(tile_size = 96, mode="rgb_array").astype(np.uint8))
            self._trajectory["vec_obs"].append(None)
            self._trajectory["rewards"].append(reward)
            self._trajectory["actions"].append(action)
        
        # Wrap up episode information once completed (i.e. done)
        if done:
            success = 1.0 if sum(self._rewards) > 0 else 0.0
            info = {"reward": sum(self._rewards),
                    "length": len(self._rewards),
                    "success": success}
        else:
            info = None

        return None, vec_obs, reward, done, info

    def close(self):
        """Shuts down the environment."""
        self._env.close()

    def process_obs(self, obs):
        one_hot_obs = []
        for i in range(obs.shape[0]):
            for j in range(obs.shape[1]):
                # print(obs[i,j,0])
                # if i == 1 and j == 2:
                #     one_hot_obs.append([1, 0, 0, 0, 0]) # agent tile
                # else:
                if obs[i,j,0] == 1:
                    one_hot_obs.append([0, 1, 0, 0, 0]) # walkable tile
                elif obs[i,j,0] == 2:
                    one_hot_obs.append([0, 0, 1, 0, 0]) # blocked tile
                elif obs[i,j,0] == 5:
                    one_hot_obs.append([0, 0, 0, 1, 0]) # key tile
                elif obs[i,j,0] == 6:
                    one_hot_obs.append([0, 0, 0, 0, 1]) # circle tile
                else:
                    one_hot_obs.append([0, 0, 0, 0, 0]) # anything else
        # return flattened one-hot encoded observation
        return np.asarray(one_hot_obs, dtype=np.float32).reshape(-1)