import time
import numpy as np
import minigrid
import gymnasium as gym
from gymnasium import error, spaces
from random import randint
from neroRL.environments.env import Env
from minigrid.wrappers import *

class MinigridWrapper(Env):
    """This class wraps Gym Minigrid environments.
    https://github.com/maximecb/gym-minigrid
    Custom Environments:
        MiniGrid-MortarAB-v0
        MiniGrid-MortarB-v0
    Available Environments:
        Empty
            - MiniGrid-Empty-5x5-v0
            - MiniGrid-Empty-Random-5x5-v0
            - MiniGrid-Empty-6x6-v0
            - MiniGrid-Empty-Random-6x6-v0
            - MiniGrid-Empty-8x8-v0
            - MiniGrid-Empty-16x16-v0
        Four rooms
            - MiniGrid-FourRooms-v0
        Door & key
            - MiniGrid-DoorKey-5x5-v0
            - MiniGrid-DoorKey-6x6-v0
            - MiniGrid-DoorKey-8x8-v0
            - MiniGrid-DoorKey-16x16-v0
        Multi-room
            - MiniGrid-MultiRoom-N2-S4-v0
            - MiniGrid-MultiRoom-N4-S5-v0
            - MiniGrid-MultiRoom-N6-v0
        Fetch
            - MiniGrid-Fetch-5x5-N2-v0
            - MiniGrid-Fetch-6x6-N2-v0
            - MiniGrid-Fetch-8x8-N3-v0
        Go-to-door
            - MiniGrid-GoToDoor-5x5-v0
            - MiniGrid-GoToDoor-6x6-v0
            - MiniGrid-GoToDoor-8x8-v0
        Put near
            - MiniGrid-PutNear-6x6-N2-v0
            - MiniGrid-PutNear-8x8-N2-v0
        Red and blue doors
            - MiniGrid-RedBlueDoors-6x6-v0
            - MiniGrid-RedBlueDoors-8x8-v0
        Memory
            - MiniGrid-MemoryS17Random-v0
            - MiniGrid-MemoryS13Random-v0
            - MiniGrid-MemoryS13-v0
            - MiniGrid-MemoryS11-v0
            - MiniGrid-MemoryS9-v0
            - MiniGrid-MemoryS7-v0
        Locked room
            - MiniGrid-LockedRoom-v0
        Key corridor
            - MiniGrid-KeyCorridorS3R1-v0
            - MiniGrid-KeyCorridorS3R2-v0
            - MiniGrid-KeyCorridorS3R3-v0
            - MiniGrid-KeyCorridorS4R3-v0
            - MiniGrid-KeyCorridorS5R3-v0
            - MiniGrid-KeyCorridorS6R3-v0
        Unlock
            - MiniGrid-Unlock-v0
        Unlock Pickup
            - MiniGrid-UnlockPickup-v0
        Blocked unlock pickup
            - MiniGrid-BlockedUnlockPickup-v0
        Obstructed mazed
            - MiniGrid-ObstructedMaze-1Dl-v0
            - MiniGrid-ObstructedMaze-1Dlh-v0
            - MiniGrid-ObstructedMaze-1Dlhb-v0
            - MiniGrid-ObstructedMaze-2Dl-v0
            - MiniGrid-ObstructedMaze-2Dlh-v0
            - MiniGrid-ObstructedMaze-2Dlhb-v0
            - MiniGrid-ObstructedMaze-1Q-v0
            - MiniGrid-ObstructedMaze-2Q-v0
            - MiniGrid-ObstructedMaze-Full-v0
        Distributional shift
            - MiniGrid-DistShift1-v0
            - MiniGrid-DistShift2-v0
        Lava gap
            - MiniGrid-LavaGapS5-v0
            - MiniGrid-LavaGapS6-v0
            - MiniGrid-LavaGapS7-v0
        Lava crossing
            - MiniGrid-LavaCrossingS9N1-v0
            - MiniGrid-LavaCrossingS9N2-v0
            - MiniGrid-LavaCrossingS9N3-v0
            - MiniGrid-LavaCrossingS11N5-v0
        Simple crossing
            - MiniGrid-SimpleCrossingS9N1-v0
            - MiniGrid-SimpleCrossingS9N2-v0
            - MiniGrid-SimpleCrossingS9N3-v0
            - MiniGrid-SimpleCrossingS11N5-v0
        Dynaimc obstacles
            - MiniGrid-Dynamic-Obstacles-5x5-v0
            - MiniGrid-Dynamic-Obstacles-Random-5x5-v0
            - MiniGrid-Dynamic-Obstacles-6x6-v0
            - MiniGrid-Dynamic-Obstacles-Random-6x6-v0
            - MiniGrid-Dynamic-Obstacles-8x8-v0
            - MiniGrid-Dynamic-Obstacles-16x16-v0
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
            self._default_reset_params = {"start-seed": 0, "num-seeds": 100, "view-size": 3, "max-episode-steps": 96}
        else:
            self._default_reset_params = reset_params

        self.max_episode_steps = self._default_reset_params["max-episode-steps"]
        render_mode = None
        if realtime_mode:
            render_mode = "human"
        if record_trajectory:
            render_mode = "rgb_array"

        # Instantiate the environment and apply various wrappers
        self._env = gym.make(env_name, render_mode=render_mode, agent_view_size=self._default_reset_params["view-size"], tile_size=28)
        self._env = RGBImgPartialObsWrapper(self._env, tile_size=28)
        self._env = ImgObsWrapper(self._env)

        self._realtime_mode = realtime_mode
        self._record = record_trajectory

        # Prepare observation space
        self._visual_observation_space = spaces.Box(
                low = 0,
                high = 1.0,
                shape = (84, 84, 3),
                dtype = np.float32)

        # Set action space
        if "Memory" in env_name:
            self._action_space = spaces.Discrete(4)
            self._action_names = [["left", "right", "forward", "no-ops"]] # pickup is used as a no-ops action
        else:
            self._action_space = self._env.action_space
            self._action_names = [["left", "right", "forward", "pickup", "drop", "toggle", "done"]]

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
        return self._action_space

    @property
    def seed(self):
        """Returns the seed of the current episode."""
        return self._seed

    @property
    def action_names(self):
        """Returns a list of action names."""
        return self._action_names

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
        # Set default reset parameters if none were provided
        if reset_params is None:
            reset_params = self._default_reset_params
        else:
            reset_params = reset_params
            
        # Set seed
        self._seed = randint(reset_params["start-seed"], reset_params["start-seed"] + reset_params["num-seeds"] - 1)
        
        # Track rewards of an entire episode
        self._rewards = []
        # Reset the environment and retrieve the initial observation
        obs, _ = self._env.reset(seed=self._seed)
        # Retrieve the RGB frame of the agent's vision
        vis_obs = obs.astype(np.float32) / 255.

        # Render environment?
        if self._realtime_mode:
            self._env.render()

        # Prepare trajectory recording
        self._trajectory = {
            "vis_obs": [self._env.render().astype(np.uint8)], "vec_obs": [None],
            "rewards": [0.0], "actions": []
        } if self._record else None # The render function seems to be very very costly, so don't use this even once during training or evaluation

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
            {dict} -- Further episode information (e.g. cumulated reward) retrieved from the environment once an episode completed
        """
        obs, reward, done, truncated, info = self._env.step(action[0])
        self._rewards.append(reward)
        # Retrieve the RGB frame of the agent's vision
        vis_obs = obs.astype(np.float32) / 255.

        # Render the environment in realtime
        if self._realtime_mode:
            self._env.render()
            time.sleep(0.5)

        # Record trajectory data
        if self._record:
            self._trajectory["vis_obs"].append(self._env.render().astype(np.uint8))
            self._trajectory["vec_obs"].append(None)
            self._trajectory["rewards"].append(reward)
            self._trajectory["actions"].append(action)
        
        # Check time limit
        if len(self._rewards) == self.max_episode_steps:
            done = True

        # Wrap up episode information once completed (i.e. done)
        if done or truncated:
            success = 1.0 if sum(self._rewards) > 0 else 0.0
            info = {"reward": sum(self._rewards),
                    "length": len(self._rewards),
                    "success": success}
        else:
            info = None

        return vis_obs, None, reward, done or truncated, info

    def close(self):
        """Shuts down the environment."""
        self._env.close()
