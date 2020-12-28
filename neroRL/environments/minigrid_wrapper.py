import time
import numpy as np
import gym_minigrid
import gym
from gym import error, spaces
from random import randint
from neroRL.environments.env import Env

class MinigridWrapper(Env):
    """This class wraps Gym Minigrid environments.
    https://github.com/maximecb/gym-minigrid
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

    def __init__(self, env_name, reset_params = None, realtime_mode = False):
        """Instantiates the Minigrid environment.
        
        Arguments:
            env_name {string} -- Name of the Minigrid environment
            reset_params {dict} -- Provides parameters, like a seed, to configure the environment. (default: {None})
            realtime_mode {bool} -- Whether to render the environment in realtime. (default: {False})
        """
        # Set default reset parameters if none were provided
        if reset_params is None:
            self._default_reset_params = {"start-seed": 0, "num-seeds": 100}
        else:
            self._default_reset_params = reset_params

        self._env = gym.make(env_name)

        self._realtime_mode = realtime_mode

        # Prepare observation space
        self._visual_observation_space = spaces.Box(
                low = 0,
                high = 1.0,
                shape = (84, 84, 3),
                dtype = np.float32)

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
        return ["left", "right", "forward", "toggle", "pickup", "drop", "done"]

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
        # Retrieve the RGB frame of the agent"s vision
        vis_obs = self._env.get_obs_render(obs["image"], tile_size=12)
        vis_obs = vis_obs.astype(np.float32) / 255.

        # Render the environment in realtime
        if self._realtime_mode:
            self._env.render(tile_size = 96)

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
        obs, reward, done, info = self._env.step(action[0])
        self._rewards.append(reward)
        # Retrieve the RGB frame of the agent's vision
        vis_obs = self._env.get_obs_render(obs["image"], tile_size=12)  / 255.

        # Render the environment in realtime
        if self._realtime_mode:
            self._env.render(tile_size = 96)
            time.sleep(0.5)
        
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
