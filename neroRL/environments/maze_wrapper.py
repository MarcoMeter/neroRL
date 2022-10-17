# https://github.com/ml-jku/helm/
import numpy as np
import gym
import time

from mazelab.generators import random_maze
from mazelab import BaseMaze
from mazelab import Object
from mazelab import DeepMindColor as color
from mazelab import BaseEnv
from mazelab import VonNeumannMotion

from neroRL.environments.env import Env
from gym import error, spaces
from gym.spaces import Box
from gym.spaces import Discrete
from random import randint

class MazeWrapper(Env):
    def __init__(self, reset_params = None, realtime_mode = False, record_trajectory = False):
        self._env = gym.make("RandomMaze-v0")
        # Set default reset parameters if none were provided
        if reset_params is None:
            self._default_reset_params = {"start-seed": 0, "num-seeds": 100}
        else:
            self._default_reset_params = reset_params
        
        # Prepare observation space
        self._visual_observation_space = spaces.Box(
                low = 0,
                high = 1.0,
                shape = (9, 9, 3),
                dtype = np.float32)
        
        self._action_space = self._env.action_space
        self._action_names = ["North", "South", "West", "East"]
        self._realtime_mode = realtime_mode
        self._record = record_trajectory
        self._trajectory = {
            "vis_obs": [], "vec_obs": [None],
            "rewards": [0.0], "actions": []
        }
        
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
    def get_episode_trajectory(self):
        """Returns the trajectory of an entire episode as dictionary (vis_obs, vec_obs, rewards, actions). 
        """
        self._trajectory["action_names"] = self.action_names
        return self._trajectory if self._trajectory else None
    
    @property
    def action_names(self):
        """Returns a list of action names."""
        return self._action_names
    
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
        seed = randint(reset_params["start-seed"], reset_params["start-seed"] + reset_params["num-seeds"] - 1)
        self._env.seed(seed)
        np.random.seed(seed)
        # Track rewards of an entire episode
        self._rewards = []
        # Reset the environment and retrieve the initial observation
        obs = self._env.reset()
        # Retrieve the RGB frame of the agent's vision
        vis_obs = obs.astype(np.float32) / 255.

        # Render environment?
        if self._realtime_mode:
           self._env.render(mode = "human")
        # Prepare trajectory recording
        #self._trajectory = None # The render function seems to be very very costly, so don't use this even once during training or evaluation

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
        vis_obs = obs.astype(np.float32) / 255.

        # Render the environment in realtime
        if self._realtime_mode:
            self._env.render(mode = "human")
            time.sleep(0.1)

        # Record trajectory data
        if self._record:
            self._trajectory["vis_obs"].append(self._env.get_image().astype(np.uint8))
            self._trajectory["vec_obs"].append(None)
            self._trajectory["rewards"].append(reward)
            self._trajectory["actions"].append(action)
        
        # Wrap up episode information once completed (i.e. done)
        if done:
            success = 1.0 if reward == 1 else 0.0
            info = {"reward": sum(self._rewards),
                    "length": len(self._rewards),
                    "success": success}
        else:
            info = {"reward": sum(self._rewards), "length": len(self._rewards)}

        return vis_obs, None, reward, done, info

    def close(self):
        """Shuts down the environment."""
        self._env.close()

class Maze(BaseMaze):

    def __init__(self, width=81, height=51, complexity=.25, density=.25, agent_view_size=4):
        x = random_maze(width=width, height=height, complexity=complexity, density=density)
        height, width = x.shape
        new_maze = np.ones((height+2*agent_view_size, width+2*agent_view_size))
        new_maze[agent_view_size:height+agent_view_size, agent_view_size:width+agent_view_size] = x
        self.x = new_maze
        super(Maze, self).__init__()

    @property
    def size(self):
        return self.x.shape

    def make_objects(self):
        free = Object('free', 0, color.free, False, np.stack(np.where(self.x == 0), axis=1))
        obstacle = Object('obstacle', 1, color.obstacle, True, np.stack(np.where(self.x == 1), axis=1))
        agent = Object('agent', 2, color.agent, False, [])
        goal = Object('goal', 3, color.goal, False, [])
        return free, obstacle, agent, goal

    def print(self):
        rows = []
        for row in range(self.x.shape[0]):
            str = np.array(self.x[row], dtype=np.str)
            rows.append(' '.join(str))
        print('\n'.join(rows))


class Env(BaseEnv):
    def __init__(self, complexity=.75, density=.75, agent_view_size=4):
        super().__init__()

        self.complexity = complexity
        self.density = density
        self.agent_view_size = agent_view_size
        # self.maze = Maze(width, height, complexity, density, agent_view_size)
        self.motions = VonNeumannMotion()
        self._sample_env()

        self.observation_space = Box(low=0, high=255, shape=(agent_view_size*2+1, agent_view_size*2+1, 3), dtype=np.uint8)
        self.action_space = Discrete(len(self.motions))


    def _sample_env(self):
        size = np.random.choice(np.arange(5, 26), size=1)[0]
        self.maze = Maze(width=size, height=size, complexity=self.complexity, density=self.density,
                         agent_view_size=self.agent_view_size)
        free_pos = self.maze.objects.free.positions
        start_ind = np.random.choice(len(free_pos))
        self.start_idx = [free_pos[start_ind]]
        self.goal_idx = [[self.agent_view_size + size - 2, self.agent_view_size + size - 2]]

    def _get_pomdp_view(self):
        full_obs = self.get_image()
        y, x = self.maze.objects.agent.positions[0]
        view_size = self.agent_view_size
        partial_obs = full_obs[y-view_size:y+view_size+1, x-view_size:x+view_size+1]
        return partial_obs

    def step(self, action):
        motion = self.motions[action]
        current_position = self.maze.objects.agent.positions[0]
        new_position = [current_position[0] + motion[0], current_position[1] + motion[1]]
        valid = self._is_valid(new_position)
        if valid:
            self.maze.objects.agent.positions = [new_position]

        if self._is_goal(new_position):
            reward = +1
            done = True
        elif not valid:
            reward = -1
            done = False
        else:
            reward = -0.01
            done = False
        return self._get_pomdp_view(), reward, done, {}

    def reset(self):
        self._sample_env()
        self.maze.objects.agent.positions = self.start_idx
        size = self.maze.size[0]
        self.maze.objects.goal.positions = [[size-2-self.agent_view_size, size-2-self.agent_view_size]]
        return self._get_pomdp_view()

    def _is_valid(self, position):
        nonnegative = position[0] >= 0 and position[1] >= 0
        within_edge = position[0] < self.maze.size[0] and position[1] < self.maze.size[1]
        passable = not self.maze.to_impassable()[position[0]][position[1]]
        return nonnegative and within_edge and passable

    def _is_goal(self, position):
        out = False
        for pos in self.maze.objects.goal.positions:
            if position[0] == pos[0] and position[1] == pos[1]:
                out = True
                break
        return out

    def get_image(self):
        return self.maze.to_rgb()


gym.envs.register(id="RandomMaze-v0", entry_point=Env, max_episode_steps=100)
