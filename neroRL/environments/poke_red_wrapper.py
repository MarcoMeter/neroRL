import uuid
from pathlib import Path


import numpy as np
import gymnasium as gym
from random import randint
from neroRL.environments.env import Env
from neroRL.environments.poke_red.red_gym_env import RedGymEnv

class PokeRedWrapper(Env):
    """This class wraps the pokemon red environment.
    https://github.com/PWhiddy/PokemonRedExperiments
    """

    def __init__(self, env_name, reset_params = None, realtime_mode = False, record_trajectory = False):
        """Instantiates the Pokemon Red environment.

        Arguments:
            env_name {string} -- Path to the pokemon red ROM

        Keyword Arguments:
            reset_params {dict} -- Provides parameters (default: {None})
        """
        # Set default reset parameters if none were providedfmas
        if reset_params is None:
            self._default_reset_params = {"start-seed": 0, "num-seeds": 100, "initial-state": "./neroRL/environments/poke_red/has_pokedex_nballs.state"}
        else:
            self._default_reset_params = reset_params

        self._realtime_mode = realtime_mode
        self._record = record_trajectory

        sess_id = str(uuid.uuid4())[:8]
        Path("./poke").mkdir(parents=True, exist_ok=True)
        sess_path = Path(f'./poke/session_{sess_id}')
        config = {
                        'headless': False, 'save_final_state': True, 'early_stop': False,
                        'action_freq': 24, 'init_state': reset_params["initial-state"], 'max_steps': reset_params["max-steps"], 
                        'print_rewards': True, 'save_video': False, 'fast_video': True, 'session_path': sess_path,
                        'gb_path': env_name, 'debug': False, 'sim_frame_dist': 2_000_000.0, 
                        'use_screen_explore': True, 'reward_scale': 4, 'extra_buttons': True,
                        'explore_weight': 3 # 2.5
                    }
        if not realtime_mode:
            config["headless"] = True
            config["print_rewards"] = False
        self._max_episode_steps = config["max_steps"]

        # Initialize environment
        self._env = RedGymEnv(config)

        # Prepare observation space
        self._visual_observation_space = self._env.observation_space


    @property
    def unwrapped(self):
        """Return this environment in its vanilla (i.e. unwrapped) state."""
        return self

    @property
    def visual_observation_space(self):
        """Returns the shape of the visual component of the observation space as a tuple."""
        return self._env.observation_space

    @property
    def vector_observation_space(self):
        """Returns the shape of the vector component of the observation space as a tuple."""
        return None

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
        return self._seed

    @property
    def action_names(self):
        """Returns a list of action names."""
        return [["down", "left", "right", "up", "A", "B", "start", "select"]]

    @property
    def get_episode_trajectory(self):
        """Returns the trajectory of an entire episode as dictionary (vis_obs, vec_obs, rewards, actions). 
        """
        self._trajectory["action_names"] = self.action_names
        return self._trajectory if self._trajectory else None

    def reset(self, reset_params = None):
        """Resets the environment.
        
        Keyword Arguments:
            reset_params {dict} -- Provides parameters. (default: {None})
        
        Returns:
            {numpy.ndarray} -- Visual observation
            {numpy.ndarray} -- Vector observation
        """
        # Set default reset parameters if none were provided
        if reset_params is None:
            reset_params = self._default_reset_params

        # Sample seed
        self._seed = randint(reset_params["start-seed"], reset_params["start-seed"] + reset_params["num-seeds"] - 1)

        # Track rewards of an entire episode
        self._rewards = []

        # Retrieve the agent's initial observation
        vis_obs, _ = self._env.reset(seed=self._seed)

        # Render environment?
        if self._realtime_mode:
            self._env.render()

        # Prepare trajectory recording
        if self._record:
            self._trajectory = {
                "vis_obs": [self._env.render()], "vec_obs": [None],
                "rewards": [0.0], "actions": [], "frame_rate": 20
            }

        return vis_obs / 255.0, None, {}

    def step(self, action):
        """Runs one timestep of the environment's dynamics.
        
        Arguments:
            action {int} -- The to be executed action
        
        Returns:
            {numpy.ndarray} -- Visual observation
            {numpy.ndarray} -- Vector observation
            {float} -- (Total) Scalar reward signaled by the environment
            {bool} -- Whether the episode of the environment terminated
            {dict} -- Further information (e.g. episode length) retrieved from the environment once an episode completed
        """
        # Execute action
        vis_obs, reward, done, truncation, info = self._env.step(action[0])
        self._rewards.append(reward)

        # Record trajectory data
        if self._record:
            self._trajectory["vis_obs"].append(self._env.render())
            self._trajectory["vec_obs"].append(None)
            self._trajectory["rewards"].append(reward)
            self._trajectory["actions"].append(action)

        # Wrap up episode information once completed (i.e. done)
        if done or truncation:
            info = {"reward": sum(self._rewards),
                    "length": len(self._rewards)}
        else:
            info = None

        return vis_obs / 255.0, None, reward, done or truncation, info

    def close(self):
        """Shuts down the environment."""
        self._env.close()
