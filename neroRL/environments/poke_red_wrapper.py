import uuid
import numpy as np
import os
from random import randint
from gymnasium import spaces
from pathlib import Path

from neroRL.environments.poke_red.red_gym_env_v2 import RedGymEnv
from neroRL.environments.env import Env

class PokeRedV2Wrapper(Env):
    """
    This class wraps pokemon red v2 by pdubs.
    """
    
    def __init__(self, env_name, reset_params = None, realtime_mode = False, record_trajectory = False) -> None:
        """Instantiates the memory-gym environment.
        
        Arguments:
            env_name {string} -- Path to the rom file.
            reset_params {dict} -- Provides parameters, like a seed, to configure the environment. (default: {None})
            realtime_mode {bool} -- Whether to render the environment in realtime. (default: {False})
            record_trajectory {bool} -- Whether to record the trajectory of an entire episode. This can be used for video recording. (default: {False})
        """
        if reset_params is None:
            self._default_reset_params = {
                "start-seed": 0,
                "num-seeds": 100,
                "initial-state": "./neroRL/environments/poke_red/has_pokedex_nballs.state",
                "max_steps": 2048 * 40,
                "reward_scale": 0.5,
                "event_weight": 4.0,
                "level_weight": 1.0,
                "heal_weight": 5.0,
                "op_lvl_weight": 0.2,
                "dead_weight": -0.1,
                "badge_weight": 0.5,
                "explore_weight": 1.0,
                "use_explore_map_obs": True,
                "use_recent_actions_obs": True,
                "zero_recent_actions": False
            }
            reset_params = self._default_reset_params
        else:
            self._default_reset_params = reset_params

        # Setup
        self._max_episode_steps = reset_params["max_steps"]
        sess_id = str(uuid.uuid4())[:8]
        os.makedirs("./session", exist_ok=True)
        sess_path = Path(f'session/{sess_id}')
        env_config = {
                'headless': not realtime_mode,
                'save_final_state': False,
                'early_stop': False,
                'action_freq': 24,
                'init_state': reset_params["init-state"],
                'max_steps': self._max_episode_steps, 
                'print_rewards': False,
                'save_video': False,
                'fast_video': True,
                'session_path': sess_path,
                'gb_path': env_name,
                'debug': False,
                'reset_params': reset_params,
            }
        
        # Instantiate env
        self._env = RedGymEnv(env_config)

        # Prepare observation space:
        # health, level, badges, and recent_actions shall be concatenated into a single vector observation
        # Totaling to 4 modalities: screens, map, events, and game_state
        shape = self._env.observation_space.spaces['screens'].shape
        screen_space = spaces.Box(low=0.0, high=1.0, shape=shape, dtype=np.float32)
        if reset_params["use_explore_map_obs"]:
            shape = self._env.observation_space.spaces['map'].shape
            map_space = spaces.Box(low=0.0, high=1.0, shape=shape, dtype=np.float32)
        shape = self._env.observation_space.spaces['events'].shape
        event_space = spaces.Box(low=0.0, high=1.0, shape=shape, dtype=np.float32)
        num_game_state_obs = self._env.observation_space.spaces['health'].shape[0]
        num_game_state_obs += self._env.observation_space.spaces['level'].shape[0]
        num_game_state_obs += self._env.observation_space.spaces['badges'].shape[0]
        self.use_recent_actions_obs = reset_params["use_recent_actions_obs"]
        if self.use_recent_actions_obs:
            num_game_state_obs += self._env.observation_space.spaces['recent_actions'].shape[0]
        shape = (num_game_state_obs,)
        game_state_space = spaces.Box(low=-1.0, high=1.0, shape=shape, dtype=np.float32)
        obs_spaces = {
            'screens': screen_space, 'events': event_space, 'game_state': game_state_space
        }
        if reset_params["use_explore_map_obs"]:
            obs_spaces["map"] = map_space
        self._observation_space = spaces.Dict(obs_spaces)

        self._realtime_mode = realtime_mode
        self._record = record_trajectory

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
        return None

    @property
    def action_space(self):
        """Returns the action space of the agent."""
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
        """Returns a list of action names. It has to be noted that only the names of action branches are provided and not the actions themselves!"""
        return [["down", "left", "right", "up", "a", "b", "start"]]

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

        self._rewards = []

        # Reset the environment to retrieve the initial observation
        obs, info = self._env.reset(seed=self._seed, options=options)
        # Prepare observations so that the keys health, level, badges, and recent_actions are concatenated
        if self.use_recent_actions_obs:
            vec_obs = np.concatenate([obs["health"], obs["level"], obs["badges"], obs["recent_actions"]])
        else:
            vec_obs = np.concatenate([obs["health"], obs["level"], obs["badges"]])
        obs_out = {
            "screens": obs["screens"] / 255.0,
            "events": obs["events"],
            "game_state": vec_obs
        }
        if "map" in obs:
            obs_out["map"] = obs["map"] / 255.0
    
        if self._realtime_mode:
            self._env.render()

        # Prepare trajectory recording
        self._trajectory = {
            "vis_obs": [self._env.render()], "vec_obs": [None],
            "rewards": [0.0], "actions": [], "frame_rate": 100
        } if self._record else None

        return obs_out, info

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
        # Step the environment
        obs, reward, done, truncation, info = self._env.step(action)
        # Prepare observations so that the keys health, level, badges, and recent_actions are concatenated
        if self.use_recent_actions_obs:
            vec_obs = np.concatenate([obs["health"], obs["level"], obs["badges"], obs["recent_actions"]])
        else:
            vec_obs = np.concatenate([obs["health"], obs["level"], obs["badges"]])
        obs_out = {
            "screens": obs["screens"] / 255.0,
            "events": obs["events"],
            "game_state": vec_obs
        }
        if "map" in obs:
            obs_out["map"] = obs["map"] / 255.0

        self._rewards.append(reward)

        if done or truncation:
            info["reward"] = sum(self._rewards)
            info["length"] = len(self._rewards)
            info["events_sum"] = sum(obs["events"])

        if self._realtime_mode:
            self._env.render()

        # Record trajectory data
        if self._record:
            self._trajectory["vis_obs"].append(self._env.render())
            self._trajectory["vec_obs"].append(None)
            self._trajectory["rewards"].append(reward)
            self._trajectory["actions"].append(action)

        return obs_out, reward, done or truncation, info

    def close(self):
        """Shuts down the environment."""
        # self._env.close()
        pass
