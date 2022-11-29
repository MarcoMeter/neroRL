import numpy as np
from gymnasium import spaces
from random import randint

from neroRL.environments.obstacle_tower_env import ObstacleTowerEnv
from neroRL.environments.env import Env

class ObstacleTowerWrapper(Env):
    """This class wraps the Obstacle Tower environment.
    https://github.com/Unity-Technologies/obstacle-tower-env

    In comparison to the original environment, this version uses a restricted set of actions:
        Action Dim A: No-Op, Move Forward
        Action Dim B: No-Op, Rotate Counter-Clockwise, Rotate Clockwise
        Action Dim C: No-Op, Jump
    Therefore the actions: Move Left, Move Right and Move Backward are not used.
    If flattened to a single dimension, 10 actions result.
    """

    def __init__(self, env_path, reset_params = None, worker_id = 1, no_graphis = False, realtime_mode = False, record_trajectory = False):
        """Instantiates the Obstacle Tower environment. It reduces the original action space by one dimension.
        
        Arguments:
            env_path {string} -- Path to the executable of the Obstacle Tower environment
            reset_params {dict} -- Reset parameters of the environment such as the seed and the maximum tower floors (default: {None})
            worker_id {int} -- Specifies the offset for the port to communicate with the environment (default: {1})
            no_graphics {bool} -- Whether to render the environment or not (default: {False})
            realtime_mode {bool} -- Whether the environment should run in realtime or as fast as possible (default: {False})
            record_trajectory {bool} -- Whether to record the trajectory of an entire episode. This can be used for video recording. (default: {False})
        """
        # Prepare reset parameters
        # As the keys "num-seeds", "start-seed", "retro-vis-obs" and "flat-action-space" cannot be interpreted by the environment
        # They have to be seperated from the overall reset parameters
        reset_params_clone = {}
        if reset_params is not None:
            for key in reset_params:
                if key != "start-seed" and key != "num-seeds" and key != "retro-vis-obs" and key != "flat-action-space":
                    reset_params_clone[key] = reset_params[key]
        self._default_start_seed = 0 if not reset_params else reset_params["start-seed"]
        self._default_num_seeds = 100 if not reset_params else reset_params["num-seeds"]
        self._retro_vis_obs = False if not reset_params or "retro-vis-obs" not in reset_params else reset_params["retro-vis-obs"]
        self._flat_action_space = False if not reset_params or "flat-action-space" not in reset_params else reset_params["flat-action-space"]

        # Starting floor (necessary to track the current floor)
        self._starting_floor = reset_params_clone["starting-floor"] if "starting-floor" in  reset_params_clone else 0

        # Instantiate environment
        self._env = ObstacleTowerEnv(env_path, config = reset_params_clone, worker_id = worker_id, realtime_mode = realtime_mode, retro = False)

        # Whether to record the trajectory of an entire episode
        self._record = record_trajectory

        # Flattened actions for a singular discrete action space
        self.flat_actions = [
            [0, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 1, 1, 0],
            [0, 2, 0, 0],
            [0, 2, 1, 0],
            [1, 0, 0, 0],
            [1, 1, 0, 0],
            [1, 1, 1, 0],
            [1, 2, 0, 0],
            [1, 2, 1, 0]]

    @property
    def unwrapped(self):
        """Return this environment in its vanilla (i.e. unwrapped) state."""
        return self

    @property
    def visual_observation_space(self):
        """Returns the shape of the visual component of the observation space as a tuple."""
        return self._env.observation_space[0]

    @property
    def vector_observation_space(self):
        """Returns the shape of the vector component of the observation space as a tuple."""
        if self._retro_vis_obs:
            return None
        else:
            return (2,)

    @property
    def action_space(self):
        """Returns the shape of the action space of the agent."""
        if self._flat_action_space:
            return spaces.Discrete(10)
        else:
            return spaces.MultiDiscrete((2,3,2))

    @property
    def seed(self):
        """Returns the seed of the current episode."""
        return self._seed

    @property
    def action_names(self):
        """Returns a list of action names."""
        if self._flat_action_space:
            return ["No-Op", "Rotate Left", "Rotate Left + Jump", "Rotate Right", "Rotate Right + Jump",
                    "Move Forward", "Move Forward + Rotate Left", "Move Forward + Rotate Left + Jump",
                    "Move Forward + Rotate Right", "Move Forward + Rotate Right + Jump"]
        else:
            return [["No-Op", "Forward"], ["No-Op", "Rotate CC", "Rotate C"], ["No-Op", "Jump"]]

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
        # Process reset parameters
        # Changing other parameters cannot be done at this moment (you have to initialize a new environment)
        if reset_params is not None:
            start_seed = reset_params["start-seed"]
            num_seeds = reset_params["num-seeds"]
        else:
            start_seed = self._default_start_seed
            num_seeds = self._default_num_seeds
        # Track rewards of an entire episode
        self._rewards = []
        # Track current floor
        self._current_floor = self._starting_floor
        # Sample seed
        self._seed = randint(start_seed, start_seed + num_seeds - 1)
        self._env.seed(self._seed)
        # Reset the environment and retrieve the initial observation
        obs = self._env.reset()
        # Retrieve the RGB frame of the agent's vision and the vector observation
        vis_obs = obs[0].astype(np.float32)
        # Add vector observation to the visual observation
        if self._retro_vis_obs:
            vis_obs = self._add_stats_to_image(vis_obs, obs[1], obs[2])
        normalized_time = np.clip(obs[2], 0, 10000) / 10000.
        vec_obs = np.array([obs[1], normalized_time])

        # Prepare trajectory recording
        self._trajectory = {
            "vis_obs": [(vis_obs * 255).astype(np.uint8)], "vec_obs": [vec_obs],
            "rewards": [0.0], "actions": []
        }

        if self._retro_vis_obs:
            return vis_obs, None
        else:
            return vis_obs, vec_obs

    def step(self, action):
        """Runs one timestep of the environment's dynamics.
        
        Arguments:
            action {list} -- The to be executed multi-discrete action as a list of indices
        
        Returns:
            {numpy.ndarray} -- Visual observation
            {numpy.ndarray} -- Vector observation
            {float} -- (Total) Scalar reward signaled by the environment
            {bool} -- Whether the episode of the environment terminated
            {dict} -- Further episode information (e.g. reached floor) retrieved from the environment once an episode completed
        """
        if self._flat_action_space:
            # Execute action that is selected from the flattened singular action space
            obs, reward, done, info = self._env.step(np.array(self.flat_actions[action[0]]))
        else:
            # Add dummy action, because we are not using the last action dimension that lets the agent stride left or right
            action = np.append(action, 0)
            # Execute MultiDiscrete action
            obs, reward, done, info = self._env.step(action)
        self._rewards.append(reward)
        # Retrieve the RGB frame of the agent's vision and the vector observation
        vis_obs = obs[0].astype(np.float32)
        # Add vector observation to the visual observation
        if self._retro_vis_obs:
            vis_obs = self._add_stats_to_image(vis_obs, obs[1], obs[2])
        normalized_time = np.clip(obs[2], 0, 10000) / 10000.
        vec_obs = np.array([obs[1], normalized_time])
        # Update current floor
        if reward >= 1:
            self._current_floor +=1

        # Record trajectory data
        if self._record:
            self._trajectory["vis_obs"].append((vis_obs * 255).astype(np.uint8))
            self._trajectory["vec_obs"].append(vec_obs)
            self._trajectory["rewards"].append(reward)
            self._trajectory["actions"].append(action)

        # Wrap up episode information once completed (i.e. done)
        if done:
            info = {"reward": sum(self._rewards),
                    "length": len(self._rewards),
                    "floor": self._current_floor}
        else:
            info = None

        if self._retro_vis_obs:
            return vis_obs, None, reward, done, info
        else:
            return vis_obs, vec_obs, reward, done, info

    def close(self):
        """Shuts down the environment."""
        self._env.close()

    @staticmethod
    def _add_stats_to_image(vis_obs, keys, time_left):
        """Adds the time left and the number of keys to the visual observation.

        Arguments:
            vis_obs {numpy.ndarray} -- The visual observation of the agent
            keys {int} -- The number of keys
            time_left {int} -- The agent's remaining time

        Returns:
            {numpy.ndarray} -- The processsed visual observation that displays the agent's number of keys and the remaining time
        """
        time_left = min(time_left, 10000) / 10000
        vis_obs[0:20, :, :] = 0
        vis_obs[10:20, 0:int(time_left * 168), 1] = 1
        if keys > 0:
            vis_obs[0:10, 0:int(33.6), 0:2] = 1
        return vis_obs