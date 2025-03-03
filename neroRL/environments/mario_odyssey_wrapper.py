import gymnasium as gym
from gymnasium import spaces
import numpy as np
from random import randint

# DEBUG
import time

from neroRL.environments.odyssey_env import OdysseyEnv
from neroRL.environments.env import Env

class MarioOdysseyWrapper(Env):
    """This class wraps the Mario Odyssey environment."""

    def __init__(self, worker_id, reset_params = None, realtime_mode = False, record_trajectory = False):
        """Instantiates the Pendulum environment.

        Arguments:
            worker_id {int} -- Id of the worker

        Keyword Arguments:
            reset_params {dict} -- Provides parameters for configuring the environment (default: {None})
        """
        # Set default reset parameters if none were provided
        if reset_params is None:
            self._default_reset_params = {"start-seed": 0,
                                          "num-seeds": 200,
                                          "max_steps": 1536,
                                          "stage": "SandWorldMeganeExStageMap",
                                          "scenario": 0,
                                          "rom_path": "/scratch/odyssey/romfs",
                                          "action_bucket": [-1, -0.5, -0.25, 0, 0.25, 0.5, 1],
                                          "raycast_length": 900,
                                          "num_action_repeat": 5,
                                          "start_position": None,
                                          "use_waypoints": True,
                                          "waypoints": [  # Waypoints forming a path through the maze
                                                           [130, 150, 900],      # Obere Ecke vom L
                                                           [-1100, 150, 900],    # Linker Rand vom ersten Raum
                                                           [-2650, 150, 900],    # Rechter Rand vom zweiten Raum
                                                           [-3900, 150, 900],   # Linker Rand im zweiten Raum
                                                           [-3900, 150, 1700],   # Durchgang zum dritten Raum
                                                           [-5000, 50, 1700],    # Nach dem Durchgang
                                                           [-5000, 150, 500],    # Obere rechte Ecke dritter Raum
                                                           [-5850, 150, 500],    # Mittlere Plattform oben im dritten Raum
                                                           [-6700, 150, 500],    # Plattform vom Mond
                                                       ],
                                          "waypoint_range": 250.0  # Distance threshold for reaching a waypoint
                                          }
        else:
            self._default_reset_params = reset_params

        self._realtime_mode = realtime_mode
        self._record = record_trajectory
        self._max_steps = self._default_reset_params["max_steps"]
        self._num_action_repeat = self._default_reset_params["num_action_repeat"]
        self._raycast_length = self._default_reset_params["raycast_length"]
        self._use_waypoints = self._default_reset_params["use_waypoints"]
        self._waypoints = self._default_reset_params["waypoints"]
        self._waypoint_range = self._default_reset_params["waypoint_range"]
        self._visited_waypoints = set()

        # Initialize environment
        if realtime_mode:
            render_mode = "human"
        elif record_trajectory:
            render_mode = "rgb_array"
        else:
            render_mode = None
        
        ###### ENABLE DEBUG HERE ##############
        self.DEBUG_MODE = False
        #self.DEBUG_MODE = worker_id == 211
        
        #if self.DEBUG_MODE:
        #    render_mode = "human"
        #######################################
        
        self._env = OdysseyEnv(self._default_reset_params["stage"], self._default_reset_params["scenario"], str(worker_id), self._default_reset_params["rom_path"], render_mode=render_mode)
        # Count number of observations
        num_obs = 0
        for space in self._env.observation_space:
            if isinstance(self._env.observation_space[space], spaces.Box):
                num_obs += self._env.observation_space[space].shape[0]
            elif isinstance(self._env.observation_space[space], spaces.Discrete):
                if not space in ["isTouchingMoon", "isTouchingPoison"]:
                    num_obs += 1
            elif isinstance(self._env.observation_space[space], spaces.MultiBinary):
                num_obs += self._env.observation_space[space].n

        # Setup flattened observation space
        self._observation_space = spaces.Dict({"vec_obs": spaces.Box(low=-1, high=1, shape=(num_obs,))})

        # Setup Multi-Discrete action space
        self._bucket_size = len(self._default_reset_params["action_bucket"])
        self._continuous_action_bucket = np.linspace(-1, 1, self._bucket_size)
        self._action_space = spaces.MultiDiscrete([2, 2, 2, self._bucket_size, self._bucket_size, self._bucket_size, self._bucket_size])

    def dbg(self, *args):
        if self.DEBUG_MODE:
            print("DEBUG:", *args)#, flush=True)

    @property
    def unwrapped(self):
        """Return this environment in its vanilla (i.e. unwrapped) state."""
        return self

    @property
    def observation_space(self):
        """Returns the observation space of the environment."""
        return self._observation_space

    @property
    def action_space(self):
        """Returns the action space of the agent."""
        return self._action_space

    @property
    def max_episode_steps(self):
        """Returns the maximum number of steps that an episode can last."""
        return self._max_steps

    @property
    def seed(self):
        """Returns the seed of the current episode."""
        return self._seed

    @property
    def action_names(self):
        """Returns a list of action names."""
        return None

    @property
    def get_episode_trajectory(self):
        """Returns the trajectory of an entire episode as dictionary (vis_obs, vec_obs, rewards, actions)."""
        self._trajectory["action_names"] = self.action_names
        return self._trajectory if self._trajectory else None

    def _process_obs(self, obs):
        vec_obs = []

        for space, value in obs.items():
            if isinstance(self._env.observation_space[space], spaces.Box):
                if space == "raycastResults":
                    #self.dbg(value)
                    # Distance measures that are larger than the raycast length are clipped to -1
                    ray_obs = value  # Assign the updated array to ray_obs
                    ray_obs = ray_obs / self._raycast_length  # Normalize the values
                    ray_obs[ray_obs > 1] = 1  # Modify the array in place
                    vec_obs.append(ray_obs.flatten())  # Flatten and append to vec_obs
                    #self.dbg(ray_obs)
                else:
                    # Normalize Box spaces using low and high bounds
                    low = self._env.observation_space[space].low
                    high = self._env.observation_space[space].high
                    normalized_value = (value - low) / (high - low)
                    vec_obs.append(normalized_value.flatten())

            elif isinstance(self._env.observation_space[space], spaces.MultiBinary):
                # MultiBinary spaces are already bounded between 0 and 1, so no normalization is needed
                vec_obs.append(value.flatten())

            elif isinstance(self._env.observation_space[space], spaces.Discrete):
                # Normalize Discrete spaces as before
                if space not in ["isTouchingMoon", "isTouchingPoison"]:
                    vec_obs.append([value / self._env.observation_space[space].n])
        return np.concatenate(vec_obs)
    
    def _get_distance_to_moon(self, obs):
        position = obs["playerPos"]
        distance = np.linalg.norm(position - self.moon_position)
        return distance

    def _get_distance(self, pos1, pos2):
        return np.linalg.norm(np.array(pos1) - np.array(pos2))

    def _reward_waypoint(self, player_pos):
        if not self._waypoints:
            return 0.0
    
        for i, waypoint in enumerate(self._waypoints):
            waypoint_distance = self._get_distance(player_pos, waypoint)
            self.dbg("waypoint", i, "distance:", waypoint_distance)
            if waypoint_distance <= self._waypoint_range:
                self._waypoints.pop(i)  # Remove the reached waypoint
                self.dbg("Found waypoint", self._waypoint_count - len(self._waypoints))
                #return 1.0 / self._waypoint_count
                #return 1.0 - 0.9 * (self._current_step / self._max_steps)
                return min(1.0, 1.1 - 0.9 * (self._current_step / self._max_steps))

        return 0.0

    def reset(self, reset_params = None):
        """Resets the environment.
        
        Keyword Arguments:
            reset_params {dict} -- Provides parameters, like if the observed velocity should be masked. (default: {None})
        
        Returns:
            {dict} -- Observation of the environment
            {dict} -- Empty info
        """
        # Set default reset parameters if none were provided
        if reset_params is None:
            reset_params = self._default_reset_params

        # Sample seed
        self._seed = randint(reset_params["start-seed"], reset_params["start-seed"] + reset_params["num-seeds"] - 1)

        # Track rewards of an entire episode
        self._rewards = []
        self._current_step = 0
        self._max_steps = reset_params["max_steps"]
        # Waypoint members
        self._waypoint_range = reset_params["waypoint_range"]
        self._waypoints = reset_params["waypoints"].copy()
        self._waypoint_count = len(self._waypoints)

        # Retrieve the agent's initial observation
        obs, _ = self._env.reset(seed=self._seed, options={"startPos": reset_params["start_position"]})
        self.moon_position = np.array(self._waypoints[-1])
        self.moon_distance = self._get_distance_to_moon(obs)
        self.start_distance = self.moon_distance
        self.best_distance = self.moon_distance
        vec_obs = self._process_obs(obs)

        # Render environment?
        if self._realtime_mode:
            frame = self._env.render()

        # Prepare trajectory recording
        if self._record:
            self._trajectory = {
                "vis_obs": [frame], "vec_obs": [vec_obs],
                "rewards": [0.0], "actions": [], "frame_rate": 20
            }
        
        return {"vec_obs": vec_obs}, {}

    def step(self, action):
        """Runs one timestep of the environment's dynamics.
        
        Arguments:
            action {int} -- The to be executed action
        
        Returns:
            {dict} -- Observation of the environment
            {float} -- Scalar reward signaled by the environment
            {bool} -- Whether the episode of the environment terminated
            {dict} -- Further information (e.g. episode length) retrieved from the environment once an episode completed
        """
        # Map action to the original environment's action space
        action = {
            "buttons": np.asarray(action[:3]),
            "stickLeft": self._continuous_action_bucket[action[3:5]],
            "stickRight": self._continuous_action_bucket[action[5:7]]
        }

        # Execute action for num action repeat frames
        for _ in range(self._num_action_repeat):
            obs, reward, done, truncation, info = self._env.step(action)
            if done or truncation:
                break
                
        self.dbg("original reward:", reward)
        moon_found = reward > 0.0

        # Process obs, reward and done
        self._current_step += 1
        truncation = self._current_step == self._max_steps
        success = 0.0
        if reward <= -0.11:
            reward = 0.0
        if reward > 0.9:
            success = 1.0
            
        moon_current_distance = self._get_distance_to_moon(obs)
        self.dbg("moon_current_distance:", moon_current_distance)
        
        self.dbg("Calculate moon distance reward..")        
        if moon_current_distance < self.moon_distance:
            if moon_current_distance < self.best_distance:
                self.best_distance = moon_current_distance
                # give moon reward
                reward += 0.01 #* self.start_distance / moon_current_distance
            self.moon_distance = moon_current_distance
        self.dbg("best_distance:", self.best_distance)

        if self._use_waypoints:
            self.dbg("Calculate waypoints reward..")
            player_pos = obs["playerPos"]
            waypoint_reward = self._reward_waypoint(player_pos)
            reward += waypoint_reward
            self.dbg("waypoint_reward:", waypoint_reward)
            
        if moon_found:
            self.dbg("Found moon!")
            #reward *= 10

        vec_obs = self._process_obs(obs)
        self._rewards.append(reward)

        # Render environment?
        if self._realtime_mode:
            frame = self._env.render()

        # Record trajectory data
        if self._record:
            self._trajectory["vis_obs"].append(frame)
            self._trajectory["vec_obs"].append(vec_obs)
            self._trajectory["rewards"].append(reward)
            self._trajectory["actions"].append(action)

        # Wrap up episode information once completed (i.e. done)
        if done or truncation:
            info = {"reward": sum(self._rewards),
                    "length": len(self._rewards),
                    "success": success,
                    "best_distance": self.best_distance,
                    "waypoints_visisted": self._waypoint_count - len(self._waypoints)}
            self.dbg("info:", info)
        else:
            info = None
            
        self.dbg("curr. reward:", reward)
        #self.dbg("rewards:", self._rewards)
        self.dbg("cum. reward:", sum(self._rewards))
        if self.DEBUG_MODE:
            print("--", flush=True)
            #time.sleep(0.1)
            
        return {"vec_obs": vec_obs}, reward, done or truncation, info

    def close(self):
        """Shuts down the environment."""
        self._env.close()
