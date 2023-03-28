from neroRL.environments.env import Env
from random import randint
import gymnasium as gym
from gymnasium.spaces import Box, MultiDiscrete, Discrete

import popgym 

class POPGymWrapper(Env):
    """
    https://github.com/proroklab/popgym
    This class wraps POPGym environment.
    Available Environments:
        popgym-Autoencode{DIFFICULTY}-v0
        popgym-RepeatPrevious{DIFFICULTY}-v0
        popgym-RepeatFirst{DIFFICULTY}-v0
        popgym-CountRecall{DIFFICULTY}-v0
        popgym-StatelessCartPole{DIFFICULTY}-v0
        popgym-StatelessPendulum{DIFFICULTY}-v0
        popgym-NoisyStatelessCartPole{DIFFICULTY}-v0
        popgym-NoisyStatelessPendulum{DIFFICULTY}-v0
        popgym-MultiarmedBandit{DIFFICULTY}-v0
        popgym-HigherLower{DIFFICULTY}-v0
        popgym-Battleship{DIFFICULTY}-v0
        popgym-Concentration{DIFFICULTY}-v0
        popgym-MineSweeper{DIFFICULTY}-v0
        popgym-LabyrinthExplore{DIFFICULTY}-v0
        popgym-LabyrinthEscape{DIFFICULTY}-v0
    
    Where
        {DIFFICULTY} => Easy | Medium | Hard
    
    Example:
        popgym-AutoencodeEasy-v0, popgym-AutoencodeMedium-v0, popgym-AutoencodeHard-v0
    """
    def __init__(self, env_name, reset_params = None, realtime_mode = False, record_trajectory = False) -> None:
        """Instantiates the POPGym environment.
        
        Arguments:
            env_name {string} -- Name of the POPGym environment
            reset_params {dict} -- Provides parameters, like a seed, to configure the environment. (default: {None})
            realtime_mode {bool} -- Whether to render the environment in realtime. (default: {False})
            record_trajectory {bool} -- Whether to record the trajectory of an entire episode. This can be used for video recording. (default: {False})
        """
        if reset_params is None:
            self._default_reset_params = {"start-seed": 0, "num-seeds": 100}
        else:
            self._default_reset_params = reset_params

        render_mode = None if not realtime_mode else "rgb_array"
        # The keyword: render_mode is only supported for some environments
        if render_mode is not None:
            self._env = gym.make(env_name, render_mode=render_mode)
        else:
            self._env = gym.make(env_name)

        self._realtime_mode = realtime_mode
        self._record = record_trajectory

        self._vector_observation_space, self._visual_observation_space = None, None
        if isinstance(self._env.observation_space, Box):
            self._vector_observation_space = self._env.observation_space.shape
        if isinstance(self._env.observation_space, Discrete):
            self._vector_observation_space = (1,)

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
        return self._vector_observation_space

    @property
    def action_space(self):
        """Returns the shape of the action space of the agent."""
        return self._env.action_space

    @property
    def max_episode_steps(self):
        """Returns the maximum number of steps that an episode can last."""
        return self._env.max_episode_length

    @property
    def seed(self):
        """Returns the seed of the current episode."""
        return self._seed

    @property
    def action_names(self):
        """Returns a list of action names. It has to be noted that only the names of action branches are provided and not the actions themselves!"""
        pass

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
        # Process reset parameters
        if reset_params is None:
            reset_params = self._default_reset_params
        else:
            reset_params = reset_params

        # Sample seed
        self._seed = randint(reset_params["start-seed"], reset_params["start-seed"] + reset_params["num-seeds"] - 1)
        

        vis_obs, vec_obs = None, None

        if self.vector_observation_space is not None:
            vec_obs, info = self._env.reset(seed = self._seed)
            
        return vis_obs, vec_obs

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
        if len(action) == 1:
            action = action[0]
            
        vis_obs, vec_obs = None, None
        
        if self.vector_observation_space is not None:
            vec_obs, reward, done, truncation, info = self._env.step(action)

        #if type(self._env.observation_space) is spaces.Dict:
        #    vis_obs = obs["visual_observation"]
        #    vec_obs = obs["vector_observation"]
        #else:
        #    vis_obs = obs
        #    vec_obs = None

        if self._realtime_mode or self._record:
            img = self._env.render()

        # Record trajectory data
        if self._record:
            self._trajectory["vis_obs"].append(img)
            self._trajectory["vec_obs"].append(vec_obs)
            self._trajectory["rewards"].append(reward)
            self._trajectory["actions"].append(action)

        return vis_obs, vec_obs, reward, done or truncation, info

    def close(self):
        """Shuts down the environment."""
        self._env.close()