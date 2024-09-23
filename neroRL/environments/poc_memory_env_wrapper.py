from gymnasium import spaces
import gymnasium as gym
from random import randint
from neroRL.environments.env import Env
from neroRL.environments.pom_env import PoMEnv

class PocMemoryEnvWrapper(Env):
    """
    Proof of Concept Memory Environment

    This environment is intended to assess whether the underlying recurrent policy is working or not.
    The environment is based on a one dimensional grid where the agent can move left or right.
    At both ends, a goal is spawned that is either punishing or rewarding.
    During the very first two steps, the agent gets to know which goal leads to a positive or negative reward.
    Afterwards, this information is hidden in the agent's observation.
    The last value of the agent's observation is its current position inside the environment.
    Optionally and to increase the difficulty of the task, the agent's position can be frozen until the goal information is hidden.
    To further challenge the agent, the step_size can be decreased.
    """
    def __init__(self, reset_params = None, realtime_mode = False, record_trajectory = False):
        """
        Arguments:
            step_size {float} -- Step size of the agent. Defaults to 0.2.
            glob {bool} -- Whether to sample starting positions across the entire space. Defaults to False.
            freeze_agent {bool} -- Whether to freeze the agent's position until goal positions are hidden. Defaults to False.
        """
        if reset_params is None:
            self._default_reset_params = {"start-seed": 0, "num-seeds": 1000}
            reset_params = self._default_reset_params
        else:
            self._default_reset_params = reset_params

        self._realtime_mode = realtime_mode
        self._record = record_trajectory
        if realtime_mode:
            self._env = gym.make("ProofofMemory-v0", render_mode = "human")
        else:
            self._env = gym.make("ProofofMemory-v0", render_mode = "rgb_array")
        self._observation_space = spaces.Dict({"vec_obs": self._env.observation_space})

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
        """
        Returns:
            {spaces.Discrete}: The agent has two actions: going left or going right
        """
        return self._env.action_space

    @property
    def get_episode_trajectory(self):
        self._trajectory["action_names"] = self.action_names
        return self._trajectory if self._trajectory else None
    
    @property
    def action_names(self):
        return ["left", "right"]

    @property
    def seed(self):
        """Returns the seed of the current episode."""
        return self._seed

    @property
    def max_episode_steps(self):
        """Returns the maximum number of steps that an episode can last."""
        return 16
    
    def reset(self, reset_params = None):
        """Resets the environment.
        
        Keyword Arguments:
            reset_params {dict} -- Provides parameters, like if the observed velocity should be masked. (default: {None})
        
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

        # Reset environment
        obs, _ = self._env.reset(seed = self._seed)
        obs = {"vec_obs": obs}

        # Prepare trajectory recording
        self._trajectory = {
            "vis_obs": [self._env.render()], "vec_obs": [obs["vec_obs"]],
            "rewards": [0.0], "actions": []
        } if self._record else None

        return obs, {}

    def step(self, action):
        """
        Executes the agents action in the environment if the agent is allowed to move.

        Arguments:
            action {list} -- The agent action which should be executed.

        Returns:
            {dict} -- Observation of the environment.
            {float} -- Reward for the agent.
            {bool} -- Done flag whether the episode has terminated.
            {dict} -- Information about episode reward, length and agents success reaching the goal position
        """
        obs, reward, done, truncation, info = self._env.step(action)
        obs = {"vec_obs": obs}
        done = done or truncation

        if self._realtime_mode or self._record:
            img = self._env.render()

        # Record trajectory data
        if self._record:
            self._trajectory["vis_obs"].append(img)
            self._trajectory["vec_obs"].append(obs["vec_obs"])
            self._trajectory["rewards"].append(reward)
            self._trajectory["actions"].append(action)

        return obs, reward, done, info

    def close(self):
        """Clears the used resources properly."""
        self._env.close()
