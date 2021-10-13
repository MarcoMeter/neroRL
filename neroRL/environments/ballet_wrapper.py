import numpy as np

from gym import spaces

from neroRL.environments.env import Env
from neroRL.environments.ballet.ballet_environment import BalletEnvironment
from neroRL.environments.ballet.ballet_environment_core import DANCE_SEQUENCES

class BalletWrapper(Env):
    """
    Based on
    https://github.com/deepmind/deepmind-research/tree/master/hierarchical_transformer_memory/pycolab_ballet
    """

    def __init__(self, reset_params = None, realtime_mode = False, record_trajectory = False) -> None:
        # Set default reset parameters if none were provided
        if reset_params is None:
            self._default_reset_params = {"start-seed": 0, "num-seeds": 100, "num-dancers": 2, "dance-delay": 1}
        else:
            self._default_reset_params = reset_params

        self._realtime_mode = realtime_mode
        self._record = record_trajectory

        # Initialize environment
        self._env = BalletEnvironment(2, 1, 320, rng=None)
        self._dance_types = list(DANCE_SEQUENCES.keys())

        # Setup observation spaces
        self._visual_observation_space = spaces.Box(
                low = 0,
                high = 1.0,
                shape = (99, 99, 3),
                dtype = np.float32)
        self._vector_observatoin_space = (len(self._dance_types), )

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
        return self._vector_observatoin_space

    @property
    def action_space(self):
        """Returns the shape of the action space of the agent."""
        return spaces.Discrete(8)

    @property
    def action_names(self):
        """Returns a list of action names. It has to be noted that only the names of action branches are provided and not the actions themselves!"""
        [["", "", "", "", "", "", "", ""]]

    @property
    def get_episode_trajectory(self):
        """Returns a dictionary containing a complete trajectory of data comprising an entire episode. This is only used for recording a video."""
        self._trajectory["action_names"] = self.action_names
        return self._trajectory if self._trajectory else None

    def reset(self, reset_params = None):
        """Reset the environment. The provided config is a dictionary featuring reset parameters for the environment (e.g. seed)."""
        timestep = self._env.reset()
        vis_obs, command = timestep.observation
        vec_obs = self._command_to_one_hot(command)
        # Track rewards of an entire episode
        self._rewards = []

        # Prepare trajectory recording
        if self._record:
            self._trajectory = {
                "vis_obs": [(vis_obs * 255).astype(np.uint8)], "vec_obs": [vec_obs],
                "rewards": [0.0], "actions": [], "frame_rate": 20
            }

        return vis_obs, vec_obs

    def step(self, action):
        """Executes one step of the agent."""
        timestep = self._env.step(action[0])
        vis_obs, command = timestep.observation
        reward = timestep.reward
        vec_obs = self._command_to_one_hot(command)
        done = timestep.last()
        self._rewards.append(reward)

        # Record trajectory data
        if self._record:
            self._trajectory["vis_obs"].append((vis_obs * 255).astype(np.uint8))
            self._trajectory["vec_obs"].append(vec_obs)
            self._trajectory["rewards"].append(reward)
            self._trajectory["actions"].append(action)

        # Wrap up episode information once completed (i.e. done)
        if done:
            info = {"reward": sum(self._rewards),
                    "length": len(self._rewards)}
        else:
            info = None

        return vis_obs, vec_obs, reward, done, info

    def close(self):
        """Shuts down the environment."""
        self._env.close()

    def _command_to_one_hot(self, command):
        encoding = np.zeros(self._vector_observatoin_space, dtype=np.float32)
        index = 0
        if command == "watch":
            return encoding
        index = self._dance_types.index(command)
        encoding[index] = 1.0
        return encoding
        