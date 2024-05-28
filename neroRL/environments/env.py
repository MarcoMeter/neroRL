from abc import ABC, abstractmethod, abstractproperty

class Env(ABC):
    """
    This abstract class helps to implement new environments that fit this implementation.
    It is to some extend inspired by OpenAI gym.
    The key difference is that an environment can have a visual and a vector observation space at the same time.
    The provided arguments of the to be implemented functions are not fixed, but are recommended at minimum.
    """

    @abstractproperty
    def unwrapped(self):
        """Return this environment in its vanilla (i.e. unwrapped) state."""
        raise NotImplementedError("This abstract property has to be implemented by a child.")

    @abstractproperty
    def visual_observation_space(self):
        """Returns the shape of the visual component of the observation space as a tuple."""
        raise NotImplementedError("This abstract property has to be implemented by a child.")

    @abstractproperty
    def vector_observation_space(self):
        """Returns the shape of the vector component of the observation space as a tuple."""
        raise NotImplementedError("This abstract property has to be implemented by a child.")

    @property
    def ground_truth_space(self):
        """Returns the space of the ground truth info space if available."""
        return None

    @abstractproperty
    def action_space(self):
        """Returns the shape of the action space of the agent."""
        raise NotImplementedError("This abstract property has to be implemented by a child.")

    @abstractproperty
    def max_episode_steps(self):
        """Returns the maximum number of steps that an episode can last."""
        raise NotImplementedError("This abstract property has to be implemented by a child.")

    @abstractproperty
    def seed(self):
        """Returns the seed of the current episode."""
        raise NotImplementedError("This abstract property has to be implemented by a child.")

    @abstractproperty
    def action_names(self):
        """Returns a list of action names. It has to be noted that only the names of action branches are provided and not the actions themselves!"""
        raise NotImplementedError("This abstract property has to be implemented by a child.")

    @abstractproperty
    def get_episode_trajectory(self):
        """Returns a dictionary containing a complete trajectory of data comprising an entire episode. This is only used for recording a video."""
        raise NotImplementedError("This abstract property has to be implemented by a child.")

    @abstractmethod
    def reset(self, reset_params = None):
        """Reset the environment. The provided config is a dictionary featuring reset parameters for the environment (e.g. seed)."""
        raise NotImplementedError("This abstract method has to be implemented by a child.")

    @abstractmethod
    def step(self, action):
        """Executes one step of the agent."""
        raise NotImplementedError("This abstract method has to be implemented by a child.")

    @abstractmethod
    def close(self):
        """Shuts down the environment."""
        raise NotImplementedError("This abstract method has to be implemented by a child.")