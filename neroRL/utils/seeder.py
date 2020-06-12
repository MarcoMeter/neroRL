import numpy as np
from scipy.special import softmax

class Seeder:
    """
    The Seeder tracks the performance across all training seeds.
    It is in charge of providing new seeds.
    Seeds, where the agent performs worse, are more likely to be sampled from a multinomial distribution.
    """
    def __init__(self, start_seed = 0, num_seeds = 100, episode_window = 10, step_size = 0.1, random_seed = False):
        """[summary]

        Args:
            start_seed {int}: First training seed of the range of training seeds.
            num_seeds {int}: Number of available training seeds.
            episode_window {int}: Number of episodes to track the performances for each seed. Defaults to 10.
            step_size {float}: Update rate of the logits. Defaults to 0.1.
            random_seed {bool}: Whether to sample seeds uniformly. Defaults to False.
        """
        self.num_seeds = num_seeds
        self.episode_window = episode_window
        self.step_size = step_size
        # Init logits
        self.seed_logits = softmax(np.ones(num_seeds, dtype = np.float32))
        # Performance history of all available training seeds
        self.seed_performances = np.ones((num_seeds, episode_window), dtype = np.int32)
        self.random_seed = random_seed

    def add_seed_result(self, seed, performance):
        """Adds the performance of one episode (played on a distinct seed) to the statistics.

        Args:
            seed {int}: The seed of the played episode
            performance {float}: Performance metric of the played episode (e.g. received reward or reached floor)
        """
        # Roll values (mean floor of up to episode_window episodes) of one particular seed
        seed_history = np.roll(self.seed_performances[seed], shift = 1, axis = 0)
        # Overwrite the oldest value with the newest one
        seed_history[0] = performance
        # Apply new statistics
        self.seed_performances[seed] = seed_history
        
    def update_logits(self):
        """Updates the logits, which are used for a multinomial distribution."""
        # Compute desired logits (normalization is done with softmax function)
        median = np.median(self.seed_performances, axis = 1)
        self.seed_logits = softmax(1 - (median / np.max(median))**2)

    def lerp(self, x, y, s):
        """Linearly interpolates between two vectors at strength s."""
        return x * (1 - s) + y * s

    def sample_seed(self):
        """Samples a new training seed from a multinomial distribution.

        Returns:
            {int}: Sampled seed
        """
        if self.random_seed:
            return np.random.randint(0, 100)

        one_hot_sample = np.random.multinomial(1, self.seed_logits)
        return int(np.argmax(one_hot_sample))
