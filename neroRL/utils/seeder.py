import numpy as np
from scipy.special import softmax

class Seeder:
    """
    The Seeder tracks the performance across all training seeds.
    It is in charge of providing new seeds.
    Seeds, where the agent performs worse, are more likely to be sampled from a multinomial distribution.
    """
    def __init__(self, num_seeds = 100, episode_window = 10, step_size = 0.1, random_seed = False):
        self.num_seeds = num_seeds
        self.episode_window = episode_window
        self.step_size = step_size
        self.seed_logits = softmax(np.ones(num_seeds, dtype = np.float32))
        self.seed_stats = np.ones((num_seeds, episode_window), dtype = np.int32)
        self.random_seed = random_seed

    def add_seed_result(self, seed, reached_floor):
        """Adds a seed's result to the statistics and the updates the logits."""
        # Roll values (mean floor of up to episode_window episodes) of one particular seed
        seed_history = np.roll(self.seed_stats[seed], shift = 1, axis = 0)
        # Overwrite the oldest value with the newest one
        seed_history[0] = reached_floor
        # Apply new statistics
        self.seed_stats[seed] = seed_history
        
    def update_logits(self):
        """Updates the logits, which are used for a multinomial distribution."""
        # Compute desired logits (normalization is done with softmax function)
        median = np.median(self.seed_stats, axis = 1)
        self.seed_logits = softmax(1 - (median / np.max(median))**2)

    def lerp(self, x, y, s):
        """Linearly interpolates between two vectors at strength s."""
        return x * (1 - s) + y * s

    def sample_seed(self):
        """Samples a new training seed from a multinomial distribution."""
        if self.random_seed:
            return np.random.randint(0, 100)

        one_hot_sample = np.random.multinomial(1, self.seed_logits)
        return np.argmax(one_hot_sample)
