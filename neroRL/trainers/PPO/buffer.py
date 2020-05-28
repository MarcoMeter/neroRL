import torch
import numpy as np
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

class Buffer():
    """The buffer stores and prepares the training data. It supports recurrent policies.
    """
    def __init__(self, n_workers, worker_steps, n_mini_batch, visual_observation_space, vector_observation_space, action_space_shape, use_recurrent, hidden_state_size, device, mini_batch_device):
        """
        Arguments:
            n_workers {int} -- Number of environments/agents to sample training data
            worker_steps {int} -- Number of steps per environment/agent to sample training data
            n_mini_batch {int} -- Number of mini batches that are used for each training epoch
            visual_observation_space {Box} -- Visual observation if available, else None
            vector_observation_space {tuple} -- Vector observation space if available, else None
            action_space_shape {tuple} -- Shape of the action space
            use_recurrent {bool} -- Whether to use a recurrent model
            hidden_state_size {int} -- Size of the GRU layer (short-term memory)
            device {torch.device} -- The device that will be used for training/storing single mini batches
            mini_batch_device {torch.device} -- The device that will be used for storing the whole batch of data. This should be CPU if not enough VRAM is available.
        """
        self.device = device
        self.use_recurrent = use_recurrent
        self.n_workers = n_workers
        self.worker_steps = worker_steps
        self.n_mini_batch = n_mini_batch
        self.batch_size = self.n_workers * self.worker_steps
        self.mini_batch_size = self.batch_size // self.n_mini_batch
        self.mini_batch_device = mini_batch_device
        self.rewards = np.zeros((n_workers, worker_steps), dtype=np.float32)
        self.actions = np.zeros((n_workers, worker_steps, len(action_space_shape)), dtype=np.int32)
        self.dones = np.zeros((n_workers, worker_steps), dtype=np.bool)
        if visual_observation_space is not None:
            self.vis_obs = np.zeros((n_workers, worker_steps) + visual_observation_space.shape, dtype=np.float32)
        else:
            self.vis_obs = None
        if vector_observation_space is not None:
            self.vec_obs = np.zeros((n_workers, worker_steps,) + vector_observation_space, dtype=np.float32)
        else:
            self.vec_obs = None
        self.hidden_states = torch.zeros((n_workers, worker_steps, hidden_state_size), dtype=torch.float32)
        self.neg_log_pis = np.zeros((n_workers, worker_steps, len(action_space_shape)), dtype=np.float32)
        self.values = np.zeros((n_workers, worker_steps), dtype=np.float32)
        self.advantages = np.zeros((n_workers, worker_steps), dtype=np.float32)

    def calc_advantages(self, last_value, gamma, lamda):
        """Generalized advantage estimation (GAE)

        Arguments:
            last_value {numpy.ndarray} -- Value of the last agent's state
            gamma {float} -- Discount factor
            lamda {float} -- GAE regularization parameter
        """
        last_advantage = 0
        for t in reversed(range(self.worker_steps)):
            mask = 1.0 - self.dones[:, t] # mask value on a terminal state (i.e. done)
            last_value = last_value * mask
            last_advantage = last_advantage * mask
            delta = self.rewards[:, t] + gamma * last_value - self.values[:, t]
            last_advantage = delta + gamma * lamda * last_advantage
            self.advantages[:, t] = last_advantage
            last_value = self.values[:, t]

    def prepare_batch_dict(self):
        """Flattens the training samples and stores them inside a dictionary."""
        # Store unflattened samples
        samples = {
            'actions': self.actions,
            'values': self.values,
            'neg_log_pis': self.neg_log_pis,
            'advantages': self.advantages,
            'hidden_states': self.hidden_states,
            'dones': self.dones
        }

    	# Add observations to dictionary
        if self.vis_obs is not None:
            samples['vis_obs'] = self.vis_obs
        if self.vec_obs is not None:
            samples['vec_obs'] = self.vec_obs

        # Flatten all samples
        self.samples_flat = {}
        for key, value in samples.items():
            value = value.reshape(value.shape[0] * value.shape[1], *value.shape[2:])
            self.samples_flat[key] = torch.tensor(value, dtype = torch.float32, device = self.mini_batch_device)

    def mini_batch_generator(self):
        """A generator that returns a dictionary containing the data of a whole minibatch.
        This mini batch is completely shuffled.

        Yields:
            {dict} -- Mini batch data for training
        """
        # Prepare indices (shuffle)
        indices = torch.randperm(self.batch_size)
        for start in range(0, self.batch_size, self.mini_batch_size):
            # Arrange mini batches
            end = start + self.mini_batch_size
            mini_batch_indices = indices[start: end]
            mini_batch = {}
            for key, value in self.samples_flat.items():
                mini_batch[key] = value[mini_batch_indices].to(self.device)
            yield mini_batch

    def recurrent_mini_batch_generator(self):
        """A recurrent generator that returns a dictionary containing the data of a whole minibatch.
        In comparison to the none-recurrent one, this generator maintains the sequences of the workers' experience trajectories.

        Yields:
            {dict} -- Mini batch data for training
        """
        # Prepare indices, but only shuffle the worker indices and not the entire batch to ensure that sequences of worker trajectories are maintained
        num_envs_per_batch = self.n_workers // self.n_mini_batch
        indices = np.arange(0, self.n_workers * self.worker_steps).reshape(self.n_workers, self.worker_steps)
        worker_indices = torch.randperm(self.n_workers)
        for start in range(0, self.n_workers, num_envs_per_batch):
            # Arrange mini batches
            end = start + num_envs_per_batch
            mini_batch_indices = indices[worker_indices[start:end]].reshape(-1)
            mini_batch = {}
            for key, value in self.samples_flat.items():
                mini_batch[key] = value[mini_batch_indices].to(self.device)
            yield mini_batch