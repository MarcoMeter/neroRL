import torch

from neroRL.sampler.trajectory_sampler import TrajectorySampler
from neroRL.utils.utils import batched_index_select

class TransformerSampler(TrajectorySampler):
    """The TrajectorySampler employs n environment workers to sample data for s worker steps regardless if an episode ended.
    Hence, the collected trajectories may contain multiple episodes or incomplete ones. The TransformerSampler takes care of
    resetting and adding the agents' episodic memroies and memory masks to the buffer."""
    def __init__(self, configs, worker_id, visual_observation_space, vector_observation_space, ground_truth_space, action_space_shape, max_episode_steps, model, sample_device, train_device) -> None:
        """Initializes the TrajectorSampler and launches its environment workers.

        Arguments:
            configs {dict} -- The whole set of configurations (e.g. training and environment configs)
            worker_id {int} -- Specifies the offset for the port to communicate with the environment, which is needed for Unity ML-Agents environments.
            visual_observation_space {box} -- Dimensions of the visual observation space (None if not available)
            vector_observation_space {tuple} -- Dimensions of the vector observation space (None if not available)
            ground_truth_space {box} -- Dimensions of the ground truth space (None if not available)
            action_space_shape {tuple} -- Dimensions of the action space
            max_episode_steps {int} -- Maximum number of steps one episode can last
            model {nn.Module} -- The model to retrieve the policy and value from
            device {torch.device} -- The device that is used for retrieving the data from the model
        """
        super().__init__(configs, worker_id, visual_observation_space, vector_observation_space, ground_truth_space, action_space_shape, model, sample_device, train_device)
        # Set member variables
        self.max_episode_steps = max_episode_steps
        self.memory_length = configs["model"]["transformer"]["memory_length"]
        self.num_blocks = configs["model"]["transformer"]["num_blocks"]
        self.embed_dim = configs["model"]["transformer"]["embed_dim"]

        # The buffer has to store multiple fields for the experience tuples:
        # - Entire episode memories (episodes can be as long as max_episode_steps)
        #       - The number of episodes may vary so the outer dimension is based on a list
        # - Index for each time step to point to the correct episode memory
        # - Memory mask for each time step
        # - Memory window indices for each time step (window is as long as memory_length)
        self.buffer.init_transformer_buffer_fields(self.max_episode_steps)

        # Setup memory placeholder
        # It is designed to store an entire episode of past latent features (i.e. activations of the encoder and the transformer layers).
        # Initialization using zeros
        # Once an episode is completed (max episode steps reached or environment termination), it is added to the buffer.
        self.memory = self.model.init_transformer_memory(self.n_workers, self.max_episode_steps, self.num_blocks, self.embed_dim)
        # Setup the memory mask that reflects the desired memory (i.e. context) length for the transformer architecture
        self.memory_mask = torch.tril(torch.ones((self.memory_length, self.memory_length)))#, diagonal=-1)
        """ e.g. memory mask tensor looks like this if memory_length = 6
        0, 0, 0, 0, 0, 0
        1, 0, 0, 0, 0, 0
        1, 1, 0, 0, 0, 0
        1, 1, 1, 0, 0, 0
        1, 1, 1, 1, 0, 0
        1, 1, 1, 1, 1, 0
        """
        # Worker ids
        self.worker_ids = range(self.n_workers)
        # Setup memory window indices
        repetitions = torch.repeat_interleave(torch.arange(0, self.memory_length).unsqueeze(0), self.memory_length - 1, dim = 0).long()
        self.memory_indices = torch.stack([torch.arange(i, i + self.memory_length) for i in range(max_episode_steps - self.memory_length + 1)]).long()
        self.memory_indices = torch.cat((repetitions, self.memory_indices))
        """e.g. the memory window indices tensor looks like this if memory_length = 4 and max_episode_steps = 7:
        0, 1, 2, 3
        0, 1, 2, 3
        0, 1, 2, 3
        0, 1, 2, 3
        1, 2, 3, 4
        2, 3, 4, 5
        3, 4, 5, 6
        """

    def sample(self) -> list:
        """Samples training data (i.e. experience tuples) using n workers for t worker steps. But before, the memory buffer is initialized."""
        self.buffer.memories = [self.memory[w] for w in range(self.n_workers)]
        for w in range(self.n_workers):
            self.buffer.memory_index[w] = w
        return super().sample()

    def previous_model_input_to_buffer(self, t):
        """Add the model's previous input, the memory mask and the memory window incdices to the buffer."""
        super().previous_model_input_to_buffer(t)
        self.buffer.memory_mask[:, t] = self.memory_mask[torch.clip(self.worker_current_episode_step, 0, self.memory_length - 1)]
        self.buffer.memory_indices[:,t] = self.memory_indices[self.worker_current_episode_step]

    def forward_model(self, vis_obs, vec_obs, t):
        """Forwards the model to retrieve the policy and the value of the to be fed observations and memory window."""
        # Retrieve the memory window from the entire episode
        sliced_memory = batched_index_select(self.memory, 1, self.buffer.memory_indices[:,t])
        # Forward
        policy, value, memory = self.model(vis_obs, vec_obs, memory = sliced_memory, mask = self.buffer.memory_mask[:, t],
                                                memory_indices = self.buffer.memory_indices[:,t])
        # Write the new memory item to the placeholder
        self.memory[self.worker_ids, self.worker_current_episode_step] = memory
        return policy, value

    def reset_worker(self, worker, id, t):
        """Resets the specified worker and resets the agent's episodic memory."""
        super().reset_worker(worker, id, t)
        # Break the reference to the worker's memory
        mem_index = self.buffer.memory_index[id, t]
        self.buffer.memories[mem_index] = self.buffer.memories[mem_index].clone()
        # Reset episodic memory
        self.memory[id] = self.model.init_transformer_memory(1, self.max_episode_steps, self.num_blocks, self.embed_dim).squeeze(0)
        if t < self.worker_steps - 1:
            # Save memory
            self.buffer.memories.append(self.memory[id])
            # Save the reference index to the current memory
            self.buffer.memory_index[id, t + 1:] = len(self.buffer.memories) - 1

    def get_last_value(self):
        """Returns the last value of the current observation and memory window to compute GAE."""
        start = torch.clip(self.worker_current_episode_step - self.memory_length, 0)
        end = torch.clip(self.worker_current_episode_step, self.memory_length)
        indices = torch.stack([torch.arange(start[b],end[b]) for b in range(self.n_workers)]).long()
        sliced_memory = batched_index_select(self.memory, 1, indices) # Retrieve the memory window from the entire episode
        _, last_value, _ = self.model(torch.tensor(self.vis_obs) if self.vis_obs is not None else None,
                                        torch.tensor(self.vec_obs) if self.vec_obs is not None else None,
                                        memory = sliced_memory, mask = self.memory_mask[torch.clip(self.worker_current_episode_step, 0, self.memory_length - 1)],
                                        memory_indices = self.buffer.memory_indices[:,-1])
        return last_value
