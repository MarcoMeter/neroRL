import torch

from neroRL.sampler.trajectory_sampler import TrajectorySampler
from neroRL.utils.utils import batched_index_select, get_gpu_memory_map

class TransformerSampler(TrajectorySampler):
    """The TrajectorySampler employs n environment workers to sample data for s worker steps regardless if an episode ended.
    Hence, the collected trajectories may contain multiple episodes or incomplete ones. The TransformerSampler takes care of
    resetting and adding the agents' episodic memroies and memory masks to the buffer."""
    def __init__(self, configs, worker_id, visual_observation_space, vector_observation_space, action_space_shape, max_episode_steps, model, device) -> None:
        """Initializes the TrajectorSampler and launches its environment workers.

        Arguments:
            configs {dict} -- The whole set of configurations (e.g. training and environment configs)
            worker_id {int} -- Specifies the offset for the port to communicate with the environment, which is needed for Unity ML-Agents environments.
            visual_observation_space {box} -- Dimensions of the visual observation space (None if not available)
            vector_observation_space {tuple} -- Dimensions of the vector observation space (None if not available)
            action_space_shape {tuple} -- Dimensions of the action space
            max_episode_steps {int} -- Maximum number of steps one episode can last
            model {nn.Module} -- The model to retrieve the policy and value from
            device {torch.device} -- The device that is used for retrieving the data from the model
        """
        super().__init__(configs, worker_id, visual_observation_space, vector_observation_space, action_space_shape, model, device)
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
        self.memory = self.model.init_transformer_memory(self.n_workers, self.max_episode_steps, self.num_blocks, self.embed_dim, device)
        # Setup the memory mask that reflects the desired memory (i.e. context) length for the transformer architecture
        self.memory_mask = torch.tril(torch.ones((self.memory_length, self.memory_length)), diagonal=-1)
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
        self.critical_memory_usage, self.is_critical_memory = configs["sampler"]["critical_memory_usage"], False

    def sample(self, device) -> list:
        """Samples training data (i.e. experience tuples) using n workers for t worker steps. But before, the memory buffer is initialized."""
        self.buffer.memories = [self.memory[w] for w in range(self.n_workers)]
        for w in range(self.n_workers):
            self.buffer.memory_index[w] = w
        return super().sample(device)

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
        policy, value, memory, _ = self.model(vis_obs, vec_obs, memory = sliced_memory, mask = self.buffer.memory_mask[:, t],
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
        self.memory[id] = self.model.init_transformer_memory(1, self.max_episode_steps, self.num_blocks, self.embed_dim, self.device).squeeze(0)
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
        _, last_value, _, _ = self.model(torch.tensor(self.vis_obs) if self.vis_obs is not None else None,
                                        torch.tensor(self.vec_obs) if self.vec_obs is not None else None,
                                        memory = sliced_memory, mask = self.memory_mask[torch.clip(self.worker_current_episode_step, 0, self.memory_length - 1)],
                                        memory_indices = self.buffer.memory_indices[:,-1])
        return last_value
    
    def _check_for_memory_usage(self):
        """Checks if the memory usage is critical and if so, it reduces the used gpu memory by moving the necessary parts to the cpu."""
        # Check if the device is a gpu
        if self.device.type == "cpu":
            return
        
        # Get the relative free memory of the gpu
        rel_free_memory = get_gpu_memory_map()["rel_free"]
        
        # Check if the memory usage is critical
        if rel_free_memory < self.critical_memory_usage:
            print("Memory usage is critical. Reducing memory usage by moving the memory and transformer model to the cpu.")
            self.memory = self.memory.cpu()
            self.memory_mask = self.memory_mask.cpu()
            self.memory_indices = self.memory_indices.cpu()
            self.buffer.memory_mask = self.buffer.memory_mask.cpu()
            self.buffer.memory_indices = self.buffer.memory_indices.cpu()
            self.buffer.memories = [m.cpu() for m in self.buffer.memories]
            self.model.transformer = self.model.transformer.cpu()
            self.is_critical_memory = True
    
    def _reset_memory_usage(self):
        """Resets the memory usage by moving the necessary parts back to the gpu."""
        # Check if the memory usage was critical
        if self.is_critical_memory:
            # Move the memory and transformer model back to the gpu
            self.memory = self.memory.to(self.device)
            self.memory_mask = self.memory_mask.to(self.device)
            self.memory_indices = self.memory_indices.to(self.device)
            self.buffer.memory_mask = self.buffer.memory_mask.to(self.device)
            self.buffer.memory_indices = self.buffer.memory_indices.to(self.device)
            self.buffer.memories = [m.to(self.device) for m in self.buffer.memories]
            self.model.transformer = self.model.transformer.to(self.device)
        
