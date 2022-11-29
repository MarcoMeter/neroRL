import numpy as np
import torch

from neroRL.sampler.trajectory_sampler import TrajectorySampler

class TransformerSampler(TrajectorySampler):
    """The TrajectorySampler employs n environment workers to sample data for s worker steps regardless if an episode ended.
    Hence, the collected trajectories may contain multiple episodes or incomplete ones."""
    def __init__(self, configs, worker_id, visual_observation_space, vector_observation_space, action_space_shape, model, device) -> None:
        """Initializes the TrajectorSampler and launches its environment workers.

        Arguments:
            configs {dict} -- The whole set of configurations (e.g. training and environment configs)
            worker_id {int} -- Specifies the offset for the port to communicate with the environment, which is needed for Unity ML-Agents environments.
            visual_observation_space {box} -- Dimensions of the visual observation space (None if not available)
            vector_observation_space {tuple} -- Dimensions of the vector observation space (None if not available)
            action_space_shape {tuple} -- Dimensions of the action space
            model {nn.Module} -- The model to retrieve the policy and value from
            device {torch.device} -- The device that is used for retrieving the data from the model
        """
        super().__init__(configs, worker_id, visual_observation_space, vector_observation_space, action_space_shape, model, device)
        # Set member variables
        self.max_episode_length = configs["model"]["transformer"]["memory_length"]        # TODO
        self.num_mem_layers = configs["model"]["transformer"]["num_layers"]
        self.mem_layer_size = configs["model"]["transformer"]["layer_size"]

        self.buffer.init_transformer_buffer_fields(self.max_episode_length)

        # Setup memory placeholder
        self.memory = torch.zeros((self.n_workers, self.max_episode_length, self.num_mem_layers, self.mem_layer_size), dtype=torch.float32)
        # Generate episodic memory mask
        self.memory_mask = torch.tril(torch.ones((self.max_episode_length, self.max_episode_length)))
        # Shift mask by one to account for the fact that for the first timestep the memory is empty
        self.memory_mask = torch.cat((torch.zeros((1, self.max_episode_length)), self.memory_mask))[:-1]       
        # Worker ids
        self.worker_ids = range(self.n_workers)

    def sample(self, device) -> list:
        # Init memory buffer
        self.buffer.memories = [self.memory[w] for w in range(self.n_workers)]
        for w in range(self.n_workers):
            self.buffer.memory_index[w] = w
        return super().sample(device)

    def previous_model_input_to_buffer(self, t):
        super().previous_model_input_to_buffer(t)
        # Save mask
        self.buffer.memory_mask[:, t] = self.memory_mask[self.worker_current_episode_step]

    def forward_model(self, vis_obs, vec_obs, t):
        policy, value, memory, _ = self.model(vis_obs, vec_obs, memory = self.memory, mask = self.buffer.memory_mask[:, t])
        # Set memory 
        self.memory[self.worker_ids, self.worker_current_episode_step] = memory
        return policy, value

    def reset_worker(self, worker, id, t):
        super().reset_worker(worker, id, t)
        # Break the reference to the worker's memory
        mem_index = self.buffer.memory_index[id, t]
        self.buffer.memories[mem_index] = self.buffer.memories[mem_index].clone()
        # Reset episodic memory
        self.memory[id] = torch.zeros((self.max_episode_length, self.num_mem_layers, self.mem_layer_size), dtype=torch.float32)
        if t < self.worker_steps - 1:
            # Save memorie
            self.buffer.memories.append(self.memory[id])
            # Save the reference index to the current memory
            self.buffer.memory_index[id, t + 1:] = len(self.buffer.memories) - 1

    def get_last_value(self):
        _, last_value, _, _ = self.model(torch.tensor(self.vis_obs) if self.vis_obs is not None else None,
                                        torch.tensor(self.vec_obs) if self.vec_obs is not None else None,
                                        memory = self.memory, mask = self.memory_mask[self.worker_current_episode_step])
        return last_value
