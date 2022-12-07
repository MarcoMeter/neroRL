import torch
import numpy as np

class Buffer():
    """The buffer stores and prepares the training data. It supports recurrent and transformer policies."""
    def __init__(self, configs, visual_observation_space, vector_observation_space,
                    action_space_shape, device, share_parameters, sampler):
        """
        Arguments:
            configs {dict} -- The whole set of configurations (e.g. model, training, environment, ... configs)
            visual_observation_space {Box} -- Visual observation if available, else None
            vector_observation_space {tuple} -- Vector observation space if available, else None
            action_space_shape {tuple} -- Shape of the action space
            device {torch.device} -- The device that will be used for training/storing single mini batches
            share_parameters {bool} -- Whether the policy and the value function share parameters or not
            sampler {TrajectorySampler} -- The used sampler
        """
        self.device = device
        self.sampler = sampler
        self.configs = configs
        self.recurrence, self.transformer_memory = None, None       # place holder for recurrence and transformer config
        self.num_workers = configs["sampler"]["n_workers"]
        self.worker_steps = configs["sampler"]["worker_steps"]
        self.batch_size = self.num_workers * self.worker_steps
        self.action_space_shape = action_space_shape
        self.visual_observation_space = visual_observation_space
        self.vector_observation_space = vector_observation_space
        self.share_parameters = share_parameters
        self.init_default_buffer_fields()

    def init_default_buffer_fields(self):
        """Initializes buffer fields that every training run depends on."""
        if self.visual_observation_space is not None:
            self.vis_obs = torch.zeros((self.num_workers, self.worker_steps) + self.visual_observation_space.shape)
        else:
            self.vis_obs = None
        if self.vector_observation_space is not None:
            self.vec_obs = torch.zeros((self.num_workers, self.worker_steps,) + self.vector_observation_space)
        else:
            self.vec_obs = None
        self.rewards = np.zeros((self.num_workers, self.worker_steps), dtype=np.float32)
        self.actions = torch.zeros((self.num_workers, self.worker_steps, len(self.action_space_shape)), dtype=torch.long)
        self.dones = np.zeros((self.num_workers, self.worker_steps), dtype=np.bool)
        self.log_probs = torch.zeros((self.num_workers, self.worker_steps, len(self.action_space_shape)))
        self.values = torch.zeros((self.num_workers, self.worker_steps))
        self.advantages = torch.zeros((self.num_workers, self.worker_steps))

    def init_recurrent_buffer_fields(self):
        """Initializes the buffer fields and members that are needed for training recurrent policies."""
        self.num_sequences = 0
        self.actual_sequence_length = 0
        self.recurrence = self.configs["model"]["recurrence"]
        self.sequence_length = self.recurrence["sequence_length"]
        num_layers = self.recurrence["num_layers"]
        hidden_state_size = self.recurrence["hidden_state_size"]
        layer_type = self.recurrence["layer_type"]
        if self.share_parameters:
            self.hxs = torch.zeros((self.num_workers, self.worker_steps, num_layers, hidden_state_size))
            if layer_type == "lstm":
                self.cxs = torch.zeros((self.num_workers, self.worker_steps, num_layers, hidden_state_size))
        else: # if parameters are not shared then add two extra dimensions for adding enough capacity to store the hidden states of the actor and critic model
            self.hxs = torch.zeros((self.num_workers, self.worker_steps, num_layers, hidden_state_size, 2))
            if layer_type == "lstm":
                self.cxs = torch.zeros((self.num_workers, self.worker_steps, num_layers, hidden_state_size, 2))

    def init_transformer_buffer_fields(self, max_episode_length):
        """Initializes the buffer fields and members that are needed for training transformer-based policies."""
        self.max_episode_length = max_episode_length
        self.transformer_memory = self.configs["model"]["transformer"]
        self.num_mem_layers = self.transformer_memory["num_layers"]
        self.mem_layer_size = self.transformer_memory["layer_size"]
        # Episodic memory index buffer
        self.memories = []
        self.memory_mask = torch.zeros((self.num_workers, self.worker_steps, self.max_episode_length), dtype=torch.bool)
        self.memory_index = torch.zeros((self.num_workers, self.worker_steps), dtype=torch.long)

    def calc_advantages(self, last_value, gamma, lamda):
        """Generalized advantage estimation (GAE)

        Arguments:
            last_value {numpy.ndarray} -- Value of the last agent's state
            gamma {float} -- Discount factor
            lamda {float} -- GAE regularization parameter
        """
        with torch.no_grad():
            last_advantage = 0
            mask = torch.tensor(self.dones).logical_not() # mask values on terminal states
            rewards = torch.tensor(self.rewards)
            for t in reversed(range(self.worker_steps)):
                last_value = last_value * mask[:, t]
                last_advantage = last_advantage * mask[:, t]
                delta = rewards[:, t] + gamma * last_value - self.values[:, t]
                last_advantage = delta + gamma * lamda * last_advantage
                self.advantages[:, t] = last_advantage
                last_value = self.values[:, t]

    def prepare_batch_dict(self):
        """
        Flattens the training samples and stores them inside a dictionary.
        If a recurrent policy is used, the data is split into episodes or sequences beforehand.
        """
        # Supply training samples
        samples = {
            "actions": self.actions,
            "values": self.values,
            "log_probs": self.log_probs,
            "advantages": self.advantages,
            # The loss mask is used for masking the padding while computing the loss function.
            # This is only of significance while using recurrence.
            "loss_mask": torch.ones((self.num_workers, self.worker_steps), dtype=torch.bool)
        }

    	# Add available observations to the dictionary
        if self.vis_obs is not None:
            samples["vis_obs"] = self.vis_obs
        if self.vec_obs is not None:
            samples["vec_obs"] = self.vec_obs

        # Add data concerned with the episodic memory (i.e. transformer-based policy)
        if self.transformer_memory is not None:
            samples["memory_index"] = self.memory_index
            samples["memory_mask"] = self.memory_mask
            # Convert the memories to a tensor
            self.memories = torch.stack(self.memories, dim=0)

        # Add data concerned with the memory based on recurrence and arrange the entire training data into sequences
        max_sequence_length = 1
        if self.recurrence is not None:
            # Add collected recurrent cell states to the dictionary
            samples["hxs"] =  self.hxs
            if self.recurrence["layer_type"] == "lstm":
                samples["cxs"] = self.cxs

            # Split data into sequences and apply zero-padding
            # Retrieve the indices of dones as these are the last step of a whole episode
            episode_done_indices = []
            for w in range(self.num_workers):
                episode_done_indices.append(list(self.dones[w].nonzero()[0]))
                # Append the index of the last element of a trajectory as well, as it "artifically" marks the end of an episode
                if len(episode_done_indices[w]) == 0 or episode_done_indices[w][-1] != self.worker_steps - 1:
                    episode_done_indices[w].append(self.worker_steps - 1)
            
            # Split vis_obs, vec_obs, recurrent cell states and actions into episodes and then into sequences
            for key, value in samples.items():
                # Don't process (i.e. don't pad) log_probs, advantages and values
                if not (key == "log_probs" or key == "advantages" or key == "values"):
                    sequences = []
                    flat_sequence_indices = []  # we collect the indices of every unpadded sequence to correctly sample unpadded data
                    for w in range(self.num_workers):
                        start_index = 0
                        for done_index in episode_done_indices[w]:
                            # Split trajectory into episodes
                            episode = value[w, start_index:done_index + 1]
                            # Split episodes into sequences
                            if self.sequence_length > 0:
                                for start in range(0, len(episode), self.sequence_length):
                                    end = start + self.sequence_length
                                    seq = episode[start:end]
                                    sequences.append(seq)
                                    flat_start = start + w * self.worker_steps + start_index
                                    flat_sequence_indices.append(list(range(flat_start, flat_start + len(seq))))
                                max_sequence_length = self.sequence_length
                            else:
                                # If the sequence length is not set to a proper value, sequences will be based on episodes
                                sequences.append(episode)
                                max_sequence_length = len(episode) if len(episode) > max_sequence_length else max_sequence_length
                            start_index = done_index + 1
                    
                    # Apply zero-padding to ensure that each episode has the same length
                    # Therfore we can train batches of episodes in parallel instead of one episode at a time
                    for i, sequence in enumerate(sequences):
                        sequences[i] = self._pad_sequence(sequence, max_sequence_length)

                    # Stack sequences (target shape: (Sequence, Step, Data ...) & apply data to the samples dict
                    samples[key] = torch.stack(sequences, axis=0)

                    if (key == "hxs" or key == "cxs"):
                        # Select the very first recurrent cell state of a sequence and add it to the samples
                        samples[key] = samples[key][:, 0]

        # Store important information
        self.num_sequences = len(samples["actions"])
        self.actual_sequence_length = max_sequence_length
        self.flat_sequence_indices = np.asarray(flat_sequence_indices, dtype=object)
        
        # Flatten samples
        self.samples_flat = {}
        for key, value in samples.items():
            if not key == "hxs" and not key == "cxs":
                value = value.reshape(value.shape[0] * value.shape[1], *value.shape[2:])
            self.samples_flat[key] = value        

    def _pad_sequence(self, sequence, target_length):
        """Pads a sequence to the target length using zeros.

        Argumentss:
            sequence {numpy.ndarray} -- The to be padded array (i.e. sequence)
            target_length {int} -- The desired length of the sequence

        Returns:
            {numpy.ndarray} -- Returns the padded sequence
        """
        # Determine the number of zeros that have to be added to the sequence
        delta_length = target_length - len(sequence)
        # If the sequence is already as long as the target length, don't pad
        if delta_length <= 0:
            return sequence
        # Construct array of zeros
        if len(sequence.shape) > 1:
            # Case: pad multi-dimensional array like visual observation
            padding = torch.zeros(((delta_length,) + sequence.shape[1:]), dtype=sequence.dtype)
            # padding = torch.full(((delta_length,) + sequence.shape[1:]), sequence[0], dtype=sequence.dtype) # experimental
        else:
            padding = torch.zeros(delta_length, dtype=sequence.dtype)
            # padding = torch.full(delta_length, sequence[0], dtype=sequence.dtype) # experimental
        # Concatenate the zeros to the sequence
        return torch.cat((sequence, padding), axis=0)

    def mini_batch_generator(self, num_mini_batches):
        """A generator that returns a dictionary containing the data of a whole minibatch.
        This mini batch is completely shuffled.

        Arguments:
            num_mini_batches {int} -- Number of the to be sampled mini batches

        Yields:
            {dict} -- Mini batch data for training
        """
        # Prepare indices (shuffle)
        indices = torch.randperm(self.batch_size)
        mini_batch_size = self.batch_size // num_mini_batches
        for start in range(0, self.batch_size, mini_batch_size):
            # Compose mini batches
            end = start + mini_batch_size
            mini_batch_indices = indices[start: end]
            mini_batch = {}
            for key, value in self.samples_flat.items():
                if key == "memory_index":
                    mini_batch["memories"] = self.memories[value[mini_batch_indices]]
                else:
                    mini_batch[key] = value[mini_batch_indices].to(self.device)
            yield mini_batch

    def recurrent_mini_batch_generator(self, num_mini_batches):
        """A recurrent generator that returns a dictionary containing the data of a whole minibatch.
        In comparison to the none-recurrent one, this generator maintains the sequences of the workers' experience trajectories.

        Arguments:
            num_mini_batches {int} -- Number of the to be sampled mini batches

        Yields:
            {dict} -- Mini batch data for training
        """
        # Determine the number of sequences per mini batch
        num_sequences_per_batch = self.num_sequences // num_mini_batches
        num_sequences_per_batch = [num_sequences_per_batch] * num_mini_batches # Arrange a list that determines the sequence count for each mini batch
        remainder = self.num_sequences % num_mini_batches
        for i in range(remainder):
            num_sequences_per_batch[i] += 1 # Add the remainder if the sequence count and the number of mini batches do not share a common divider
        # Prepare indices, but only shuffle the sequence indices and not the entire batch to ensure that sequences are maintained as a whole.
        indices = torch.arange(0, self.num_sequences * self.actual_sequence_length).reshape(self.num_sequences, self.actual_sequence_length)
        sequence_indices = torch.randperm(self.num_sequences)

        # Compose mini batches
        start = 0
        for num_sequences in num_sequences_per_batch:
            end = start + num_sequences
            mini_batch_padded_indices = indices[sequence_indices[start:end]].reshape(-1)
            # Unpadded and flat indices are used to sample unpadded training data
            mini_batch_unpadded_indices = self.flat_sequence_indices[sequence_indices[start:end].tolist()]
            mini_batch_unpadded_indices = [item for sublist in mini_batch_unpadded_indices for item in sublist]
            mini_batch = {}
            for key, value in self.samples_flat.items():
                if key == "hxs" or key == "cxs":
                    # Select recurrent cell states of sequence starts
                    mini_batch[key] = value[sequence_indices[start:end]].to(self.device)
                elif key == "log_probs" or key == "advantages" or key == "values":
                    # Select unpadded data
                    mini_batch[key] = value[mini_batch_unpadded_indices].to(self.device)
                else:
                    # Select padded data
                    mini_batch[key] = value[mini_batch_padded_indices].to(self.device)
            start = end
            yield mini_batch

    def refresh(self, model, gamma, lamda):
        """Refreshes the buffer with the current model.

        Arguments:
            model {nn.Module} -- The model to retrieve the policy and value from
            gamma {float} -- Discount factor
            lamda {float} -- GAE regularization parameter
        """
        # Init recurrent cells
        recurrent_cell = None
        if self.recurrence is not None:
            if self.recurrence["layer_type"] == "gru":
                recurrent_cell = self.hxs[:, 0].unsqueeze(0).contiguous()
            elif self.recurrence["layer_type"] == "lstm":
                recurrent_cell = (self.hxs[:, 0].unsqueeze(0).contiguous(), self.cxs[:, 0].unsqueeze(0).contiguous())

        # Refresh values and hidden_states with current model
        for t in range(self.worker_steps):
            # Gradients can be omitted for refreshing buffer
            with torch.no_grad():
                # Refresh hidden states
                if self.recurrence is not None:
                    if self.recurrence["layer_type"] == "gru":
                        self.hxs[:, t] = recurrent_cell.squeeze(0)
                    elif self.recurrence["layer_type"] == "lstm":
                        self.hxs[:, t] = recurrent_cell[0].squeeze(0)
                        self.cxs[:, t] = recurrent_cell[1].squeeze(0)

                    # Forward the model to retrieve the policy (making decisions), 
                    # the states' value of the value function and the recurrent hidden states (if available)
                    vis_obs = self.vis_obs[:, t] if self.vis_obs is not None else None
                    vec_obs = self.vec_obs[:, t] if self.vec_obs is not None else None
                    policy, value, recurrent_cell, _ = model(vis_obs, vec_obs, recurrent_cell)
                    # Refresh values
                    self.values[:, t] = value
                    
                # Reset hidden states if necessary
                for w in range(self.num_workers):
                    if self.recurrence is not None and self.dones[w, t]:
                        if self.recurrence["reset_hidden_state"]:
                            hxs, cxs = model.init_recurrent_cell_states(1, self.device)
                            if self.recurrence["layer_type"] == "gru":
                                recurrent_cell[:, w] = hxs.contiguous()
                            elif self.recurrence["layer_type"] == "lstm":
                                recurrent_cell[0][:, w] = hxs.contiguous()
                                recurrent_cell[1][:, w] = cxs.contiguous()

        # Refresh advantages
        _, last_value, _, _ = model(self.sampler.last_vis_obs(), self.sampler.last_vec_obs(), self.sampler.last_recurrent_cell())
        self.calc_advantages(last_value, gamma, lamda)
        
        # Refresh batches
        self.prepare_batch_dict() 