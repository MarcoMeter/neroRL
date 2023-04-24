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

    def init_transformer_buffer_fields(self, max_episode_steps):
        """Initializes the buffer fields and members that are needed for training transformer-based policies."""
        self.max_episode_steps = max_episode_steps
        self.transformer_memory = self.configs["model"]["transformer"]
        self.memory_length = self.transformer_memory["memory_length"]
        self.num_mem_layers = self.transformer_memory["num_blocks"]
        self.mem_layer_size = self.transformer_memory["embed_dim"]
        # Episodic memory index buffer
        # Whole episode memories
        self.memories = []
        # Memory mask used during attention
        self.memory_mask = torch.zeros((self.num_workers, self.worker_steps, self.memory_length), dtype=torch.bool)
        # Index to select the correct episode memory
        self.memory_index = torch.zeros((self.num_workers, self.worker_steps), dtype=torch.long)
        # Indices to slice the memory window
        self.memory_indices = torch.zeros((self.num_workers, self.worker_steps, self.memory_length), dtype=torch.long)

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
        If a recurrent policy is used, the model's input data and actions are split into episodes or sequences beforehand.
        """
        # Supply training samples
        samples = {"actions": self.actions} # actions are added at this point to ensure that these are padded while using recurrence

        # OBSERVATION SAMPLES
    	# Add available observations to the dictionary
        if self.vis_obs is not None:
            samples["vis_obs"] = self.vis_obs
        if self.vec_obs is not None:
            samples["vec_obs"] = self.vec_obs

        # TRANSFORMER SAMPLES
        # Add data concerned with the episodic memory (i.e. transformer-based policy)
        if self.transformer_memory is not None:
            samples["memory_index"] = self.memory_index
            samples["memory_mask"] = self.memory_mask
            samples["memory_indices"] = self.memory_indices
            # Convert the memories to a tensor
            self.memories = torch.stack(self.memories, dim=0)

        # RECURRENCE SAMPLES
        # Add data concerned with the memory based on recurrence and arrange the entire training data into sequences
        max_sequence_length = 1
        if self.recurrence is not None:
            # The loss mask is used for masking the padding while computing the loss function.
            samples["loss_mask"] = torch.ones((self.num_workers, self.worker_steps), dtype=torch.bool)

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

            # Retrieve unpadded sequence indices
            self.flat_sequence_indices = np.asarray(self._arange_sequences(
                        np.arange(0, self.num_workers * self.worker_steps).reshape(
                            (self.num_workers, self.worker_steps)), episode_done_indices)[0], dtype=object)
            
            # Split vis_obs, vec_obs, recurrent cell states and actions into episodes and then into sequences
            for key, value in samples.items():
                # Split data into episodes or sequences
                sequences, max_sequence_length = self._arange_sequences(value, episode_done_indices)

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
            self.num_sequences = len(sequences)
            
        self.actual_sequence_length = max_sequence_length
        
        # Add remaining data samples
        samples["values"] = self.values
        samples["log_probs"] = self.log_probs
        samples["advantages"] = self.advantages

        # Flatten samples
        self.samples_flat = {}
        for key, value in samples.items():
            if not key == "hxs" and not key == "cxs":
                value = value.reshape(value.shape[0] * value.shape[1], *value.shape[2:])
            self.samples_flat[key] = value

    def _arange_sequences(self, data, episode_done_indices):
        """Splits the povided data into episodes and then into sequences.
        The split points are indicated by the envrinoments' done signals.

        Arguments:
            data {torch.tensor} -- The to be split data arrange into num_worker, worker_steps
            episode_done_indices {list} -- Nested list indicating the indices of done signals. Trajectory ends are treated as done
            max_length {int} -- The maximum length of all sequences

        Returns:
            {list} -- Data arranged into sequences of variable length as list
        """
        sequences = []
        max_length = 1
        for w in range(self.num_workers):
            start_index = 0
            for done_index in episode_done_indices[w]:
                # Split trajectory into episodes
                episode = data[w, start_index:done_index + 1]
                # Split episodes into sequences
                if self.sequence_length > 0:
                    for start in range(0, len(episode), self.sequence_length):
                        end = start + self.sequence_length
                        sequences.append(episode[start:end])
                    max_length = self.sequence_length
                else:
                    # If the sequence length is not set to a proper value, sequences will be based on episodes
                    sequences.append(episode)
                    max_length = len(episode) if len(episode) > max_length else max_length
                start_index = done_index + 1
        return sequences, max_length

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
                # https://pytorch.org/docs/stable/notes/faq.html#my-out-of-memory-exception-handler-can-t-allocate-memory
                if key == "memory_index":
                    oom = False
                    try:
                        mini_batch["memories"] = self.memories[value[mini_batch_indices]]
                    except RuntimeError: # Out of memory
                        oom = True
                    if oom:
                        mini_batch = self._reduce_memory_usage(mini_batch)
                        mini_batch["memories"] = self.memories[value[mini_batch_indices]].to(self.device)
                        
                elif key == "memory_indices" or key == "memory_mask":
                    oom = False
                    try:
                        mini_batch[key] = value[mini_batch_indices]
                    except RuntimeError: # Out of memory
                        oom = True
                    if oom:
                        mini_batch = self._reduce_memory_usage(mini_batch)
                        mini_batch[key] = value[mini_batch_indices].to(self.device)
                else:
                    mini_batch[key] = value[mini_batch_indices].to(self.device)
            yield mini_batch
            
    def _reduce_memory_usage(self, mini_batch):
        """Reduces the used gpu memory by moving the necessary parts to the cpu.
        
            Arguments:
                mini_batch {dict} -- The mini batch that is currently being processed
            
            Returns:
                {dict} -- The mini batch with the memories and the memory indices moved to the cpu
        """
        # Check if the device is on cpu or if the memory usage is critical to avoid unnecessary checks
        print("Memory usage is critical. Reducing memory usage by moving the necessary transformer parts to the cpu.", flush=True)
        self.memory_mask = self.memory_mask.cpu()
        self.memory_indices = self.memory_indices.cpu()
        self.memories = self.memories.cpu()
        keys = ["memories", "memory_indices", "memory_mask"]
        self.samples_flat["memory_indices"] = self.samples_flat["memory_indices"].cpu()
        self.samples_flat["memory_mask"] = self.samples_flat["memory_mask"].cpu()
        #for key in keys:
        #    if key in mini_batch:
        #        mini_batch[key] = mini_batch[key].cpu()
        return mini_batch

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
                elif key == "log_probs" or "advantages" in key or key == "values":
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

    def to(self, device):
        """Args:
            device {torch.device} -- Desired device for all current tensors"""
        for k in dir(self):
            att = getattr(self, k)
            if torch.is_tensor(att):
                setattr(self, k, att.to(device))