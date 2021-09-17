import torch
import numpy as np
class Buffer():
    """
    The buffer stores and prepares the training data. It supports recurrent policies.
    """
    def __init__(self, num_workers, worker_steps, visual_observation_space, vector_observation_space,
                    action_space_shape, recurrence, device, share_parameters):
        """
        Arguments:
            num_workers {int} -- Number of environments/agents to sample training data
            worker_steps {int} -- Number of steps per environment/agent to sample training data
            num_mini_batches {int} -- Number of mini batches that are used for each training epoch
            visual_observation_space {Box} -- Visual observation if available, else None
            vector_observation_space {tuple} -- Vector observation space if available, else None
            action_space_shape {tuple} -- Shape of the action space
            recurrence {dict} -- None if no recurrent policy is used, otherwise contains relevant details:
                - layer_type {str}, sequence_length {int}, hidden_state_size {int}, hiddens_state_init {str}, reset_hidden_state {bool}
            device {torch.device} -- The device that will be used for training/storing single mini batches
        """
        self.device = device
        self.recurrence = recurrence
        self.sequence_length = recurrence["sequence_length"] if recurrence is not None else None
        self.num_workers = num_workers
        self.worker_steps = worker_steps
        self.batch_size = self.num_workers * self.worker_steps
        self.rewards = np.zeros((num_workers, worker_steps), dtype=np.float32)
        self.actions = torch.zeros((num_workers, worker_steps, len(action_space_shape)), dtype=torch.long)
        self.dones = np.zeros((num_workers, worker_steps), dtype=np.bool)
        if visual_observation_space is not None:
            self.vis_obs = torch.zeros((num_workers, worker_steps) + visual_observation_space.shape, dtype=torch.float32)
        else:
            self.vis_obs = None
        if vector_observation_space is not None:
            self.vec_obs = torch.zeros((num_workers, worker_steps,) + vector_observation_space, dtype=torch.float32)
        else:
            self.vec_obs = None
        
        if share_parameters:
            self.hxs = torch.zeros((num_workers, worker_steps, recurrence["hidden_state_size"]), dtype=torch.float32) if recurrence is not None else None
            self.cxs = torch.zeros((num_workers, worker_steps, recurrence["hidden_state_size"]), dtype=torch.float32) if recurrence is not None else None
        else: # if parameters are not shared then add two extra dimensions for adding enough capacity to store the hidden states of the actor and critic model
            self.hxs = torch.zeros((num_workers, worker_steps, recurrence["hidden_state_size"], 2), dtype=torch.float32) if recurrence is not None else None
            self.cxs = torch.zeros((num_workers, worker_steps, recurrence["hidden_state_size"], 2), dtype=torch.float32) if recurrence is not None else None

        self.log_probs = torch.zeros((num_workers, worker_steps, len(action_space_shape)), dtype=torch.float32)
        self.values = torch.zeros((num_workers, worker_steps), dtype=torch.float32)
        self.advantages = torch.zeros((num_workers, worker_steps), dtype=torch.float32)
        self.num_sequences = 0
        self.actual_sequence_length = 0

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
            delta = torch.FloatTensor(self.rewards[:, t]) + gamma * last_value - self.values[:, t]
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
            "loss_mask": torch.ones((self.num_workers, self.worker_steps), dtype=torch.float32)
        }

    	# Add available observations to the dictionary
        if self.vis_obs is not None:
            samples["vis_obs"] = self.vis_obs
        if self.vec_obs is not None:
            samples["vec_obs"] = self.vec_obs

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
            
            # Split vis_obs, vec_obs, values, advantages, recurrent cell states, actions and log_probs into episodes and then into sequences
            for key, value in samples.items():
                sequences = []
                for w in range(self.num_workers):
                    start_index = 0
                    for done_index in episode_done_indices[w]:
                        # Split trajectory into episodes
                        episode = value[w, start_index:done_index + 1]
                        start_index = done_index + 1
                        # Split episodes into sequences
                        if self.sequence_length > 0:
                            for start in range(0, len(episode), self.sequence_length):
                                end = start + self.sequence_length
                                sequences.append(episode[start:end])
                            max_sequence_length = self.sequence_length
                        else:
                            # If the sequence length is not set to a proper value, sequences will be based on episodes
                            sequences.append(episode)
                            max_sequence_length = len(episode) if len(episode) > max_sequence_length else max_sequence_length
                
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
        self.num_sequences = len(samples["values"])
        self.actual_sequence_length = max_sequence_length
        
        # Flatten all samples
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
        # At this point it is assumed that all of the available training data (values, observations, actions, ...) is padded.

        # Compose mini batches
        start = 0
        for num_sequences in num_sequences_per_batch:
            end = start + num_sequences
            mini_batch_indices = indices[sequence_indices[start:end]].reshape(-1)
            mini_batch = {}
            for key, value in self.samples_flat.items():
                if key != "hxs" and key != "cxs":
                    mini_batch[key] = value[mini_batch_indices].to(self.device)
                else:
                    # Collect recurrent cell states
                    mini_batch[key] = value[sequence_indices[start:end]].to(self.device)
            start = end
            yield mini_batch