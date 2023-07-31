import torch
import numpy as np

from neroRL.utils.utils import batched_index_select

class Buffer():
    """The buffer stores and prepares the training data. It supports recurrent and transformer policies."""
    def __init__(self, configs, visual_observation_space, vector_observation_space, ground_truth_space,
                    action_space_shape, train_device, sampler):
        """
        Arguments:
            configs {dict} -- The whole set of configurations (e.g. model, training, environment, ... configs)
            visual_observation_space {Box} -- Visual observation if available, else None
            vector_observation_space {tuple} -- Vector observation space if available, else None
            ground_truth_space {Box} -- Ground truth space if available, else None
            action_space_shape {tuple} -- Shape of the action space
            train_device {torch.device} -- Single mini batches will be moved to this device for model optimization
            sampler {TrajectorySampler} -- The used sampler
        """
        self.train_device = train_device
        self.sampler = sampler
        self.configs = configs
        self.recurrence, self.transformer_memory = None, None       # place holder for recurrence and transformer config
        self.num_workers = configs["sampler"]["n_workers"]
        self.worker_steps = configs["sampler"]["worker_steps"]
        self.batch_size = self.num_workers * self.worker_steps
        self.action_space_shape = action_space_shape
        self.visual_observation_space = visual_observation_space
        self.vector_observation_space = vector_observation_space
        self.ground_truth_space = ground_truth_space
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
        if self.ground_truth_space is not None:
            self.ground_truth = torch.zeros((self.num_workers, self.worker_steps) + self.ground_truth_space.shape)
        self.rewards = np.zeros((self.num_workers, self.worker_steps), dtype=np.float32)
        self.actions = torch.zeros((self.num_workers, self.worker_steps, len(self.action_space_shape)), dtype=torch.long)
        self.dones = np.zeros((self.num_workers, self.worker_steps), dtype=bool)
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
        self.hxs = torch.zeros((self.num_workers, self.worker_steps, num_layers, hidden_state_size))
        if layer_type == "lstm":
            self.cxs = torch.zeros((self.num_workers, self.worker_steps, num_layers, hidden_state_size))

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
        # Flag that indicates whether the GPU is out of memory
        self.out_of_memory = False

        # Supply training samples
        samples = {"actions": self.actions} # actions are added at this point to ensure that these are padded while using recurrence

        # OBSERVATION SAMPLES
    	# Add available observations to the dictionary
        if self.vis_obs is not None:
            samples["vis_obs"] = self.vis_obs
        if self.vec_obs is not None:
            samples["vec_obs"] = self.vec_obs
        if self.ground_truth_space is not None:
            samples["ground_truth"] = self.ground_truth

        # TRANSFORMER SAMPLES
        # Add data concerned with the episodic memory (i.e. transformer-based policy)
        if self.transformer_memory is not None:
            # Determine max episode steps (also consideres ongoing episodes)
            self.actual_max_episode_steps = self.memory_indices.max().item() + 1
            # Stack memories, add buffer fields to samples and remove unnecessary padding
            self.memories = torch.stack(self.memories)
            if self.max_episode_steps >= self.actual_max_episode_steps:
                samples["memory_mask"] = self.memory_mask[:, :, :self.actual_max_episode_steps]
                samples["memory_indices"] = self.memory_indices[:, :, :self.actual_max_episode_steps]
                self.memories = self.memories[:, :self.actual_max_episode_steps]
            else:
                samples["memory_mask"] = self.memory_mask[:, :, :self.actual_max_episode_steps].clone()
                samples["memory_indices"] = self.memory_indices[:, :, :self.actual_max_episode_steps].clone()
                self.memories = self.memories[:, :self.actual_max_episode_steps].clone()
            samples["memory_index"] = self.memory_index

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
                try:
                    samples[key] = torch.stack(sequences, axis=0)
                except RuntimeError:
                    self.out_of_memory = True
                    print("OUT OF MEMORY - Stack Recurrence Sequences - Data Key: " + str(key) + " - Shape: " + str(sequences[0].shape), flush=True)
                
                if self.out_of_memory:
                    # Send sequences to CPU
                    sequences = [seq.cpu() for seq in sequences]
                    # Stack sequences as tensor and send to GPU
                    samples[key] = torch.stack(sequences, axis=0).to(self.train_device)
                    self.out_of_memory = False

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

    def _gather_memory_windows_loop(self, mini_batch_size, mini_batch_indices):
        """Gathers the memory windows for the concerned mini batch.
        To avoid out of memory errors, the data is processed using a loop that processes single batch items.

        Arguments:
            mini_batch_size {int} -- Size of the mini batch that deterimines the number of memory windows to be gathered
            mini_batch_indices {torch.tensor} -- Indices that determine the memory windows to be gathered

        Returns:
            torch.tensor -- The gathered memory windows for the concerned mini batch update
        """
        memory_windows = torch.zeros((mini_batch_size, min(self.actual_max_episode_steps, self.memory_length), self.num_mem_layers, self.mem_layer_size)).to(self.train_device)
        for i in range(mini_batch_size):
            # Select memory
            memory_index = self.samples_flat["memory_index"][mini_batch_indices[i]]
            memory = self.memories[memory_index]
            # Slice memory window
            memory_indices = self.samples_flat["memory_indices"][mini_batch_indices[i]]
            memory_window = memory[memory_indices[0] : memory_indices[-1] + 1]
            # Add memory window to mini batch
            memory_windows[i][:memory_window.shape[0]] = memory_window
        return memory_windows
    
    def _gather_memory_windows_batched(self, mini_batch_size, mini_batch_indices):
        """Gathers the memory windows for the concerned mini batch.
        To avoid out of memory errors, the data is processed using a loop that processes chunks.
        This is the default function that is used.

        Arguments:
            mini_batch_size {int} -- Size of the mini batch that deterimines the number of memory windows to be gathered
            mini_batch_indices {torch.tensor} -- Indices that determine the memory windows to be gathered

        Returns:
            torch.tensor -- The gathered memory windows for the concerned mini batch update
        """
        memory_windows = torch.zeros((mini_batch_size, min(self.actual_max_episode_steps, self.memory_length), self.num_mem_layers, self.mem_layer_size)).to(self.train_device)
        step_size = 256
        for i in range(0, mini_batch_size, step_size):
            # Slice mini batch indices
            indices = mini_batch_indices[i:i+step_size]
            # Select memories (memory overhead)
            selected_memories = self.memories[self.samples_flat["memory_index"][indices]]
            # Select and write memory windows (memory overhead)
            memory_indices = self.samples_flat["memory_indices"][indices, :self.actual_max_episode_steps]
            memory_windows[i:i+step_size, :memory_windows.shape[1]] = batched_index_select(selected_memories, 1, memory_indices)
        return memory_windows
    
    def _gather_memory_windows_cpu(self, mini_batch_size, mini_batch_indices):
        """Gathers the memory windows for the concerned mini batch.
        To avoid out of memory errors, the data is sent to CPU, processed and then sent back to GPU.

        Arguments:
            mini_batch_size {int} -- Size of the mini batch that deterimines the number of memory windows to be gathered
            mini_batch_indices {torch.tensor} -- Indices that determine the memory windows to be gathered

        Returns:
            torch.tensor -- The gathered memory windows for the concerned mini batch update
        """
        memory_windows = torch.zeros((mini_batch_size, min(self.actual_max_episode_steps, self.memory_length), self.num_mem_layers, self.mem_layer_size)).cpu()
        # Select memories (memory overhead)
        selected_memories = self.memories[self.samples_flat["memory_index"][mini_batch_indices].cpu()].cpu()
        # Select and write memory windows (memory overhead)
        memory_indices = self.samples_flat["memory_indices"][mini_batch_indices].cpu()
        memory_windows[:, :memory_windows.shape[1]] = batched_index_select(selected_memories, 1, memory_indices)
        self.samples_flat["memory_indices"] = self.samples_flat["memory_indices"].to(self.train_device)
        return memory_windows.to(self.train_device)

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
                mini_batch[key] = value[mini_batch_indices].to(self.train_device)
            # Retrieve memory windows for the concerned mini batch, if TrXL is used
            if self.transformer_memory is not None:
                mini_batch["memory_window"] = self._gather_memory_windows_batched(mini_batch_size, mini_batch_indices)
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
                    mini_batch[key] = value[sequence_indices[start:end]].to(self.train_device)
                elif key == "log_probs" or "advantages" in key or key == "values":
                    # Select unpadded data
                    mini_batch[key] = value[mini_batch_unpadded_indices].to(self.train_device)
                else:
                    # Select padded data
                    mini_batch[key] = value[mini_batch_padded_indices].to(self.train_device)
            start = end
            yield mini_batch

    def to(self, device):
        """Args:
            device {torch.device} -- Desired device for all current tensors"""
        for k in dir(self):
            att = getattr(self, k)
            if torch.is_tensor(att):
                setattr(self, k, att.to(device))