import numpy as np
import torch

from neroRL.sampler.trajectory_sampler import TrajectorySampler

class RecurrentSampler(TrajectorySampler):
    """The TrajectorySampler employs n environment workers to sample data for s worker steps regardless if an episode ended.
    Hence, the collected trajectories may contain multiple episodes or incomplete ones. The RecurrentSampler takes care of
    resetting and adding recurrent cell states (i.e. agent memory) to the buffer."""
    def __init__(self, configs, worker_id, observation_space, ground_truth_space, action_space_shape, model, sample_device, train_device) -> None:
        """Initializes the RecurrentSampler and launches its environment worker."""
        super().__init__(configs, worker_id, observation_space, ground_truth_space, action_space_shape, model, sample_device, train_device)
        # Set member variables
        self.layer_type = configs["model"]["recurrence"]["layer_type"]
        self.reset_hidden_state = configs["model"]["recurrence"]["reset_hidden_state"]
        # Add relevant fields to the buffer
        self.buffer.init_recurrent_buffer_fields()

        # Setup initial recurrent cell
        hxs, cxs = self.model.init_recurrent_cell_states(self.n_workers, self.sample_device)
        if self.layer_type == "gru":
            self.recurrent_cell = hxs
        elif self.layer_type == "lstm":
            self.recurrent_cell = (hxs, cxs)

    def previous_model_input_to_buffer(self, t):
        """Add the model's previous input, as well as the recurrent cell states, to the buffer."""
        super().previous_model_input_to_buffer(t)
        # Store recurrent cell states inside the buffer
        if self.layer_type == "gru":
            self.buffer.hxs[:, t] = self.recurrent_cell
        elif self.layer_type == "lstm":
            self.buffer.hxs[:, t] = self.recurrent_cell[0]
            self.buffer.cxs[:, t] = self.recurrent_cell[1]

    def forward_model(self, obs_batch, t):
        """Forwards the model to retrieve the policy and the value of the to be fed observations and recurrent cell state."""
        # The recurrent cell state is the agent's memory
        policy, value, self.recurrent_cell = self.model(obs_batch, self.recurrent_cell)
        return policy, value

    def reset_worker(self, worker, id, t):
        """Resets the specified worker and resets the agent's recurrent cell state."""
        super().reset_worker(worker, id, t)
        if self.reset_hidden_state:
            hxs, cxs = self.model.init_recurrent_cell_states(1, self.sample_device)
            if self.layer_type == "gru":
                self.recurrent_cell[id] = hxs
            elif self.layer_type == "lstm":
                self.recurrent_cell[0][id] = hxs
                self.recurrent_cell[1][id] = cxs

    def get_last_value(self):
        """Returns the last value of the current observation and recurrent cell state to compute GAE."""
        obs = {}
        for key, value in self.current_obs.items():
            obs[key] = torch.tensor(value)
        _, last_value, _ = self.model(obs, self.recurrent_cell)
        return last_value
