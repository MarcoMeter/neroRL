import numpy as np
import torch

from neroRL.sampler.trajectory_sampler import TrajectorySampler

class RecurrentSampler(TrajectorySampler):
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
        self.layer_type = configs["model"]["recurrence"]["layer_type"]
        self.reset_hidden_state = configs["model"]["recurrence"]["reset_hidden_state"]
        self.buffer.init_recurrent_buffer_fields()

        # Setup initial recurrent cell
        hxs, cxs = self.model.init_recurrent_cell_states(self.n_workers, self.device)
        if self.layer_type == "gru":
            self.recurrent_cell = hxs
        elif self.layer_type == "lstm":
            self.recurrent_cell = (hxs, cxs)

    def previous_model_input_to_buffer(self, t):
        super().previous_model_input_to_buffer(t)
        # Store recurrent cell states inside the buffer
        if self.layer_type == "gru":
            self.buffer.hxs[:, t] = self.recurrent_cell
        elif self.layer_type == "lstm":
            self.buffer.hxs[:, t] = self.recurrent_cell[0]
            self.buffer.cxs[:, t] = self.recurrent_cell[1]

    def forward_model(self, vis_obs, vec_obs, t):
        policy, value, self.recurrent_cell, _ = self.model(vis_obs, vec_obs, self.recurrent_cell)
        return policy, value

    def reset_worker(self, worker, id, t):
        super().reset_worker(worker, id, t)
        if self.reset_hidden_state:
            hxs, cxs = self.model.init_recurrent_cell_states(1, self.device)
            if self.layer_type == "gru":
                self.recurrent_cell[id] = hxs
            elif self.layer_type == "lstm":
                self.recurrent_cell[0][id] = hxs
                self.recurrent_cell[1][id] = cxs

    def get_last_value(self):
        _, last_value, _, _ = self.model(torch.tensor(self.vis_obs) if self.vis_obs is not None else None,
                                        torch.tensor(self.vec_obs) if self.vec_obs is not None else None,
                                        self.recurrent_cell)
        return last_value
