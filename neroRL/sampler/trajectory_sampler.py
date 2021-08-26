import numpy as np
import torch

from neroRL.utils.worker import Worker

class TrajectorySampler():
    def __init__(self, configs, worker_id, visual_observation_space, vector_observation_space, model, buffer, device, mini_batch_device) -> None:
        # Set member variables
        self.configs = configs
        self.visual_observation_space = visual_observation_space
        self.vector_observation_space = vector_observation_space
        self.model = model
        self.buffer = buffer
        self.n_workers = configs["trainer"]["n_workers"]
        self.worker_steps = configs["trainer"]["worker_steps"]
        self.recurrence = None if not "recurrence" in configs["model"] else configs["model"]["recurrence"]
        self.device = device
        self.mini_batch_device = mini_batch_device

        # TODO
        self.gamma = configs["trainer"]["gamma"]
        self.lamda = configs["trainer"]["lamda"]

        # Launch workers
        self.workers = [Worker(configs["environment"], worker_id + 200 + w) for w in range(self.n_workers)]
        
        # Setup initial observations
        if visual_observation_space is not None:
            self.vis_obs = np.zeros((self.n_workers,) + visual_observation_space.shape, dtype=np.float32)
        else:
            self.vis_obs = None
        if vector_observation_space is not None:
            self.vec_obs = np.zeros((self.n_workers,) + vector_observation_space, dtype=np.float32)
        else:
            self.vec_obs = None

        # Setup initial recurrent cell
        if self.recurrence is not None:
            hxs, cxs = self.model.init_recurrent_cell_states(self.n_workers, self.mini_batch_device)
            if self.recurrence["layer_type"] == "gru":
                self.recurrent_cell = hxs
            elif self.recurrence["layer_type"] == "lstm":
                self.recurrent_cell = (hxs, cxs)
        else:
            self.recurrent_cell = None

        # Reset workers
        for worker in self.workers:
            worker.child.send(("reset", None))
        # Grab initial observations
        for i, worker in enumerate(self.workers):
            vis_obs, vec_obs = worker.child.recv()
            if self.vis_obs is not None:
                self.vis_obs[i] = vis_obs
            if self.vec_obs is not None:
                self.vec_obs[i] = vec_obs

    def sample(self, device) -> list:
        episode_infos = []

        # Sample actions from the model and collect experiences for training
        for t in range(self.worker_steps):
            # Gradients can be omitted for sampling data
            with torch.no_grad():
                # Save the initial observations and hidden states
                if self.vis_obs is not None:
                    self.buffer.vis_obs[:, t] = self.vis_obs
                if self.vec_obs is not None:
                    self.buffer.vec_obs[:, t] = self.vec_obs
                # Store recurrent cell states inside the buffer
                if self.recurrence is not None:
                    if self.recurrence["layer_type"] == "gru":
                        self.buffer.hxs[:, t] = self.recurrent_cell.squeeze(0).cpu().numpy()
                    elif self.recurrence["layer_type"] == "lstm":
                        self.buffer.hxs[:, t] = self.recurrent_cell[0].squeeze(0).cpu().numpy()
                        self.buffer.cxs[:, t] = self.recurrent_cell[1].squeeze(0).cpu().numpy()

                # Forward the model to retrieve the policy (making decisions), the states' value of the value function and the recurrent hidden states (if available)
                policy, value, self.recurrent_cell = self.model(self.vis_obs, self.vec_obs, self.recurrent_cell, device)
                self.buffer.values[:, t] = value.cpu().data.numpy()

                # Sample actions from each individual policy branch
                actions = []
                log_probs = []
                for action_branch in policy:
                    action = action_branch.sample()
                    actions.append(action.cpu().data.numpy())
                    log_probs.append(action_branch.log_prob(action).cpu().data.numpy())
                actions = np.transpose(actions)
                log_probs = np.transpose(log_probs)
                self.buffer.actions[:, t] = actions
                self.buffer.log_probs[:, t] = log_probs

            # Execute actions
            for w, worker in enumerate(self.workers):
                worker.child.send(("step", self.buffer.actions[w, t]))

            # Retrieve results
            for w, worker in enumerate(self.workers):
                vis_obs, vec_obs, self.buffer.rewards[w, t], self.buffer.dones[w, t], info = worker.child.recv()
                if self.vis_obs is not None:
                    self.vis_obs[w] = vis_obs
                if self.vec_obs is not None:
                    self.vec_obs[w] = vec_obs
                if info:
                    # Store the information of the completed episode (e.g. total reward, episode length)
                    episode_infos.append(info)
                    # Reset agent (potential interface for providing reset parameters)
                    worker.child.send(("reset", None))
                    # Get data from reset
                    vis_obs, vec_obs = worker.child.recv()
                    if self.vis_obs is not None:
                        self.vis_obs[w] = vis_obs
                    if self.vec_obs is not None:
                        self.vec_obs[w] = vec_obs
                    # Reset recurrent cell states
                    if self.recurrence is not None:
                        if self.recurrence["reset_hidden_state"]:
                            hxs, cxs = self.model.init_recurrent_cell_states(1, self.mini_batch_device)
                            if self.recurrence["layer_type"] == "gru":
                                self.recurrent_cell[:, w] = hxs
                            elif self.recurrence["layer_type"] == "lstm":
                                self.recurrent_cell[0][:, w] = hxs
                                self.recurrent_cell[1][:, w] = cxs
                                
        # Calculate advantages
        _, last_value, _ = self.model(self.vis_obs, self.vec_obs, self.recurrent_cell, device)
        self.buffer.calc_advantages(last_value.cpu().data.numpy(), self.gamma, self.lamda)

        return episode_infos

    def close(self) -> None:
        try:
            for worker in self.workers:
                worker.child.send(("close", None))
        except:
            pass