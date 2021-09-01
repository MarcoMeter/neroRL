import numpy as np
import torch

from neroRL.sampler.buffer import Buffer
from neroRL.utils.worker import Worker

class TrajectorySampler():
    """The TrajectorySampler employs n environment workers to sample data for s worker steps regardless if an episode ended.
    Hence, the collected trajectories may contain multiple episodes or incomplete ones."""
    def __init__(self, configs, worker_id, visual_observation_space, vector_observation_space, action_space_shape, model, device) -> None:
        """Initializes the TrajectorSampler and launches its environment workers.

        Arguments:
            configs {dict} -- The whole set of configurations (e.g. training and environment configs)
            worker_id {int} -- Specifies the offset for the port to communicate with the environment, which is needed for Unity ML-Agents environments.
            visual_observation_space {box} -- Dimensions of the visual observation space (None if not available)
            vector_observation_space {} -- {tuple} -- Dimensions of the vector observation space (None if not available)
            model {nn.Module} -- The model that represents the policy and the value function
            device {torch.device} -- The device that is used for retrieving the data from the model.
        """
        # Set member variables
        self.configs = configs
        self.visual_observation_space = visual_observation_space
        self.vector_observation_space = vector_observation_space
        self.model = model
        self.n_workers = configs["sampler"]["n_workers"]
        self.worker_steps = configs["sampler"]["worker_steps"]
        self.recurrence = None if not "recurrence" in configs["model"] else configs["model"]["recurrence"]
        self.device = device

        # Create Buffer
        self.buffer = Buffer(self.n_workers, self.worker_steps, visual_observation_space, vector_observation_space,
                        action_space_shape, self.recurrence, self.device, self.model.share_parameters)

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
            hxs, cxs = self.model.init_recurrent_cell_states(self.n_workers, self.device)
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
        """Samples training data (i.e. experience tuples) using n workers for t worker steps.

        Arguments:
            device {torch.device} -- The device that is used for retrieving the data from the model.

        Returns:
            {list} -- List of completed episodes. Each episode outputs a dictionary containing at least the
            achieved reward and the episode length.
        """
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
                            hxs, cxs = self.model.init_recurrent_cell_states(1, self.device)
                            if self.recurrence["layer_type"] == "gru":
                                self.recurrent_cell[:, w] = hxs
                            elif self.recurrence["layer_type"] == "lstm":
                                self.recurrent_cell[0][:, w] = hxs
                                self.recurrent_cell[1][:, w] = cxs

        return episode_infos

    def last_vis_obs(self) -> np.ndarray:
        """
        Returns:
            {np.ndarray} -- The last visual observation of the sampling process, which can be used to calculate the advantage.
        """
        return self.vis_obs

    def last_vec_obs(self) -> np.ndarray:
        """
        Returns:
            {np.ndarray} -- The last vector observation of the sampling process, which can be used to calculate the advantage.
        """
        return self.vec_obs

    def last_recurrent_cell(self) -> tuple:
        """
        Returns:
            {tuple} -- The latest recurrent cell of the sampling process, which can be used to calculate the advantage.
        """
        return self.recurrent_cell

    def close(self) -> None:
        """Closes the sampler and shuts down its environment workers."""
        try:
            for worker in self.workers:
                worker.child.send(("close", None))
        except:
            pass