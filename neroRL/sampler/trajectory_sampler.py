import numpy as np
import torch

from neroRL.sampler.buffer import Buffer
from neroRL.utils.worker import Worker

class TrajectorySampler():
    """The TrajectorySampler employs n environment workers to sample data for s worker steps regardless if an episode ended.
    Hence, the collected trajectories may contain multiple episodes or incomplete ones."""
    def __init__(self, configs, worker_id, observation_space, ground_truth_space, action_space_shape, model, sample_device, train_device) -> None:
        """Initializes the TrajectorSampler and launches its environment workers.

        Arguments:
            configs {dict} -- The whole set of configurations (e.g. training and environment configs)
            worker_id {int} -- Specifies the offset for the port to communicate with the environment, which is needed for Unity ML-Agents environments.
            observation_space {spaces.Dict} -- Observation space of the environment
            ground_truth_space {box} -- Dimensions of the ground truth space (None if not available)
            action_space_shape {tuple} -- Dimensions of the action space
            model {nn.Module} -- The model to retrieve the policy and value from
            sample_device {torch.device} -- The device that is used for retrieving the data from the model
            train_device {torch.device} -- The device that is used for training the model
        """
        # Set member variables
        self.observation_space = observation_space
        self.ground_truth_space = ground_truth_space
        self.model = model
        self.n_workers = configs["sampler"]["n_workers"]
        self.worker_steps = configs["sampler"]["worker_steps"]
        self.sample_device = sample_device
        self.train_device = train_device

        # Create Buffer
        self.buffer = Buffer(configs, observation_space, ground_truth_space, action_space_shape, self.train_device, self)

        # Launch workers
        self.workers = [Worker(configs["environment"], worker_id + 200 + w) for w in range(self.n_workers)]
        # Setup timestep placeholder
        self.worker_current_episode_step = torch.zeros((self.n_workers, ), dtype=torch.long)
        
        # Setup initial observations and ground truth information
        self.current_obs = {}
        for key, value in observation_space.spaces.items():
            self.current_obs[key] = np.zeros((self.n_workers,) + value.shape, dtype=np.float32)

        if ground_truth_space is not None:
            self.ground_truth = np.zeros((self.n_workers,) + ground_truth_space.shape, dtype=np.float32)
        else:
            self.ground_truth = None

        # Reset workers
        for worker in self.workers:
            worker.child.send(("reset", None))
        # Grab initial observations and ground truth information
        for i, worker in enumerate(self.workers):
            obs, info = worker.child.recv()
            for key, value in obs.items():
                self.current_obs[key][i] = value
            if self.ground_truth is not None:
                self.ground_truth[i] = info["ground_truth"]

    def sample(self) -> list:
        """Samples training data (i.e. experience tuples) using n workers for t worker steps.

        Returns:
            {list} -- List of completed episodes. Each episode outputs a dictionary containing at least the
            achieved reward and the episode length.
        """
        episode_infos = []

        # Sample actions from the model and collect experiences for training
        for t in range(self.worker_steps):
            # Gradients can be omitted for sampling data
            with torch.no_grad():
                # Write observations and recurrent cell to buffer
                self.previous_model_input_to_buffer(t)

                # Forward the model to retrieve the policy (making decisions), 
                # the states' value of the value function and the recurrent hidden states (if available)
                obs_batch = {}
                for key, item in self.current_obs.items():
                    obs_batch[key] = torch.tensor(item)
                policy, value = self.forward_model(obs_batch, t)      

                # Sample actions from each individual policy branch
                actions = []
                log_probs = []
                for action_branch in policy:
                    action = action_branch.sample()
                    actions.append(action)
                    log_probs.append(action_branch.log_prob(action))
                # Write actions, log_probs, and values to buffer
                self.buffer.actions[:, t] = torch.stack(actions, dim=1)
                self.buffer.log_probs[:, t] = torch.stack(log_probs, dim=1)
                self.buffer.values[:, t] = value.data

            # Execute actions
            actions = self.buffer.actions[:, t].cpu().numpy() # send actions as batch to the CPU, to save IO time
            for w, worker in enumerate(self.workers):
                worker.child.send(("step", actions[w]))

            # Retrieve results
            for w, worker in enumerate(self.workers):
                obs, self.buffer.rewards[w, t], self.buffer.dones[w, t], info = worker.child.recv()
                for key, value in obs.items():
                    self.current_obs[key][w] = value
                if self.ground_truth is not None:
                    self.ground_truth[w] = info["ground_truth"]
                if self.buffer.dones[w, t]:
                    # Store the information of the completed episode (e.g. total reward, episode length)
                    episode_infos.append(info)
                    # Reset the worker that concluded its episode
                    self.reset_worker(worker, w, t)
                else:
                    # Increment worker timestep
                    self.worker_current_episode_step[w] +=1

        return episode_infos

    def previous_model_input_to_buffer(self, t):
        """Add the model's previous input to the buffer.

        Arguments:
            t {int} -- Current step of sampling
        """
        for key, item in self.current_obs.items():
            self.buffer.obs[key][:, t] = torch.tensor(item)
        # The ground truth information is not used as model input, but is used as label to an auxiliary loss during optimization
        if self.ground_truth is not None:
            self.buffer.ground_truth[:, t] = torch.tensor(self.ground_truth)

    def forward_model(self, obs_batch, t):
        """Forwards the model to retrieve the policy and the value of the to be fed observations.

        Arguments:
            vis_obs {dict} -- Dict observations batched across workers
            t {int} -- Current step of sampling

        Returns:
            {tuple} -- policy {list of categorical distributions}, value {torch.tensor}
        """
        policy, value, _ = self.model(obs_batch)
        return policy, value

    def reset_worker(self, worker, id, t):
        """Resets the specified worker.

        Arguments:
            worker {remote} -- The to be reset worker
            id {int} -- The ID of the to be reset worker
            t {int} -- Current step of sampling data
        """
        # Reset the worker's current timestep
        self.worker_current_episode_step[id] = 0
        # Reset agent (potential interface for providing reset parameters)
        worker.child.send(("reset", None))
        # Get data from reset
        obs, info = worker.child.recv()
        for key, value in obs.items():
            self.current_obs[key][id] = value
        if self.ground_truth is not None:
            self.ground_truth[id] = info["ground_truth"]

    def get_last_value(self):
        """Returns the last value of the current observation to compute GAE.

        Returns:
            {torch.tensor} -- Last value
        """
        _, last_value, _ = self.model(torch.tensor(self.vis_obs) if self.vis_obs is not None else None,
                                        torch.tensor(self.vec_obs) if self.vec_obs is not None else None,
                                        None)
        return last_value

    def close(self) -> None:
        """Closes the sampler and shuts down its environment workers."""
        try:
            for worker in self.workers:
                worker.close()
        except:
            pass

    def to(self, device):
        """Args:
            device {torch.device} -- Desired device for all current tensors"""
        for k in dir(self):
            att = getattr(self, k)
            if torch.is_tensor(att):
                setattr(self, k, att.to(device))