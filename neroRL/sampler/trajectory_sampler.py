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
            vector_observation_space {tuple} -- Dimensions of the vector observation space (None if not available)
            action_space_shape {tuple} -- Dimensions of the action space
            model {nn.Module} -- The model to retrieve the policy and value from
            device {torch.device} -- The device that is used for retrieving the data from the model
        """
        # Set member variables
        self.visual_observation_space = visual_observation_space
        self.vector_observation_space = vector_observation_space
        self.model = model
        self.configs = configs
        self.n_workers = configs["sampler"]["n_workers"]
        self.worker_steps = configs["sampler"]["worker_steps"]
        self.device = device

        # Create Buffer
        self.use_helm = "helmv1" in configs["model"] or "helmv2" in configs["model"]
        self.buffer = Buffer(configs, visual_observation_space, vector_observation_space,
                        action_space_shape, self.use_helm, self.device, self.model.share_parameters, self)

        # Launch workers
        self.workers = [Worker(configs["environment"], worker_id + 200 + w) for w in range(self.n_workers)]
        # Setup timestep placeholder
        self.worker_current_episode_step = torch.zeros((self.n_workers, ), dtype=torch.long)
        
        # Setup initial observations
        if visual_observation_space is not None:
            self.vis_obs = np.zeros((self.n_workers,) + visual_observation_space.shape, dtype=np.float32)
        else:
            self.vis_obs = None
        if vector_observation_space is not None:
            self.vec_obs = np.zeros((self.n_workers,) + vector_observation_space, dtype=np.float32)
        else:
            self.vec_obs = None
            
        # Setup HELM memory
        if "helmv1" in configs["model"]:
            self.helm_memory = [torch.zeros((511, self.n_workers, 1024)) for _ in range(18)]
        elif "helmv2" in configs["model"]:
            self.helm_memory = [torch.zeros((128, self.n_workers, 1024)) for _ in range(18)]
        else:
            self.helm_memory = None

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
            device {torch.device} -- The device that is used for retrieving the data from the model

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
                vis_obs_batch = torch.tensor(self.vis_obs) if self.vis_obs is not None else None
                vec_obs_batch = torch.tensor(self.vec_obs) if self.vec_obs is not None else None
                if self.helm_memory is not None:
                    self.model.helm_encoder.memory = self.helm_memory
                policy, value, h_helm = self.forward_model(vis_obs_batch, vec_obs_batch, t)   
                if self.helm_memory is not None:
                    self.helm_memory = self.model.helm_encoder.memory
                if self.use_helm:
                    self.buffer.h_helm[:, t] = h_helm   

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
                vis_obs, vec_obs, self.buffer.rewards[w, t], self.buffer.dones[w, t], info = worker.child.recv()
                if self.vis_obs is not None:
                    self.vis_obs[w] = vis_obs
                if self.vec_obs is not None:
                    self.vec_obs[w] = vec_obs
                if info:
                    # Store the information of the completed episode (e.g. total reward, episode length)
                    episode_infos.append(info)
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
        if self.vis_obs is not None:
            self.buffer.vis_obs[:, t] = torch.tensor(self.vis_obs)
        if self.vec_obs is not None:
            self.buffer.vec_obs[:, t] = torch.tensor(self.vec_obs)

    def forward_model(self, vis_obs, vec_obs, t):
        """Forwards the model to retrieve the policy and the value of the to be fed observations.

        Arguments:
            vis_obs {torch.tensor} -- Visual observations batched across workers
            vec_obs {torch.tensor} -- Vector observations batched across workers
            t {int} -- Current step of sampling

        Returns:
            {tuple} -- policy {list of categorical distributions}, value {torch.tensor}
        """
        policy, value, _, _, h_helm = self.model(vis_obs, vec_obs)
        return policy, value, h_helm

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
        vis_obs, vec_obs = worker.child.recv()
        if self.vis_obs is not None:
            self.vis_obs[id] = vis_obs
        if self.vec_obs is not None:
            self.vec_obs[id] = vec_obs
        # Reset HELM Memory
        if self.use_helm:
            for l in range(len(self.helm_memory)):
                self.helm_memory[l][:, id] = 0.

    def get_last_value(self):
        """Returns the last value of the current observation to compute GAE.

        Returns:
            {torch.tensor} -- Last value
        """
        _, last_value, *_ = self.model(torch.tensor(self.vis_obs) if self.vis_obs is not None else None,
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