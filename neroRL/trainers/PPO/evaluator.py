import numpy as np
import torch
import time

from neroRL.utils.worker import Worker

class Evaluator():
    """Evaluates a model based on the initially provided config."""
    def __init__(self, configs, worker_id, visual_observation_space, vector_observation_space):
        """Initializes the evaluator and its environments
        
        Arguments:
            eval_config {dict} -- The configuration of the evaluation
            env_config {dict} -- The configurgation of the environment
            worker_id {int} -- The offset of the port to communicate with the environment
            visual_observation_space {box} -- Visual observation space of the environment
            vector_observation_space {tuple} -- Vector observation space of the environment
        """
        # Set members
        self.configs = configs
        self.n_workers = configs["evaluation"]["n_workers"]
        self.seeds = configs["evaluation"]["seeds"]
        self.visual_observation_space = visual_observation_space
        self.vector_observation_space = vector_observation_space

        # Launch environments
        self.workers = []
        for i in range(self.n_workers):
            id = worker_id + i + 200 - self.n_workers
            self.workers.append(Worker(configs["environment"], id))

        # Check for recurrent policy
        self.recurrence = None if not "recurrence" in configs["model"] else configs["model"]["recurrence"]

    def evaluate(self, model, device):
        """Evaluates a provided model on the already initialized evaluation environments.

        Arguments:
            model {nn.Module} -- The to be evaluated model
            device {torch.device} -- The to be used device for executing the model

        Returns:
            eval_duration {float} -- The duration of the completed evaluation
            episode_infos {dict} -- The raw results of each evaluated episode
        """
        time_start = time.time()
        episode_infos = []
        # Loop over all seeds
        for seed in self.seeds:
            # Initialize observations
            if self.visual_observation_space is not None:
                vis_obs = np.zeros((self.n_workers,) + self.visual_observation_space.shape, dtype=np.float32)
            else:
                vis_obs = None
            if self.vector_observation_space is not None:
                vec_obs = np.zeros((self.n_workers,) + self.vector_observation_space, dtype=np.float32)
            else:
                vec_obs = None

            # Initialize recurrent cell (hidden/cell state)
            # We specifically initialize a recurrent cell for each worker,
            # because one of the available initialization methods samples a hidden cell state.
            recurrent_cell = []
            for _ in range(self.n_workers):
                if self.recurrence is not None:
                    hxs, cxs = model.init_recurrent_cell_states(1, device)
                    if self.recurrence["layer_type"] == "gru":
                        recurrent_cell.append(hxs)
                    elif self.recurrence["layer_type"] == "lstm":
                        recurrent_cell.append((hxs, cxs))
                else:
                    recurrent_cell.append(None)
            
            # Reset workers and set evaluation seed
            for worker in self.workers:
                worker.child.send(("reset", {"start-seed": seed, "num-seeds": 1}))
            # Grab initial observations
            for w, worker in enumerate(self.workers):
                vis, vec = worker.child.recv()
                if vis_obs is not None:
                    vis_obs[w] = vis
                if vec_obs is not None:
                    vec_obs[w] = vec

            # Every worker plays its episode
            dones = np.zeros(self.n_workers, dtype=bool)

            with torch.no_grad():
                while not np.all(dones):
                    # Sample action and send it to the worker if not done
                    for w, worker in enumerate(self.workers):
                        if not dones[w]:
                            # While sampling data for training we feed batches containing all workers,
                            # but as we evaluate entire episodes, we feed one worker at a time
                            policy, _, recurrent_cell[w] = model(np.expand_dims(vis_obs[w], 0) if vis_obs is not None else None,
                                                np.expand_dims(vec_obs[w], 0) if vec_obs is not None else None,
                                                recurrent_cell[w],
                                                device)

                            actions = []
                            for action_branch in policy:
                                action = action_branch.sample()
                                actions.append(action.cpu().data.item())
                            worker.child.send(("step", actions))

                    # Receive and process step result if not done
                    for w, worker in enumerate(self.workers):
                        if not dones[w]:
                            vis, vec, _, dones[w], info = worker.child.recv()
                            if vis_obs is not None:
                                vis_obs[w] = vis
                            if vec_obs is not None:
                                vec_obs[w] = vec
                            if info:
                                info["seed"] = seed
                                episode_infos.append(info)
        
            # Seconds needed for a whole update
            time_end = time.time()
            eval_duration = int(time_end - time_start)

        # Return the duration of the evaluation and the raw episode results
        return eval_duration, episode_infos

    def close(self):
        """Closes the Evaluator and destroys all worker."""
        for worker in self.workers:
                worker.child.send(("close", None))
