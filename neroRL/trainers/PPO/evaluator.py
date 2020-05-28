import numpy as np
import torch
import time

from neroRL.utils.worker import Worker

class Evaluator():
    """Evaluates a model based on the initially provided config."""
    def __init__(self, eval_config, env_config, worker_id, visual_observation_space, vector_observation_space):
        """Initializes the evaluator and its environments
        
        Arguments:
            eval_config {dict} -- The configuration of the evaluation
            env_config {dict} -- The configurgation of the environment
            worker_id {int} -- The offset of the port to communicate with the environment
            visual_observation_space {box} -- Visual observation space of the environment
            vector_observation_space {tuple} -- Vector observation space of the environment
        """
        # Set members
        self.n_workers = eval_config["n_workers"]
        self.seeds = eval_config["seeds"]
        self.visual_observation_space = visual_observation_space
        self.vector_observation_space = vector_observation_space

        # Launch environments
        self.workers = []
        for i in range(self.n_workers):
            id = worker_id + i + 200 - self.n_workers
            self.workers.append(Worker(env_config, id))

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
            # Initialize hidden state
            if model.use_recurrent:
                hidden_state = torch.zeros((self.n_workers, 1, model.hidden_state_size), dtype=torch.float32, device=device)
            else:
                hidden_state = [None] * self.n_workers

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
                    # Sample action and send to worker if not done
                    for w, worker in enumerate(self.workers):
                        if not dones[w]:
                            policy, _, hidden_state[w] = model(np.expand_dims(vis_obs[w], 0) if vis_obs is not None else None,
                                                np.expand_dims(vec_obs[w], 0) if vec_obs is not None else None,
                                                hidden_state[w],
                                                device)

                            actions = []
                            for action_branch in policy:
                                action = action_branch.sample()
                                actions.append(action.cpu().data.numpy())
                            actions = np.transpose(actions)
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
