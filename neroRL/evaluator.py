import numpy as np
import torch
import time

from neroRL.utils.worker import Worker
from neroRL.utils.video_recorder import VideoRecorder
from neroRL.wrapper.truncate_memory import TruncateMemory

class Evaluator():
    """Evaluates a model based on the initially provided config."""
    def __init__(self, configs, model_config, worker_id, visual_observation_space, vector_observation_space,
                max_episode_steps, memory_length = -2, video_path = "video", record_video = False, frame_rate = 1, generate_website = False):
        """Initializes the evaluator and its environments
        
        Arguments:
            eval_config {dict} -- The config of the evaluation
            env_config {dict} -- The config of the environment
            worker_id {int} -- The offset of the port to communicate with the environment
            visual_observation_space {box} -- Visual observation space of the environment
            vector_observation_space {tuple} -- Vector observation space of the environment
        """
        # Set members
        self.configs = configs
        self.n_workers = configs["evaluation"]["n_workers"]
        if "explicit-seeds" in configs["evaluation"]["seeds"]:
            self.seeds = configs["evaluation"]["seeds"]["explicit-seeds"]
        else:
            start = configs["evaluation"]["seeds"]["start-seed"]
            self.seeds = list(range(start, start + configs["evaluation"]["seeds"]["num-seeds"]))
        self.visual_observation_space = visual_observation_space
        self.vector_observation_space = vector_observation_space
        self.max_episode_steps = max_episode_steps
        self.memory_length = memory_length
        self.video_path = video_path
        self.record_video = record_video
        self.frame_rate = frame_rate
        self.generate_website = generate_website

        # Launch environments
        self.workers = []
        for i in range(self.n_workers):
            id = worker_id + i + 200 - self.n_workers
            self.workers.append(Worker(configs["environment"], id, record_video = record_video))

        # Check for recurrent policy
        self.recurrence_config = model_config["recurrence"] if "recurrence" in model_config else None
        # Check for transformer policy
        self.transformer_config = model_config["transformer"] if "transformer" in model_config else None

    def init_recurrent_cell(self, recurrence_config, model, device):
        # Initialize recurrent cell (hidden/cell state)
        # We specifically initialize a recurrent cell for each worker,
        # because one of the available initialization methods samples a hidden cell state.
        recurrent_cell = []
        for _ in range(self.n_workers):
            hxs, cxs = model.init_recurrent_cell_states(1, device)
            if recurrence_config["layer_type"] == "gru":
                recurrent_cell.append(hxs)
            elif recurrence_config["layer_type"] == "lstm":
                recurrent_cell.append((hxs, cxs))
        return recurrent_cell

    def init_transformer_memory(self, trxl_conf, model, device):
        self.memory_length = trxl_conf["memory_length"]
        memory_mask = torch.tril(torch.ones((self.memory_length, self.memory_length)), diagonal=-1)
        repetitions = torch.repeat_interleave(torch.arange(0, self.memory_length).unsqueeze(0), self.memory_length - 1, dim = 0).long()
        memory_indices = torch.stack([torch.arange(i, i + self.memory_length) for i in range(trxl_conf["max_episode_steps"] - self.memory_length + 1)]).long()
        memory_indices = torch.cat((repetitions, memory_indices))
        memory = []
        for _ in range(self.n_workers):
            memory.append(model.init_transformer_memory(1, trxl_conf["max_episode_steps"], trxl_conf["num_blocks"], trxl_conf["embed_dim"], device))
        return memory, memory_mask, memory_indices

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
        # Truncate memory if specified
        if self.memory_length != -2:
            model = TruncateMemory(model, self.model_config, self.memory_length, device)
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

            # Init memory if applicable
            memory = [None for _ in range(self.n_workers)]
            memory_mask, memory_indices, mask = None, None, None
            # Initialize recurrent cell (hidden/cell state)
            if self.recurrence_config is not None:
                memory = self.init_recurrent_cell(self.recurrence_config, model, device)
            # Initialize the transformer memory
            if self.transformer_config is not None:
                memory, memory_mask, memory_indices = self.init_transformer_memory(self.transformer_config, model, device)
            
            # Reset workers and set evaluation seed
            for worker in self.workers:
                reset_params = self.configs["environment"]["reset_params"].copy()
                reset_params["start-seed"] = seed
                reset_params["num-seeds"] = 1
                worker.child.send(("reset", reset_params))
            # Grab initial observations
            for w, worker in enumerate(self.workers):
                vis, vec = worker.child.recv()
                if vis_obs is not None:
                    vis_obs[w] = vis
                if vec_obs is not None:
                    vec_obs[w] = vec
            
            # Every worker plays its episode
            dones = np.zeros(self.n_workers, dtype=bool)
            worker_steps = [0 for _ in range(self.n_workers)]

            # Store data for video recording
            probs = [[] for worker in self.workers]
            entropies = [[] for worker in self.workers]
            values = [[] for worker in self.workers]
            actions = [[] for worker in self.workers]

            with torch.no_grad():
                while not np.all(dones):
                    # Sample action and send it to the worker if not done
                    for w, worker in enumerate(self.workers):
                        if not dones[w]:
                            # While sampling data for training we feed batches containing all workers,
                            # but as we evaluate entire episodes, we feed one worker at a time
                            vis_obs_batch = torch.tensor(np.expand_dims(vis_obs[w], 0), dtype=torch.float32, device=device) if vis_obs is not None else None
                            vec_obs_batch = torch.tensor(np.expand_dims(vec_obs[w], 0), dtype=torch.float32, device=device) if vec_obs is not None else None

                            # Prepare transformer memory
                            if self.transformer_config is not None:
                                in_memory = memory[w][0, memory_indices[worker_steps[w]]].unsqueeze(0)
                                t = max(0, min(worker_steps[w], self.memory_length - 1))
                                mask = memory_mask[t].unsqueeze(0)
                                indices = memory_indices[worker_steps[w]].unsqueeze(0)
                            else:
                                in_memory = memory[w]

                            # Forward model
                            if self.memory_length == -2:
                                policy, value, new_memory, _ = model(vis_obs_batch, vec_obs_batch, in_memory, mask, indices)
                            else:
                                policy, value, new_memory, _ = model[w](vis_obs_batch, vec_obs_batch, in_memory, mask, indices)

                            # Set memory if used
                            if self.recurrence_config is not None:
                                memory[w] = new_memory
                            if self.transformer_config is not None:
                                memory[w][:, worker_steps[w]] = new_memory

                            _actions = []
                            _probs = []
                            entropy = []
                            # Sample action
                            for action_branch in policy:
                                action = action_branch.sample()
                                _actions.append(action.cpu().data.item())
                                _probs.append(action_branch.probs)
                                entropy.append(action_branch.entropy().item())

                            # Store data for video recording
                            actions[w].append(_actions)
                            probs[w].append(torch.stack(_probs))
                            entropies[w].append(entropy)
                            values[w].append(value.cpu().numpy())

                            # Step environment
                            worker.child.send(("step", _actions))
                            worker_steps[w] += 1

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
                                # record video for this particular worker
                                if self.record_video or self.generate_website:
                                    worker.child.send(("video", None))
                                    trajectory_data = worker.child.recv()
                                    trajectory_data["actions"] = actions[w]
                                    trajectory_data["probs"] = probs[w]
                                    trajectory_data["entropies"] = entropies[w]
                                    trajectory_data["values"] = values[w]
                                    trajectory_data["episode_reward"] = info["reward"]
                                    trajectory_data["seed"] = seed
                                    # Init VideoRecorder
                                    video_recorder = VideoRecorder(self.video_path + "_" + str(w), self.frame_rate)
                                    # Render and serialize video
                                    if self.record_video:
                                        video_recorder.render_video(trajectory_data)
                                    # Generate website
                                    if self.generate_website:
                                        video_recorder.generate_website(trajectory_data, self.configs)
        
            # Seconds needed for a whole update
            time_end = time.time()
            eval_duration = int(time_end - time_start)

        # Return the duration of the evaluation and the raw episode results
        return eval_duration, episode_infos

    def close(self):
        """Closes the Evaluator and destroys all worker."""
        try:
            for worker in self.workers:
                worker.close()
        except:
            pass