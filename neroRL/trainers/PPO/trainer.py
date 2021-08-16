import logging
import os
import torch
import numpy as np
from torch import optim
import time
from typing import Dict, List
from collections import deque
from torch.utils.tensorboard import SummaryWriter
from gym import spaces
from sys import exit
from signal import signal, SIGINT

from neroRL.environments.wrapper import wrap_environment
from neroRL.trainers.PPO.models.actor_critic import create_actor_critic_model
from neroRL.trainers.PPO.buffer import Buffer
from neroRL.trainers.PPO.evaluator import Evaluator
from neroRL.utils.worker import Worker
from neroRL.utils.decay_schedules import polynomial_decay
from neroRL.utils.serialization import save_checkpoint, load_checkpoint

class PPOTrainer():
    """The PPOTrainer is in charge of setting up the whole training loop while utilizing the PPO algorithm based on Schulman et al. 2017."""
    def __init__(self, configs, worker_id, run_id  = "default", low_mem_fix = False, out_path = "./"):
        """Initializes the trainer, the model, the buffer, the evaluator and launches training environments

        Arguments:
            configs {dict} -- The whole set of configurations (e.g. training and environment configs)
            worker_id {int} -- Specifies the offset for the port to communicate with the environment, which is needed for Unity ML-Agents environments (default: {1})
            run_id {string} -- The run_id is used to tag the training runs (directory names to store summaries and checkpoints) (default: {"default"})
            low_mem_fix {bool} -- Determines whethere to do the training/sampling on CPU or GPU. This is necessary for too small GPU memory capacities (default: {False})
        """
        # Handle Ctrl + C event, which aborts and shuts down the training process in a controlled manner
        signal(SIGINT, self._handler)
        # Create directories for storing checkpoints, logs and tensorboard summaries based on the current time and provided run_id
        if not os.path.exists(out_path + "summaries"):
            os.makedirs(out_path + "summaries")
        if not os.path.exists(out_path + "checkpoints"):
            os.makedirs(out_path + "checkpoints")
        if not os.path.exists(out_path + "logs") or not os.path.exists(out_path + "logs/" + run_id):
            os.makedirs(out_path + "logs/" + run_id)
        timestamp = time.strftime("/%Y%m%d-%H%M%S"+ "_" + str(worker_id) + "/")
        self.checkpoint_path = out_path + "checkpoints/" + run_id + timestamp
        os.makedirs(self.checkpoint_path)

        # Setup logger
        logging.basicConfig(level = logging.INFO, handlers=[])
        self.logger = logging.getLogger("train")
        console = logging.StreamHandler()
        console.setFormatter(logging.Formatter("%(asctime)s: %(message)s", "%Y-%m-%d %H:%M:%S"))
        path = out_path + "logs/" + run_id + timestamp[:-1] + ".log"
        logfile = logging.FileHandler(path, mode="w")
        self.logger.addHandler(console)
        self.logger.addHandler(logfile)

        # Determine cuda availability
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Init members
        self.worker_id = worker_id
        self.run_id = run_id
        self.low_mem_fix = low_mem_fix
        if self.low_mem_fix:
            self.mini_batch_device = torch.device("cpu")
        else:
            self.mini_batch_device = self.device
        self.configs = configs
        self.resume_at = configs["trainer"]['resume_at']
        self.gamma = configs["trainer"]['gamma']
        self.lamda = configs["trainer"]['lamda']
        self.updates = configs["trainer"]['updates']
        self.epochs = configs["trainer"]['epochs']
        self.n_workers = configs["trainer"]['n_workers']
        self.worker_steps = configs["trainer"]['worker_steps']
        self.n_mini_batch = configs["trainer"]['n_mini_batch']
        self.recurrence = None if not "recurrence" in configs["model"] else configs["model"]["recurrence"]
        self.lr_schedule = configs["trainer"]['learning_rate_schedule']
        self.beta_schedule = configs["trainer"]['beta_schedule']
        self.cr_schedule = configs["trainer"]['clip_range_schedule']
        self.checkpoint_interval = configs["model"]["checkpoint_interval"]

        self.batch_size = self.n_workers * self.worker_steps
        self.mini_batch_size = self.batch_size // self.n_mini_batch
        assert (self.batch_size % self.n_mini_batch == 0), "Batch Size divided by number of mini batches has a remainder."
        self.writer = SummaryWriter(out_path + "summaries/" + run_id + timestamp)
        self._write_hyperparameters(configs)

        # Start logging the training setup
        self.logger.info("Step 1: Provided config:")
        for key in configs:
            self.logger.info("Step 1: " + str(key) + ":")
            for k, v in configs[key].items():
                self.logger.info("Step 1: " + str(k) + ": " + str(v))

        # Create dummy environment to retrieve the shapes of the observation and action space for further processing
        self.logger.info("Step 2: Creating dummy environment")
        self.dummy_env = wrap_environment(configs["environment"], worker_id)
        visual_observation_space = self.dummy_env.visual_observation_space
        vector_observation_space = self.dummy_env.vector_observation_space
        if isinstance(self.dummy_env.action_space, spaces.Discrete):
            self.action_space_shape = (self.dummy_env.action_space.n,)
        else:
            self.action_space_shape = tuple(self.dummy_env.action_space.nvec)
        self.dummy_env.close()

        self.logger.info("Step 2: Visual Observation Space: " + str(visual_observation_space))
        self.logger.info("Step 2: Vector Observation Space: " + str(vector_observation_space))
        self.logger.info("Step 2: Action Space Shape: " + str(self.action_space_shape))
        self.logger.info("Step 2: Action Names: " + str(self.dummy_env.action_names))

        # Prepare evaluator if configured
        self.eval = configs["evaluation"]["evaluate"]
        self.eval_interval = configs["evaluation"]["interval"]
        if self.eval and self.eval_interval > 0:
            self.logger.info("Step 2b: Initializing evaluator")
            self.evaluator = Evaluator(configs, worker_id, visual_observation_space, vector_observation_space)

        # Instantiate experience/training data buffer
        self.buffer = Buffer(
            self.n_workers, self.worker_steps, self.n_mini_batch,
            visual_observation_space, vector_observation_space,
            self.action_space_shape, self.recurrence,
            self.device, self.mini_batch_device, configs["model"]["share_parameters"])

        # Init model
        self.logger.info("Step 3: Creating model")
        self.model = create_actor_critic_model(configs["model"], visual_observation_space, vector_observation_space,
                                self.action_space_shape, self.recurrence, self.device)

        # Instantiate optimizer
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr_schedule["initial"])

        # Load checkpoint and apply data
        if configs["model"]["load_model"]:
            self.logger.info("Step 3: Loading model from " + configs["model"]["model_path"])
            checkpoint = load_checkpoint(configs["model"]["model_path"])
            self.model.load_state_dict(checkpoint["model_state_dict"])
            if self.recurrence is not None:
                self.model.set_mean_recurrent_cell_states(checkpoint["hxs"], checkpoint["cxs"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            # self.resume_at = checkpoint["update"] + 1

        # Set model to train mode
        self.model.train()

        # Launch workers
        self.logger.info("Step 4: Launching training environments of type " + configs["environment"]["type"])
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

    def run_training(self):
        """Orchestrates the PPO training:
            1. Decays training parameters in relation to training progression
            2. Samples data from current policy
                2a. Computes advantages
            3. Organizes the mini batches
            4. Optimizes policy and value functions
            5. Processes training statistics and results
            6. Evaluates model every n-th update if configured
        """
        if(self.resume_at > 0):
            self.logger.info("Step 5: Resuming training at step " + str(self.resume_at) + " using " + str(self.device) + " . . .")
        else:
            self.logger.info("Step 5: Starting training using " + str(self.device) + " . . .")
        # List that stores the most recent episodes for training statistics
        episode_info = deque(maxlen=100)

        # Training loop
        for update in range(self.resume_at, self.updates):
            self.currentUpdate = update
            time_start = time.time()

            # 1.: Decay hyperparameters polynomially based on the provided config
            learning_rate = polynomial_decay(self.lr_schedule["initial"], self.lr_schedule["final"], self.lr_schedule["max_decay_steps"], self.lr_schedule["power"], update)
            beta = polynomial_decay(self.beta_schedule["initial"], self.beta_schedule["final"], self.beta_schedule["max_decay_steps"], self.beta_schedule["power"], update)
            clip_range = polynomial_decay(self.cr_schedule["initial"], self.cr_schedule["final"], self.cr_schedule["max_decay_steps"], self.cr_schedule["power"], update)

            # 2.: Sample data from each worker for worker steps
            if self.low_mem_fix:
                self.model.cpu() # Sample on CPU
                sample_episode_info = self._sample_training_data(self.mini_batch_device)
            else:
                sample_episode_info = self._sample_training_data(self.device)
            
            # 3.: If a recurrent policy is used, set the mean of the recurrent cell states for future initializations
            if self.recurrence:
                self.model.set_mean_recurrent_cell_states(
                        np.mean(self.buffer.hxs.reshape(self.n_workers * self.worker_steps, self.recurrence["hidden_state_size"]), axis=0),
                        np.mean(self.buffer.cxs.reshape(self.n_workers * self.worker_steps, self.recurrence["hidden_state_size"]), axis=0))

            # 4.: Prepare the sampled data inside the buffer
            self.buffer.prepare_batch_dict()

            # 5.: Train n epochs over the sampled data using mini batches
            if torch.cuda.is_available():
                self.model.cuda() # Train on GPU
            training_stats = self._train_epochs(learning_rate, clip_range, beta)
            training_stats = np.mean(training_stats, axis=0)
            
            # Store recent episode infos
            episode_info.extend(sample_episode_info)
    
            # Measure seconds needed for a whole update
            time_end = time.time()
            update_duration = int(time_end - time_start)

            # Save checkpoint (update, model, optimizer, configs)
            if update % self.checkpoint_interval == 0 or update == (self.updates - 1):
                save_checkpoint(self.checkpoint_path + self.run_id + "-" + str(update) + ".pt",
                                update,
                                self.model.state_dict(),
                                self.optimizer.state_dict(),
                                self.model.mean_hxs if self.recurrence is not None else None,
                                self.model.mean_cxs if self.recurrence is not None else None,
                                self.configs)

            # 5.: Write training statistics to console
            episode_result = self._process_episode_info(episode_info)
            if episode_result:
                self.logger.info("{:4} sec={:2} reward={:.2f} std={:.2f} length={:.1f} std={:.2f} loss={:3f} entropy={:.3f} value={:3f} std={:.3f} advantage={:.3f} std={:.3f} sequence length={:3}".format(
                    update, update_duration, episode_result["reward_mean"], episode_result["reward_std"], episode_result["length_mean"], episode_result["length_std"],
                    training_stats[2], training_stats[3], np.mean(self.buffer.values), np.std(self.buffer.values),
                    np.mean(self.buffer.advantages), np.std(self.buffer.advantages), self.buffer.actual_sequence_length))
            else:
                self.logger.info("{:4} sec={:2} loss={:3f} entropy={:.3f} value={:3f} std={:.3f} advantage={:.3f} std={:.3f} sequence length={:3}".format(
                    update, update_duration, training_stats[2], training_stats[3], np.mean(self.buffer.values),
                    np.std(self.buffer.values), np.mean(self.buffer.advantages), np.std(self.buffer.advantages), self.buffer.actual_sequence_length))

            # 6.: Evaluate model
            if self.eval:
                if update % self.eval_interval == 0 or update == (self.updates - 1):
                    eval_duration, eval_episode_info = self.evaluator.evaluate(self.model, self.device)
                    episode_result = self._process_episode_info(eval_episode_info)
                    self.logger.info("eval: sec={:3} reward={:.2f} length={:.1f}".format(
                        eval_duration, episode_result["reward_mean"], episode_result["length_mean"]))
                    self._write_eval_summary(update, episode_result)
            
            # Write training statistics to tensorboard
            self._write_training_summary(update, training_stats, episode_result, learning_rate, clip_range, beta)

    def _sample_training_data(self, device):
        """Sample data (batch) with current policy from all workers for worker_steps.
        At the end the advantages are computed.
        
        Arguments:
            device {torch.device} -- The to be used device for sampling training data

        Returns:
            episode_infos {list} -- Results of completed episodes
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

    def _train_epochs(self, learning_rate: float, clip_range: float, beta: float):
        """Trains several PPO epochs over one batch of data while dividing the batch into mini batches.
        
        Arguments:
            learning_rate {float} -- The current learning rate
            clip_range {float} -- The current clip range
            beta {float} -- The current entropy bonus coefficient
            
        Returns:
            {numpy.ndarray} -- Mean training statistics of one training epoch"""
        train_info = []

        for _ in range(self.epochs):
            # Retrieve the to be trained mini_batches via a generator
            # Use the recurrent mini batch generator for training a recurrent policy
            if self.recurrence is not None:
                mini_batch_generator = self.buffer.recurrent_mini_batch_generator()
            else:
                mini_batch_generator = self.buffer.mini_batch_generator()
            for mini_batch in mini_batch_generator:
                res = self._train_mini_batch(learning_rate=learning_rate,
                                         clip_range=clip_range,
                                         beta = beta,
                                         samples=mini_batch)
                train_info.append(res)
        # Return the mean of the training statistics
        return train_info

    def _train_mini_batch(self, samples, learning_rate, clip_range, beta):
        """ Optimizes the policy based on the PPO algorithm

        Arguments:
            samples {dict} -- The sampled mini-batch to optimize the model
            learning_rate {float} -- The to be used learning rate
            clip_range {float} -- The to be used clip range
            beta {float} -- The to be used entropy coefficient
        
        Returns:
            training_stats {list} -- Losses, entropy, kl-divergence and clip fraction
        """
        # Retrieve sampled recurrent cell states to feed the model
        recurrent_cell = None
        if self.recurrence is not None:
            if self.recurrence["layer_type"] == "gru":
                recurrent_cell = samples["hxs"].unsqueeze(0)
            elif self.recurrence["layer_type"] == "lstm":
                recurrent_cell = (samples["hxs"].unsqueeze(0), samples["cxs"].unsqueeze(0))
        
        policy, value, _ = self.model(samples['vis_obs'] if self.vis_obs is not None else None,
                                    samples['vec_obs'] if self.vec_obs is not None else None,
                                    recurrent_cell,
                                    self.device,
                                    self.buffer.actual_sequence_length)
        
        # Policy Loss
        # Retreive and process log_probs from each policy branch
        log_probs = []
        for i, policy_branch in enumerate(policy):
            log_probs.append(policy_branch.log_prob(samples['actions'][:, i]))
        log_probs = torch.stack(log_probs, dim=1)

        # Compute surrogates
        normalized_advantage = (samples["advantages"] - samples["advantages"].mean()) / (samples["advantages"].std() + 1e-8)
        # Repeat is necessary for multi-discrete action spaces
        normalized_advantage = normalized_advantage.unsqueeze(1).repeat(1, len(self.action_space_shape))
        ratio = torch.exp(log_probs - samples['log_probs'])
        surr1 = ratio * normalized_advantage
        surr2 = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * normalized_advantage
        policy_loss = torch.min(surr1, surr2)
        policy_loss = PPOTrainer._masked_mean(policy_loss, samples["loss_mask"])

        # Value
        sampled_return = samples['values'] + samples['advantages']
        clipped_value = samples['values'] + (value - samples['values']).clamp(min=-clip_range, max=clip_range)
        vf_loss = torch.max((value - sampled_return) ** 2, (clipped_value - sampled_return) ** 2)
        vf_loss = PPOTrainer._masked_mean(vf_loss, samples["loss_mask"])
        vf_loss = .25 * vf_loss

        # Entropy Bonus
        entropies = []
        for policy_branch in policy:
            entropies.append(policy_branch.entropy())
        entropy_bonus = PPOTrainer._masked_mean(torch.stack(entropies, dim=1).sum(1).reshape(-1), samples["loss_mask"])

        # Complete loss
        loss = -(policy_loss - vf_loss + beta * entropy_bonus)

        # Compute gradients
        for pg in self.optimizer.param_groups:
            pg['lr'] = learning_rate
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
        self.optimizer.step()

        # Monitor training statistics
        approx_kl = PPOTrainer._masked_mean((torch.exp(ratio) - 1) - ratio, samples["loss_mask"])
        clip_fraction = (abs((ratio - 1.0)) > clip_range).type(torch.FloatTensor).mean()

        return [policy_loss.cpu().data.numpy(),
                vf_loss.cpu().data.numpy(),
                loss.cpu().data.numpy(),
                entropy_bonus.cpu().data.numpy(),
                approx_kl.cpu().data.numpy(),
                clip_fraction.cpu().data.numpy()]

    @staticmethod
    def _masked_mean(tensor:torch.Tensor, mask:torch.Tensor) -> torch.Tensor:
            """
            Returns the mean of the tensor but ignores the values specified by the mask.
            This is used for masking out the padding of the loss functions.

            Args:
                tensor {Tensor} -- The to be masked tensor
                mask {Tensor} -- The mask that is used to mask out padded values of a loss function

            Returns:
                {Tensor} -- Returns the mean of the masked tensor.
            """
            return (tensor.T * mask).sum() / torch.clamp((torch.ones_like(tensor.T) * mask).float().sum(), min=1.0)

    def _write_training_summary(self, update, training_stats, episode_result, learning_rate, clip_range, beta):
        """Writes to an event file based on the run-id argument."""
        if episode_result:
            for key in episode_result:
                if "std" not in key:
                    self.writer.add_scalar("episode/" + key, episode_result[key], update)
        self.writer.add_scalar("losses/loss", training_stats[2], update)
        self.writer.add_scalar("losses/policy_loss", training_stats[0], update)
        self.writer.add_scalar("losses/value_loss", training_stats[1], update)
        self.writer.add_scalar("other/entropy", training_stats[3], update)
        self.writer.add_scalar("other/clip_fraction", training_stats[5], update)
        self.writer.add_scalar("other/sequence_length", self.buffer.actual_sequence_length, update)
        self.writer.add_scalar("episode/value_mean", np.mean(self.buffer.values), update)
        self.writer.add_scalar("episode/advantage_mean", np.mean(self.buffer.advantages), update)
        self.writer.add_scalar("decay/learning_rate", learning_rate, update)
        self.writer.add_scalar("decay/clip_range", clip_range, update)
        self.writer.add_scalar("decay/beta", beta, update)

    def _write_eval_summary(self, update, episode_result):
        """Writes to an event file based on the run-id argument."""
        if episode_result:
            for key in episode_result:
                if "std" not in key:
                    self.writer.add_scalar("evaluation/" + key, episode_result[key], update)

    def _write_hyperparameters(self, configs):
        """Writes hyperparameters to tensorboard"""
        for key, value in configs.items():
            if isinstance(value, dict):
                for k, v in value.items():
                    self.writer.add_text("Hyperparameters", k + " " + str(v))
            else:
                self.writer.add_text("Hyperparameters", key + " " + str(value))

    @staticmethod
    def _process_episode_info(episode_info):
        """Extracts the mean and std of completed episodes. At minimum the episode length and the collected reward is available."""
        result = {}
        if len(episode_info) > 0:
            keys = episode_info[0].keys()
            for key in keys:
                # skip seed
                if key == "seed":
                    continue
                result[key + "_mean"] = np.mean([info[key] for info in episode_info])
                result[key + "_min"] = np.min([info[key] for info in episode_info])
                result[key + "_max"] = np.max([info[key] for info in episode_info])
                result[key + "_std"] = np.std([info[key] for info in episode_info])
        return result

    def close(self):
        """Closes the environment and destroys the workers"""
        self.logger.info("Terminate: Closing dummy ennvironment . . .")
        try:
            self.dummy_env.close()
        except:
            pass

        self.logger.info("Terminate: Closing Summary Writer . . .")
        try:
            self.writer.close()
        except:
            pass

        self.logger.info("Terminate: Shutting down workers . . .")
        try:
            for worker in self.workers:
                worker.child.send(("close", None))
        except:
            pass

        if self.eval:
            self.logger.info("Terminate: Closing evaluator")
            try:
                self.evaluator.close()
            except:
                    pass
        
        try:
            if self.currentUpdate > 0:
                self.logger.info("Terminate: Saving model . . .")
                try:
                        save_checkpoint(self.checkpoint_path + self.run_id + "-" + str(self.currentUpdate) + ".pt",
                                        self.currentUpdate,
                                        self.model.state_dict(),
                                        self.optimizer.state_dict(),
                                        self.model.mean_hxs if self.recurrence is not None else None,
                                        self.model.mean_cxs if self.recurrence is not None else None,
                                        self.configs)
                        self.logger.info("Terminate: Saved model to: " + self.checkpoint_path + self.run_id + "-" + str(self.currentUpdate) + ".pt")
                except:
                    pass
        except:
            pass

    def _handler(self, signal_received, frame):
        """Invoked by the Ctrl-C event, the trainer is being closed and the python program is being exited."""
        self.logger.info("Terminate: Training aborted . . .")
        self.close()
        exit(0)
