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
from neroRL.trainers.PPO.otc_model import OTCModel
from neroRL.trainers.PPO.buffer import Buffer
from neroRL.trainers.PPO.evaluator import Evaluator
from neroRL.utils.worker import Worker
from neroRL.utils.seeder import Seeder
from neroRL.utils.decay_schedules import polynomial_decay

class PPOTrainer():
    """The PPOTrainer is in charge of setting up the whole training loop while utilizing the PPO algorithm based on Schulman et al. 2017."""
    def __init__(self, configs, worker_id, run_id  = "default", low_mem_fix = False):
        """Initializes the trainer, the model, the buffer, the evaluator and launches training environments

        Arguments:
            configs {dict} -- The whole set of configurations (e.g. training and environment configs)
            worker_id {int} -- Specifies the offset for the port to communicate with the environment, which is needed for Unity ML-Agents environments (default: {1})
            run_id {string} -- The run_id is used to tag the training runs (directory names to store summaries and checkpoints) (default: {"default"})
            low_mem_fix {bool} -- Determines whethere to do the training/sampling on cpu or gpu. This is necessary for too small GPU memory capacities (default: {False})
        """
        # Handle Ctrl + C event, which aborts and shuts down the training process in a controlled manner
        signal(SIGINT, self.handler)
        # Create directories for storing checkpoints, models and tensorboard summaries based on the current time and provided run_id
        if not os.path.exists("summaries"):
            os.makedirs("summaries")
        if not os.path.exists("checkpoints"):
            os.makedirs("checkpoints")
        timestamp = time.strftime("/%Y%m%d-%H%M%S"+ "_" + str(worker_id) + "/")
        self.checkpoint_path = "checkpoints/" + run_id + timestamp
        os.makedirs(self.checkpoint_path)

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
        self.resume_at = configs["trainer"]['resume_at']
        self.gamma = configs["trainer"]['gamma']
        self.lamda = configs["trainer"]['lamda']
        self.updates = configs["trainer"]['updates']
        self.epochs = configs["trainer"]['epochs']
        self.n_workers = configs["trainer"]['n_workers']
        self.worker_steps = configs["trainer"]['worker_steps']
        self.n_mini_batch = configs["trainer"]['n_mini_batch']
        self.use_recurrent = configs["model"]["use_recurrent"]
        self.hidden_state_size = configs["model"]["hidden_state_size"]
        self.lr_schedule = configs["trainer"]['learning_rate_schedule']
        self.beta_schedule = configs["trainer"]['beta_schedule']
        self.cr_schedule = configs["trainer"]['clip_range_schedule']

        self.batch_size = self.n_workers * self.worker_steps
        self.mini_batch_size = self.batch_size // self.n_mini_batch
        assert (self.batch_size % self.n_mini_batch == 0), "Batch Size divided by number of mini batches has a remainder."
        self.writer = SummaryWriter("summaries/" + run_id + timestamp)
        self.write_hyperparameters(configs["trainer"])

        self.checkpoint_interval = configs["model"]["checkpoint_interval"]

        print("Step 1: Provided config:")
        for key in configs:
            print("Step 1: " + str(key) + ":")
            for k, v in configs[key].items():
                print("Step 1: " + str(k) + ": " + str(v))

        print("Step 2: Creating dummy environment")
        # Create dummy environment to retrieve the shapes of the observation and action space for further processing
        self.dummy_env = wrap_environment(configs["environment"], worker_id)
        
        visual_observation_space = self.dummy_env.visual_observation_space
        vector_observation_space = self.dummy_env.vector_observation_space
        if isinstance(self.dummy_env.action_space, spaces.Discrete):
            self.action_space_shape = (self.dummy_env.action_space.n,)
        else:
            self.action_space_shape = tuple(self.dummy_env.action_space.nvec)
        self.dummy_env.close()

        print("Step 2: Visual Observation Space: " + str(visual_observation_space))
        print("Step 2: Vector Observation Space: " + str(vector_observation_space))
        print("Step 2: Action Space Shape: " + str(self.action_space_shape))
        print("Step 2: Action Names: " + str(self.dummy_env.action_names))

        # Prepare evaluator if configured
        self.eval = configs["evaluation"]["evaluate"]
        self.eval_interval = configs["evaluation"]["interval"]
        if self.eval:
            print("Step 2b: Initializing evaluator")
            self.evaluator = Evaluator(configs["evaluation"], configs["environment"], worker_id, visual_observation_space, vector_observation_space)

        # Build or load model
        if not configs["model"]["load_model"]:
            print("Step 3: Creating model")
            self.model = OTCModel(configs["model"], visual_observation_space,
                                    vector_observation_space, self.action_space_shape,
                                    self.use_recurrent, self.hidden_state_size).to(self.device)
        else:
            print("Step 3: Loading model from " + configs["model"]["model_path"])
            self.model = torch.load(configs["model"]["model_path"]).to(self.device)
        self.model.train()

        # Instantiate optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr_schedule["initial"])
        # Instantiate experience/training data buffer
        self.buffer = Buffer(
            self.n_workers,
            self.worker_steps,
            self.n_mini_batch,
            visual_observation_space,
            vector_observation_space,
            self.action_space_shape,
            self.use_recurrent,
            self.hidden_state_size,
            self.device,
            self.mini_batch_device)

        # Instantiate seeder
        self.seeder = Seeder(configs["environment"]["reset_params"]["start-seed"], configs["environment"]["reset_params"]["num-seeds"], random_seed= not configs["environment"]["use_seeder"])

        # Launch workers
        print("Step 4: Launching training environments of type " + configs["environment"]["type"])
        self.workers = []
        for i in range(self.n_workers):
            id = worker_id + 200 + i
            self.workers.append(Worker(configs["environment"], id))

        # Setup initial observations
        if visual_observation_space is not None:
            self.vis_obs = np.zeros((self.n_workers,) + visual_observation_space.shape, dtype=np.float32)
        else:
            self.vis_obs = None
        if vector_observation_space is not None:
            self.vec_obs = np.zeros((self.n_workers,) + vector_observation_space, dtype=np.float32)
        else:
            self.vec_obs = None

        # Setup initial hidden states
        if self.use_recurrent:
            self.hidden_state = torch.zeros((self.n_workers, self.hidden_state_size), dtype=torch.float32, device=self.device)
        else:
            self.hidden_state = None

        # Reset workers
        for worker in self.workers:
            worker.child.send(("reset", {"start-seed": self.seeder.sample_seed(), "num-seeds": 1}))
        # Grab initial observations
        for i, worker in enumerate(self.workers):
            vis_obs, vec_obs = worker.child.recv()
            if self.vis_obs is not None:
                self.vis_obs[i] = vis_obs
            if self.vec_obs is not None:
                self.vec_obs[i] = vec_obs

    @staticmethod
    def _normalize(adv: np.ndarray):
        """Normalizes the advantage

        Arguments:
            adv {numpy.ndarray} -- The to be normalized advantage

        Returns:
            (adv - adv.mean()) / (adv.std() + 1e-8) {np.ndarray} -- The normalized advantage
        """
        return (adv - adv.mean()) / (adv.std() + 1e-8)

    def sample(self, device) -> (Dict[str, np.ndarray], List):
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
            # Save the initial observations and hidden states
            if self.vis_obs is not None:
                self.buffer.vis_obs[:, t] = self.vis_obs
            if self.vec_obs is not None:
                self.buffer.vec_obs[:, t] = self.vec_obs
            if self.use_recurrent:
                self.buffer.hidden_states[:, t] = self.hidden_state
            
            # Gradients can be omitted for sampling data
            with torch.no_grad():
                # Forward the model to retrieve the policy (making decisions), the states' value of the value function and the recurrent hidden states (if available)
                policy, value, self.hidden_state = self.model(self.vis_obs, self.vec_obs, self.hidden_state, device)
                self.buffer.values[:, t] = value.cpu().data.numpy()

                # Sample actions from each individual branch
                actions = []
                neg_log_pis = []
                for action_branch in policy:
                    action = action_branch.sample()
                    actions.append(action.cpu().data.numpy())
                    neg_log_pis.append(-action_branch.log_prob(action).cpu().data.numpy())
                actions = np.transpose(actions)
                neg_log_pis = np.transpose(neg_log_pis)
                self.buffer.actions[:, t] = actions
                self.buffer.neg_log_pis[:, t] = neg_log_pis

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
                    episode_infos.append(info)
                    # Send performance data to seeder
                    self.seeder.add_seed_result(info["seed"], info["reward"])
                    # Reset agent (potential interface for providing reset parameters) and sample a new seed for the envronment
                    worker.child.send(("reset", {"start-seed": self.seeder.sample_seed(), "num-seeds": 1}))
                    # Get data from reset
                    vis_obs, vec_obs = worker.child.recv()
                    if self.vis_obs is not None:
                        self.vis_obs[w] = vis_obs
                    if self.vec_obs is not None:
                        self.vec_obs[w] = vec_obs

        # Calculate advantages
        _, last_value, _ = self.model(self.vis_obs, self.vec_obs, self.hidden_state, device)
        self.buffer.calc_advantages(last_value.cpu().data.numpy(), self.gamma, self.lamda)

        return episode_infos

    def train_mini_batch(self, samples, learning_rate, clip_range, beta):
        """ Optimizes the policy based on the PPO algorithm

        Arguments:
            samples {dict} -- The sampled mini-batch to optimize the model
            learning_rate {float} -- The to be used learning rate
            clip_range {float} -- The to be used clip range
            beta {float} -- The to be used entropy coefficient
        
        Returns:
            training_stats {list} -- Losses, entropy, kl-divergence and clip fraction
        """
        sampled_return = samples['values'] + samples['advantages']
        sampled_normalized_advantage = PPOTrainer._normalize(samples['advantages']).unsqueeze(1).repeat(1, len(self.action_space_shape))
        policy, value, _ = self.model(samples['vis_obs'] if self.vis_obs is not None else None,
                                    samples['vec_obs'] if self.vec_obs is not None else None,
                                    samples['hidden_states'] if self.use_recurrent else None,
                                    self.device)
        
        # Policy
        neg_log_pis = []
        for i, policy_branch in enumerate(policy):
            neg_log_pis.append(-policy_branch.log_prob(samples['actions'][:, i]))
        neg_log_pis = torch.stack(neg_log_pis, dim=1)

        ratio: torch.Tensor = torch.exp(samples['neg_log_pis'] - neg_log_pis)
        clipped_ratio = ratio.clamp(min=1.0 - clip_range,
                                    max=1.0 + clip_range)

        policy_loss = torch.min(ratio * sampled_normalized_advantage,
                                  clipped_ratio * sampled_normalized_advantage)
        policy_loss = policy_loss.mean()

        # Value
        clipped_value = samples['values'] + (value - samples['values']).clamp(min=-clip_range,
                                                                      max=clip_range)
        vf_loss = torch.max((value - sampled_return) ** 2, (clipped_value - sampled_return) ** 2)
        vf_loss = 0.5 * vf_loss.mean()

        # Entropy Bonus
        entropies = []
        for policy_branch in policy:
            entropies.append(policy_branch.entropy())
        entropy_bonus = torch.stack(entropies, dim=1).sum(1).reshape(-1).mean()

        # Complete loss
        loss: torch.Tensor = -(policy_loss - 0.5 * vf_loss + beta * entropy_bonus)

        # Compute gradients
        for pg in self.optimizer.param_groups:
            pg['lr'] = learning_rate
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
        self.optimizer.step()

        # Monitor training statistics
        approx_kl_divergence = .5 * ((neg_log_pis - samples['neg_log_pis']) ** 2).mean()
        clip_fraction = (abs((ratio - 1.0)) > clip_range).type(torch.FloatTensor).mean()

        return [policy_loss,
                vf_loss,
                loss,
                entropy_bonus,
                approx_kl_divergence,
                clip_fraction]

    def train_epochs(self, learning_rate: float, clip_range: float, beta: float):
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
            if self.use_recurrent:
                mini_batch_generator = self.buffer.recurrent_mini_batch_generator()
            else:
                mini_batch_generator = self.buffer.mini_batch_generator()
            for mini_batch in mini_batch_generator:
                res = self.train_mini_batch(learning_rate=learning_rate,
                                         clip_range=clip_range,
                                         beta = beta,
                                         samples=mini_batch)
                train_info.append(res)
        # Return the mean of the training statistics
        return np.mean(train_info, axis=0)

    def run_training_loop(self):
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
            print("Step 5: Resuming training at step " + str(self.resume_at) + " using " + str(self.device) + " . . .")
        else:
            print("Step 5: Starting training using " + str(self.device) + " . . .")
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

            # 2., 2a.: Sample data from each worker for worker steps
            if self.low_mem_fix:
                self.model.cpu() # Sample on CPU
                sample_episode_info = self.sample(self.mini_batch_device)
            else:
                sample_episode_info = self.sample(self.device)

            # Update seeder
            self.seeder.update_logits()
            # print((self.seeder.seed_logits * 100.))

            # 3.: Prepare the sampled data inside the buffer
            self.buffer.prepare_batch_dict()

            # 4.: Train n epochs over the sampled data using mini batches
            if torch.cuda.is_available():
                self.model.cuda() # Train on GPU
            training_stats = self.train_epochs(learning_rate, clip_range, beta)

            # Store recent episode infos
            episode_info.extend(sample_episode_info)
    
            # Seconds needed for a whole update
            time_end = time.time()
            update_duration = int(time_end - time_start)

            # Save model
            if update % self.checkpoint_interval == 0 or update == (self.updates - 1):
                torch.save(self.model, self.checkpoint_path + self.run_id + "-" + str(update) + ".pt")

            # 5.: Write training statistics to console
            episode_result = self._process_episode_info(episode_info)
            if episode_result:
                print("{:4} sec={:3} reward={:.2f} std={:.2f} length={:.1f} std={:.2f} loss={:3f} entropy={:.3f} value={:3f} std={:.3f} advantage={:.3f} std={:.3f}".format(
                    update, update_duration, episode_result["reward_mean"], episode_result["reward_std"], episode_result["length_mean"], episode_result["length_std"],
                    training_stats[2], training_stats[3], np.mean(self.buffer.values), np.std(self.buffer.values), np.mean(self.buffer.advantages), np.std(self.buffer.advantages)))
            else:
                print("{:4} sec={:3} loss={:3f} entropy={:.3f} value={:3f} std={:.3f} advantage={:.3f} std={:.3f}".format(
                    update, update_duration, training_stats[2], training_stats[3], np.mean(self.buffer.values),
                    np.std(self.buffer.values), np.mean(self.buffer.advantages), np.std(self.buffer.advantages)))

            # 6.: Evaluate model
            if self.eval:
                if update % self.eval_interval == 0 or update == self.updates:
                    eval_duration, eval_episode_info = self.evaluator.evaluate(self.model, self.device)
                    episode_result = self._process_episode_info(eval_episode_info)
                    print("eval: sec={:3} reward={:.2f} length={:.1f}".format(
                        eval_duration, episode_result["reward_mean"], episode_result["length_mean"]))
                    self.write_eval_summary(update, episode_result)
            
            # Write training statistics to tensorboard
            self.write_training_summary(update, training_stats, episode_result, learning_rate, clip_range, beta)

    def write_training_summary(self, update, training_stats, episode_result, learning_rate, clip_range, beta):
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
        self.writer.add_scalar("episode/value_mean", np.mean(self.buffer.values), update)
        self.writer.add_scalar("episode/advantage_mean", np.mean(self.buffer.advantages), update)
        self.writer.add_scalar("decay/learning_rate", learning_rate, update)
        self.writer.add_scalar("decay/clip_range", clip_range, update)
        self.writer.add_scalar("decay/beta", beta, update)

    def write_eval_summary(self, update, episode_result):
        """Writes to an event file based on the run-id argument."""
        if episode_result:
            for key in episode_result:
                if "std" not in key:
                    self.writer.add_scalar("evaluation/" + key, episode_result[key], update)

    def write_hyperparameters(self, config):
        """Writes hyperparameters to tensorboard"""
        for key, value in config.items():
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
        print("Terminate: Closing dummy ennvironment . . .")
        try:
            self.dummy_env.close()
        except:
            pass

        print("Terminate: Closing Summary Writer . . .")
        try:
            self.writer.close()
        except:
            pass

        print("Terminate: Shutting down workers . . .")
        try:
            for worker in self.workers:
                worker.child.send(("close", None))
        except:
            pass

        if self.eval:
            print("Terminate: Closing evaluator")
            try:
                self.evaluator.close()
            except:
                    pass
        
        try:
            if self.currentUpdate > 0:
                print("Terminate: Saving model . . .")
                try:
                        torch.save(self.model, self.checkpoint_path + self.run_id + "-" + str(self.currentUpdate - 1) + ".pt")
                        print("Terminate: Saved model to: " + self.checkpoint_path + self.run_id + "-" + str(self.currentUpdate - 1) + ".pt")
                except:
                    pass
        except:
            pass

    def handler(self, signal_received, frame):
        """Invoked by the Ctrl-C event, the trainer is being closed and the python program is being exited."""
        print("Terminate: Training aborted . . .")
        self.close()
        exit(0)
