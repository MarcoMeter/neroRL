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
from neroRL.sampler.trajectory_sampler import TrajectorySampler
from neroRL.trainers.PPO.models.actor_critic import create_actor_critic_model
from neroRL.trainers.PPO.buffer import Buffer
from neroRL.trainers.PPO.evaluator import Evaluator
from neroRL.utils.decay_schedules import polynomial_decay
from neroRL.utils.utils import masked_mean
from neroRL.utils.serialization import save_checkpoint, load_checkpoint

class BaseTrainer():
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
        self.visual_observation_space = self.dummy_env.visual_observation_space
        self.vector_observation_space = self.dummy_env.vector_observation_space
        if isinstance(self.dummy_env.action_space, spaces.Discrete):
            self.action_space_shape = (self.dummy_env.action_space.n,)
        else:
            self.action_space_shape = tuple(self.dummy_env.action_space.nvec)
        self.dummy_env.close()

        self.logger.info("Step 2: Visual Observation Space: " + str(self.visual_observation_space))
        self.logger.info("Step 2: Vector Observation Space: " + str(self.vector_observation_space))
        self.logger.info("Step 2: Action Space Shape: " + str(self.action_space_shape))
        self.logger.info("Step 2: Action Names: " + str(self.dummy_env.action_names))

        # Prepare evaluator if configured
        self.eval = configs["evaluation"]["evaluate"]
        self.eval_interval = configs["evaluation"]["interval"]
        if self.eval and self.eval_interval > 0:
            self.logger.info("Step 2b: Initializing evaluator")
            self.evaluator = Evaluator(configs, worker_id, self.visual_observation_space, self.vector_observation_space)

        # Instantiate experience/training data buffer
        self.buffer = Buffer(
            self.n_workers, self.worker_steps, self.n_mini_batch,
            self.visual_observation_space, self.vector_observation_space,
            self.action_space_shape, self.recurrence,
            self.device, self.mini_batch_device, configs["model"]["share_parameters"])

        # Init model
        self.logger.info("Step 3: Creating model")
        self.model = create_actor_critic_model(configs["model"], self.visual_observation_space, self.vector_observation_space,
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

        # Setup Sampler
        self.logger.info("Step 4: Launching training environments of type " + configs["environment"]["type"])
        self.sampler = TrajectorySampler(configs, worker_id, self.visual_observation_space, self.vector_observation_space,
                                        self.model, self.buffer, self.device, self.mini_batch_device)

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
            # Apply learning rate
            for pg in self.optimizer.param_groups:
                pg["lr"] = learning_rate

            # 2.: Sample data from each worker for worker steps
            if self.low_mem_fix:
                self.model.cpu() # Sample on CPU
                sample_device = self.mini_batch_device
            else:
                sample_device = self.device

            # 3.: Calculate advantages
            sample_episode_info = self.sampler.sample(self.mini_batch_device)
            _, last_value, _ = self.model(self.sampler.last_vis_obs(), self.sampler.last_vec_obs(), self.sampler.last_recurrent_cell(), sample_device)
            self.buffer.calc_advantages(last_value.cpu().data.numpy(), self.gamma, self.lamda)
            
            # 4.: If a recurrent policy is used, set the mean of the recurrent cell states for future initializations
            if self.recurrence:
                self.model.set_mean_recurrent_cell_states(
                        np.mean(self.buffer.hxs.reshape(self.n_workers * self.worker_steps, *self.buffer.hxs.shape[2:]), axis=0),
                        np.mean(self.buffer.cxs.reshape(self.n_workers * self.worker_steps, *self.buffer.cxs.shape[2:]), axis=0))

            # 5.: Prepare the sampled data inside the buffer
            self.buffer.prepare_batch_dict()

            # 6.: Train n epochs over the sampled data using mini batches
            if torch.cuda.is_available():
                self.model.cuda() # Train on GPU
            training_stats = self.train(clip_range, beta)
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
                    evaluation_result = self._process_episode_info(eval_episode_info)
                    self.logger.info("eval: sec={:3} reward={:.2f} length={:.1f}".format(
                        eval_duration, evaluation_result["reward_mean"], evaluation_result["length_mean"]))
                    self._write_eval_summary(update, evaluation_result)
            
            # Write training statistics to tensorboard
            self._write_training_summary(update, training_stats, episode_result, learning_rate, clip_range, beta)

    def train(self, learning_rate, clip_range, beta):
        raise NotImplementedError

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
        self.writer.add_scalar("other/kl_divergence", training_stats[4], update)
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

        self.logger.info("Terminate: Shutting down sampler . . .")
        try:
            self.sampler.close()
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
