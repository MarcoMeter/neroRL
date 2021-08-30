import torch
import numpy as np
import time
from collections import deque

from gym import spaces
from sys import exit
from signal import signal, SIGINT

from neroRL.environments.wrapper import wrap_environment
from neroRL.sampler.trajectory_sampler import TrajectorySampler
from neroRL.sampler.buffer import Buffer
from neroRL.evaluator import Evaluator
from neroRL.utils.monitor import Monitor
from neroRL.utils.monitor import Tag

class BaseTrainer():
    """The BaseTrainer is in charge of setting up the whole training loop of a policy gradient based algorithm."""
    def __init__(self, configs, worker_id, run_id  = "default", out_path = "./"):
        """Initializes the trainer, the model, the buffer, the evaluator and the training data sampler

        Arguments:
            configs {dict} -- The whole set of configurations (e.g. training and environment configs)
            worker_id {int} -- Specifies the offset for the port to communicate with the environment, which is needed for Unity ML-Agents environments (default: {1})
            run_id {string} -- The run_id is used to tag the training runs (directory names to store summaries and checkpoints) (default: {"default"})
            out_path {str} -- Determines the target directory for saving summaries, logs and model checkpoints. (default: "./")
        """
        # Handle Ctrl + C event, which aborts and shuts down the training process in a controlled manner
        signal(SIGINT, self._handler)

        # Determine cuda availability
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create a monitor that is used for logging and monitoring training statistics (tensorboard)
        self.monitor = Monitor(out_path, run_id, worker_id)
        # Add hyperparameters to the tensorboard summary
        self.monitor.write_hyperparameters(configs)

        # Init members
        self.run_id = run_id
        self.configs = configs
        self.resume_at = configs["trainer"]["resume_at"]
        self.gamma = configs["trainer"]["gamma"]
        self.lamda = configs["trainer"]["lamda"]
        self.updates = configs["trainer"]["updates"]
        self.n_workers = configs["sampler"]["n_workers"]
        self.worker_steps = configs["sampler"]["worker_steps"]
        self.recurrence = None if not "recurrence" in configs["model"] else configs["model"]["recurrence"]
        self.checkpoint_interval = configs["model"]["checkpoint_interval"]

        # Start logging the training setup
        self.monitor.log("Step 1: Provided config:")
        for key in configs:
            self.monitor.log("Step 1: " + str(key) + ":")
            for k, v in configs[key].items():
                self.monitor.log("Step 1: " + str(k) + ": " + str(v))

        # Create dummy environment to retrieve the shapes of the observation and action space for further processing
        self.monitor.log("Step 2: Creating dummy environment")
        self.dummy_env = wrap_environment(configs["environment"], worker_id)
        self.visual_observation_space = self.dummy_env.visual_observation_space
        self.vector_observation_space = self.dummy_env.vector_observation_space
        if isinstance(self.dummy_env.action_space, spaces.Discrete):
            self.action_space_shape = (self.dummy_env.action_space.n,)
        else:
            self.action_space_shape = tuple(self.dummy_env.action_space.nvec)
        self.dummy_env.close()

        self.monitor.log("Step 2: Visual Observation Space: " + str(self.visual_observation_space))
        self.monitor.log("Step 2: Vector Observation Space: " + str(self.vector_observation_space))
        self.monitor.log("Step 2: Action Space Shape: " + str(self.action_space_shape))
        self.monitor.log("Step 2: Action Names: " + str(self.dummy_env.action_names))

        # Prepare evaluator if configured
        self.eval = configs["evaluation"]["evaluate"]
        self.eval_interval = configs["evaluation"]["interval"]
        if self.eval and self.eval_interval > 0:
            self.monitor.log("Step 2b: Initializing evaluator")
            self.evaluator = Evaluator(configs, worker_id, self.visual_observation_space, self.vector_observation_space)

        # Instantiate experience/training data buffer
        self.buffer = Buffer(
            self.n_workers, self.worker_steps,self.visual_observation_space, 
            self.vector_observation_space, self.action_space_shape, self.recurrence,
            self.device, configs["trainer"]["share_parameters"])

        # Init model
        self.monitor.log("Step 3: Creating model")
        self.model = self.create_model()

        # Load checkpoint and apply data
        if configs["model"]["load_model"]:
            self._load_checkpoint()

        # Set model to train mode
        self.model.train()

        # Setup Sampler
        self.monitor.log("Step 4: Launching training environments of type " + configs["environment"]["type"])
        self.sampler = TrajectorySampler(configs, worker_id, self.visual_observation_space, self.vector_observation_space,
                                        self.model, self.buffer, self.device)

    def run_training(self):
        """Orchestrates the policy gradient based training:
            1. Decays training parameters in relation to training progression
            2. Samples data from current policy
            3. Computes advantages
            4. If a recurrent policy is used, set the mean of the recurrent cell states for future initializations
            5. Organizes the mini batches
            6. Optimizes policy and value functions
            7. Processes training statistics and results
            8. Evaluates model every n-th update if configured
        """
        if(self.resume_at > 0):
            self.monitor.log("Step 5: Resuming training at step " + str(self.resume_at) + " using " + str(self.device) + " . . .")
        else:
            self.monitor.log("Step 5: Starting training using " + str(self.device) + " . . .")
        # List that stores the most recent episodes for training statistics
        episode_info = deque(maxlen=100)

        # Training loop
        for update in range(self.resume_at, self.updates):
            self.currentUpdate = update
            time_start = time.time()

            # 1.: Decay hyperparameters polynomially based on the provided config
            learning_rate, beta, clip_range = self.step_decay_schedules(update)

            # 2.: Sample data from each worker for worker steps
            sample_episode_info = self.sampler.sample(self.device)

            # 3.: Calculate advantages
            _, last_value, _ = self.model(self.sampler.last_vis_obs(), self.sampler.last_vec_obs(), self.sampler.last_recurrent_cell(), self.device)
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
            training_stats = self.train()
            
            # Store recent episode infos
            episode_info.extend(sample_episode_info)
    
            # Measure seconds needed for a whole update
            time_end = time.time()
            update_duration = int(time_end - time_start)

            # Save checkpoint (update, model, optimizer, configs)
            if update % self.checkpoint_interval == 0 or update == (self.updates - 1):
                self._save_checkpoint(update)

            # 7.: Write training statistics to console
            episode_result = self._process_episode_info(episode_info)
            if episode_result:
                self.monitor.log((("{:4} sec={:2} reward={:.2f} std={:.2f} length={:.1f} std={:.2f} loss={:3f} ") +
                    ("entropy={:.3f} value={:3f} std={:.3f} advantage={:.3f} std={:.3f} sequence length={:3}")).format(
                    update, update_duration, episode_result["reward_mean"], episode_result["reward_std"],
                    episode_result["length_mean"], episode_result["length_std"], training_stats["loss"][1],
                    training_stats["entropy"][1], np.mean(self.buffer.values), np.std(self.buffer.values),
                    np.mean(self.buffer.advantages), np.std(self.buffer.advantages), self.buffer.actual_sequence_length))
            else:
                self.monitor.log("{:4} sec={:2} loss={:3f} entropy={:.3f} value={:3f} std={:.3f} advantage={:.3f} std={:.3f} sequence length={:3}".format(
                    update, update_duration, training_stats["loss"][1], training_stats["entropy"][1],
                    np.mean(self.buffer.values), np.std(self.buffer.values), np.mean(self.buffer.advantages),
                    np.std(self.buffer.advantages), self.buffer.actual_sequence_length))

            # 8.: Evaluate model
            if self.eval:
                if update % self.eval_interval == 0 or update == (self.updates - 1):
                    eval_duration, eval_episode_info = self.evaluator.evaluate(self.model, self.device)
                    evaluation_result = self._process_episode_info(eval_episode_info)
                    self.monitor.log("eval: sec={:3} reward={:.2f} length={:.1f}".format(
                        eval_duration, evaluation_result["reward_mean"], evaluation_result["length_mean"]))
                    self.monitor.write_eval_summary(update, evaluation_result)
            
            # Add some more training statistics which should be monitored
            training_stats = {
            **training_stats,
            "advantage_mean": (Tag.EPISODE, np.mean(self.buffer.advantages)),
            "value_mean": (Tag.EPISODE, np.mean(self.buffer.values)),
            "sequence_length": (Tag.OTHER, self.buffer.actual_sequence_length),
            "learning_rate": (Tag.DECAY, learning_rate),
            "beta": (Tag.DECAY, beta),
            "clip_range": (Tag.DECAY, clip_range)}

            # Write training statistics to tensorboard
            self.monitor.write_training_summary(update, training_stats, episode_result)

    def create_model(self):
        raise NotImplementedError

    def train(self):
        # This function needs to be overriden by trainers that are based on this class.
        raise NotImplementedError

    def step_decay_schedules(self, update):
        # This function needs to be overriden by trainers that are based on this class.
        raise NotImplementedError

    def _save_checkpoint(self, update):
        checkpoint_data = self.collect_checkpoint_data(update)
        torch.save(checkpoint_data, self.monitor.checkpoint_path + self.run_id + "-" + str(update) + ".pt")

    def collect_checkpoint_data(self, update):
        checkpoint_data = {}
        checkpoint_data["config"] = self.configs
        checkpoint_data["update"] = update
        checkpoint_data["hxs"] = self.model.mean_hxs if self.recurrence is not None else None
        checkpoint_data["cxs"] = self.model.mean_cxs if self.recurrence is not None else None
        return checkpoint_data

    def _load_checkpoint(self):
        self.monitor.log("Step 3: Loading model from " + self.configs["model"]["model_path"])
        checkpoint = torch.load(self.configs["model"]["model_path"])
        self.apply_checkpoint_data(checkpoint)

    def apply_checkpoint_data(self, checkpoint):
        if self.recurrence is not None:
            self.model.set_mean_recurrent_cell_states(checkpoint["hxs"], checkpoint["cxs"])

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
        self.monitor.log("Terminate: Closing dummy ennvironment . . .")
        try:
            self.dummy_env.close()
        except:
            pass

        self.monitor.log("Terminate: Closing Monitor . . .")
        try:
            self.monitor.close()
        except:
            pass

        self.monitor.log("Terminate: Shutting down sampler . . .")
        try:
            self.sampler.close()
        except:
            pass

        try:
            if self.eval:
                self.monitor.log("Terminate: Closing evaluator")
                try:
                    self.evaluator.close()
                except:
                        pass
        except:
            pass
        
        try:
            if self.currentUpdate > 0:
                self.monitor.log("Terminate: Saving model . . .")
                try:
                        self._save_checkpoint(self.currentUpdate)
                        self.monitor.log("Terminate: Saved model to: " + self.monitor.checkpoint_path + self.run_id + "-" + str(self.currentUpdate) + ".pt")
                except:
                    pass
        except:
            pass

    def _handler(self, signal_received, frame):
        """Invoked by the Ctrl-C event, the trainer is being closed and the python program is being exited."""
        self.monitor.log("Terminate: Training aborted . . .")
        self.close()
        exit(0)
