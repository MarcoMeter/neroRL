import torch
import numpy as np
import time
from collections import deque

from gymnasium import spaces
from sys import exit
from signal import signal, SIGINT

import neroRL
from neroRL.environments.wrapper import wrap_environment
from neroRL.sampler.trajectory_sampler import TrajectorySampler
from neroRL.sampler.recurrent_sampler import RecurrentSampler
from neroRL.sampler.transformer_sampler import TransformerSampler
from neroRL.evaluator import Evaluator
from neroRL.utils.monitor import Monitor
from neroRL.utils.monitor import Tag
from neroRL.utils.utils import set_library_seeds

class BaseTrainer():
    """The BaseTrainer is in charge of setting up the whole training loop of a policy gradient based algorithm."""
    def __init__(self, configs, worker_id = 1, run_id  = "default", out_path = "./", seed = 0):
        """Initializes the trainer, the model, the buffer, the evaluator and the training data sampler

        Arguments:
            configs {dict} -- The whole set of configurations (e.g. training and environment configs)
            worker_id {int} -- Specifies the offset for the port to communicate with the environment, which is needed for Unity ML-Agents environments (default: {1})
            run_id {string} -- The run_id is used to tag the training runs (directory names to store summaries and checkpoints) (default: {"default"})
            out_path {str} -- Determines the target directory for saving summaries, logs and model checkpoints. (default: "./")
            seed {int} -- Specifies the seed to use during training. (default: {0})
        """
        # Handle Ctrl + C event, which aborts and shuts down the training process in a controlled manner
        signal(SIGINT, self._handler)

        # Determine cuda availability
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            torch.set_default_tensor_type("torch.cuda.FloatTensor")
        else:
            torch.set_default_tensor_type("torch.FloatTensor")
        
        # Create a monitor that is used for logging and monitoring training statistics (tensorboard)
        self.monitor = Monitor(out_path, run_id, worker_id)
        # Add hyperparameters to the tensorboard summary
        self.monitor.write_hyperparameters(configs)

        # Init members
        self.run_id = run_id
        self.configs = configs
        self.resume_at = configs["trainer"]["resume_at"]
        self.refresh_buffer_epoch = configs["trainer"]["refresh_buffer_epoch"]
        self.gamma = configs["trainer"]["gamma"]
        self.lamda = configs["trainer"]["lamda"]
        self.updates = configs["trainer"]["updates"]
        self.n_workers = configs["sampler"]["n_workers"]
        self.worker_steps = configs["sampler"]["worker_steps"]
        self.worker_id = worker_id
        self.recurrence = None if not "recurrence" in configs["model"] else configs["model"]["recurrence"]
        self.transformer = None if not "transformer" in configs["model"] else configs["model"]["transformer"]
        self.checkpoint_interval = configs["model"]["checkpoint_interval"]

        # Set the seed for conducting the training
        self.seed = seed
        set_library_seeds(self.seed)
        self.monitor.log("Step 0: Training Seed: " + str(self.seed))

        # Start logging the training setup
        self.monitor.log("Step 1: Provided config:")
        for key in configs:
            self.monitor.log("\t" + str(key) + ":")
            for k, v in configs[key].items():
                self.monitor.log("\t" * 2 + str(k) + ": " + str(v))

        # Create dummy environment to retrieve the shapes of the observation and action space for further processing
        self.monitor.log("Step 2: Creating dummy environment")
        self.dummy_env = wrap_environment(configs["environment"], worker_id)
        vis_obs, vec_obs = self.dummy_env.reset(configs["environment"]["reset_params"])
        self.max_episode_steps = self.dummy_env.max_episode_steps
        if self.transformer is not None:
            # Add max episode steps to the transformer config
            configs["model"]["transformer"]["max_episode_steps"] = self.max_episode_steps
        self.visual_observation_space = self.dummy_env.visual_observation_space
        self.vector_observation_space = self.dummy_env.vector_observation_space
        if isinstance(self.dummy_env.action_space, spaces.Discrete):
            self.action_space_shape = (self.dummy_env.action_space.n,)
        else:
            self.action_space_shape = tuple(self.dummy_env.action_space.nvec)
        self.dummy_env.close()

        self.monitor.log("\t" + "Visual Observation Space: " + str(self.visual_observation_space))
        self.monitor.log("\t" + "Vector Observation Space: " + str(self.vector_observation_space))
        self.monitor.log("\t" + "Action Space Shape: " + str(self.action_space_shape))
        self.monitor.log("\t" + "Action Names: " + str(self.dummy_env.action_names))
        self.monitor.log("\t" + "Max Episode Steps: " + str(self.max_episode_steps))

        # Prepare evaluator if configured
        self.eval = configs["evaluation"]["evaluate"]
        self.eval_interval = configs["evaluation"]["interval"]
        if self.eval and self.eval_interval > 0:
            self.monitor.log("Step 2b: Initializing evaluator")
            self.evaluator = Evaluator(configs, configs["model"], worker_id, self.visual_observation_space, self.vector_observation_space, self.max_episode_steps)
        else:
            self.evaluator = None

        # Init model
        self.monitor.log("Step 3: Creating model")
        self.model = self.create_model()

        # Set model to train mode
        self.model.train()

        # Setup Sampler
        self.monitor.log("Step 4: Launching training environments of type " + configs["environment"]["type"])
        # Instantiate sampler for memory-less / markvoivan policies
        if self.recurrence is None and self.transformer is None:
            self.sampler = TrajectorySampler(configs, worker_id, self.visual_observation_space, self.vector_observation_space,
                                        self.action_space_shape, self.model, self.device)
        # Instantiate sampler for recurrent policies
        elif self.recurrence is not None:
            self.sampler = RecurrentSampler(configs, worker_id, self.visual_observation_space, self.vector_observation_space,
                                        self.action_space_shape, self.model, self.device)
        # Instantiate sampler for transformer policoes
        elif self.transformer is not None:
            self.sampler = TransformerSampler(configs, worker_id, self.visual_observation_space, self.vector_observation_space,
                                        self.action_space_shape, self.max_episode_steps, self.model, self.device)

        # List that stores the most recent episodes for training statistics
        self.episode_info = deque(maxlen=100)

    def run_training(self):
        """Orchestrates the policy gradient based training:
            1. Decays training parameters in relation to training progression
            2. Samples data from current policy
            3. Computes advantages
            4. If a recurrent policy is used, set the mean of the recurrent cell states for future initializations
            5. Prepare sampled data
            6. Optimize policy and value function
            7. Processes training statistics and results
            8. Evaluates model every n-th update if configured

            Returns the mean return of the latest 100 episodes
        """
        # Load checkpoint and apply data
        if self.configs["model"]["load_model"]:
            self._load_checkpoint()

        if(self.resume_at > 0):
            self.monitor.log("Step 5: Resuming training at step " + str(self.resume_at) + " using " + str(self.device) + " . . .")
        else:
            self.monitor.log("Step 5: Starting training using " + str(self.device) + " . . .")

        # Training loop
        for update in range(self.resume_at, self.updates):
            episode_result, training_stats, formatted_string, update_duration, decayed_hyperparameters = self.step(update)

            # Save checkpoint (update, model, optimizer, configs)
            if update % self.checkpoint_interval == 0 or update == (self.updates - 1):
                self._save_checkpoint(update)

            # 7.: Write training statistics to console
            episode_result = self._process_episode_info(self.episode_info)
            if episode_result:
                self.monitor.log((("{:4} sec={:2} reward={:.2f} std={:.2f} length={:.1f} std={:.2f} ") +
                    (" value={:3f} std={:.3f} advantage={:.3f} std={:.3f} sequence length={:3}")).format(
                    update, update_duration, episode_result["reward_mean"], episode_result["reward_std"],
                    episode_result["length_mean"], episode_result["length_std"], torch.mean(self.sampler.buffer.values), torch.std(self.sampler.buffer.values),
                    torch.mean(self.sampler.buffer.advantages), torch.std(self.sampler.buffer.advantages), self.sampler.buffer.actual_sequence_length) +
                    " " + formatted_string)
            else:
                self.monitor.log("{:4} sec={:2} value={:3f} std={:.3f} advantage={:.3f} std={:.3f} sequence length={:3}".format(
                    update, update_duration, torch.mean(self.sampler.buffer.values), torch.std(self.sampler.buffer.values),
                    torch.mean(self.sampler.buffer.advantages), torch.std(self.sampler.buffer.advantages), self.sampler.buffer.actual_sequence_length) +
                    " " + formatted_string)

            # 8.: Evaluate model
            if self.eval:
                if update % self.eval_interval == 0 or update == (self.updates - 1):
                    eval_duration, evaluation_result = self.eval()
                    self.monitor.log("eval: sec={:3} reward={:.2f} length={:.1f}".format(
                        eval_duration, evaluation_result["reward_mean"], evaluation_result["length_mean"]))
                    self.monitor.write_eval_summary(update, evaluation_result)
            
            # Add some more training statistics which should be monitored
            training_stats = {
            **training_stats,
            **decayed_hyperparameters,
            "advantage_mean": (Tag.EPISODE, torch.mean(self.sampler.buffer.advantages)),
            "value_mean": (Tag.EPISODE, torch.mean(self.sampler.buffer.values)),
            "sequence_length": (Tag.OTHER, self.sampler.buffer.actual_sequence_length),
            }

            # Write training statistics to tensorboard
            self.monitor.write_training_summary(update, training_stats, episode_result)

        return episode_result["reward_mean"]

    def step(self, update):
        self.currentUpdate = update
        time_start = time.time()

        # 1.: Decay hyperparameters polynomially based on the provided config
        decayed_hyperparameters = self.step_decay_schedules(update)

        # 2.: Sample data from each worker for worker steps
        sample_episode_info = self.sampler.sample(self.device)

        # 3.: Calculate advantages
        last_value = self.sampler.get_last_value()
        self.sampler.buffer.calc_advantages(last_value, self.gamma, self.lamda)
        
        # 4.: If a recurrent policy is used, set the mean of the recurrent cell states for future initializations
        if self.recurrence:
            if self.recurrence["layer_type"] == "lstm":
                self.model.set_mean_recurrent_cell_states(
                        torch.mean(self.sampler.buffer.hxs.reshape(self.n_workers * self.worker_steps, *self.sampler.buffer.hxs.shape[2:]), axis=0),
                        torch.mean(self.sampler.buffer.cxs.reshape(self.n_workers * self.worker_steps, *self.sampler.buffer.cxs.shape[2:]), axis=0))
            else:
                self.model.set_mean_recurrent_cell_states(
                        torch.mean(self.sampler.buffer.hxs.reshape(self.n_workers * self.worker_steps, *self.sampler.buffer.hxs.shape[2:]), axis=0), None)

        # 5.: Prepare the sampled data inside the buffer
        self.sampler.buffer.prepare_batch_dict()

        # 6.: Train n epochs over the sampled data
        if torch.cuda.is_available():
            self.model.cuda() # Train on GPU
        training_stats, formatted_string = self.train()

        # Free memory
        del(self.sampler.buffer.samples_flat)
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        # mem = torch.cuda.mem_get_info(device=None)
        # mem = (mem[1] - mem[0]) / 1024 / 1024
        # print(mem)
        
        # Store recent episode infos
        self.episode_info.extend(sample_episode_info)

        # Measure seconds needed for a whole update
        time_end = time.time()
        update_duration = int(time_end - time_start)

        # 7.: Process training stats to print to console and write summaries
        episode_result = self._process_episode_info(self.episode_info)

        return episode_result, training_stats, formatted_string, update_duration, decayed_hyperparameters

    def evaluate(self):
        if self.evaluator is None:
            self.evaluator = Evaluator(self.configs, self.configs["model"], self.worker_id, self.visual_observation_space, self.vector_observation_space, self.max_episode_steps)
        eval_duration, eval_episode_info = self.evaluator.evaluate(self.model, self.device)
        evaluation_result = self._process_episode_info(eval_episode_info)
        self.monitor.log("eval: sec={:3} reward={:.2f} length={:.1f}".format(
        eval_duration, evaluation_result["reward_mean"], evaluation_result["length_mean"]))
        return eval_duration, evaluation_result
    
    ### BEGIN:  Methods that need to be overriden/extended ###
    def create_model(self) -> None:
        """This method is supposed to initialize self.model. Every trainer should have the possibility to customize its own model."""
        raise NotImplementedError

    def train(self) -> dict:
        """train() is called for each update cycle. It returns a dictionary of training statistics and a formatted string featuring
        distinct training stats such as losses."""
        raise NotImplementedError

    def step_decay_schedules(self, update:int) -> dict:
        """As there might be multiple decaying hyperparameters, each trainer has to take care of them themselves.
        This function is called first during an update cycle.
        
        Arguments:
            update {int} -- Current update#

        Returns:
            {dict} -- Dictionary containing the current values of all decayed hyperparameters
        """
        raise NotImplementedError

    def collect_checkpoint_data(self, update):
        """Collects all necessary objects for serializing a checkpoint.
        The base class provide some of these information. The rest is collected from the actual trainer.

        Arguments:
            update {int} -- Current update

        Returns:
            {dict} -- Checkpoint data
        """
        checkpoint_data = {}
        checkpoint_data["configs"] = self.configs
        checkpoint_data["update"] = update
        checkpoint_data["hxs"] = self.model.mean_hxs if self.recurrence is not None else None
        checkpoint_data["cxs"] = self.model.mean_cxs if self.recurrence is not None else None
        checkpoint_data["visual_observation_space"] = self.visual_observation_space
        checkpoint_data["vector_observation_space"] = self.vector_observation_space
        checkpoint_data["action_space_shape"] = self.action_space_shape
        checkpoint_data["version"] = neroRL.__version__
        checkpoint_data["run_id"] = self.run_id
        checkpoint_data["seed"] = self.seed
        return checkpoint_data

    def apply_checkpoint_data(self, checkpoint):
        """Applies the data loaded from a checkpoint. Some are processed by this base class and the others
        by the actual trainer.

        Args:
            checkpoint {dict} -- The to be applied checkpoint data
        """
        if self.recurrence is not None:
            self.model.set_mean_recurrent_cell_states(checkpoint["hxs"], checkpoint["cxs"])
    ### END:    Methods that need to be overriden/extended ###

    def _save_checkpoint(self, update) -> None:
        """Collects data from the base and the trainer to serialize them using torch.save().

        Arguments:
            update {int} -- Current update
        """
        checkpoint_data = self.collect_checkpoint_data(update)
        checkpoint_data["seed"] = self.seed
        torch.save(checkpoint_data, self.monitor.checkpoint_path + self.run_id + "-" + str(update) + ".pt")

    def _load_checkpoint(self):
        """Loads a checkpoint from a specified file by the config and triggers the process of applying the loaded data."""
        self.monitor.log("Step 3: Loading model from " + self.configs["model"]["model_path"])
        checkpoint = torch.load(self.configs["model"]["model_path"])
        if checkpoint["version"] != neroRL.__version__:
            self.monitor.log("WARNING: The loaded model is created with a different version of neroRL. " +
                "The loaded model might not work properly.")
        self.seed = checkpoint["seed"]
        set_library_seeds(self.seed)
        self.apply_checkpoint_data(checkpoint)

    @staticmethod
    def _process_episode_info(episode_info):
        """Extracts the mean and std of completed episodes. At minimum the episode length and the collected reward is available.
        
        Arguments:
            episode_info {dict} -- Episode information, such as cummulated reward, episode length and more.
        """
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
        """Closes the environment and destroys the environment workers"""
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
