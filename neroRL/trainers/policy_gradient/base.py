import numpy as np
import platform
import time
import torch
from collections import deque

import neroRL
from neroRL.sampler.trajectory_sampler import TrajectorySampler
from neroRL.sampler.recurrent_sampler import RecurrentSampler
from neroRL.sampler.transformer_sampler import TransformerSampler
from neroRL.utils.monitor import Tag
from neroRL.utils.utils import get_environment_specs

class BaseTrainer():
    """The BaseTrainer is in charge of setting up the whole training loop of a policy gradient based algorithm."""
    def __init__(self, configs, sample_device, train_device, worker_id = 1, run_id  = "default", out_path = "./", seed = 0, compile_model = False):
        """Initializes the trainer, the model, the buffer, the evaluator and the training data sampler

        Arguments:
            configs {dict} -- The whole set of configurations (e.g. training and environment configs)
            sample_device {torch.device} -- The device used for sampling training data.
            train_device {torch.device} -- The device used for model optimization.
            worker_id {int} -- Specifies the offset for the port to communicate with the environment, which is needed for Unity ML-Agents environments (default: {1})
            run_id {string} -- The run_id is used to tag the training runs (directory names to store summaries and checkpoints) (default: {"default"})
            out_path {str} -- Determines the target directory for saving summaries, logs and model checkpoints. (default: "./")
            seed {int} -- Specifies the seed to use during training. (default: {0})
            compile_model {bool} -- Specifies whether the model should be compiled or not. (default: {False})
        """
        # Init members
        self.sample_device = sample_device
        self.train_device = train_device
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
        self.seed = seed

        # Create dummy environment to retrieve the shapes of the observation and action space for further processing
        self.vis_obs_space, self.vec_obs_space, self.action_space_shape, self.max_episode_steps = get_environment_specs(configs["environment"], worker_id + 1)
        if self.transformer is not None:
            # Add max episode steps to the transformer config
            configs["model"]["transformer"]["max_episode_steps"] = self.max_episode_steps
            # If memory length is -1 or invalid, set it to max episode steps
            memory_length = configs["model"]["transformer"]["memory_length"]
            if memory_length < 0 or memory_length > self.max_episode_steps:
                print("WARNING: Transformer memory length is invalid. Setting it to max episode steps.")
                configs["model"]["transformer"]["memory_length"] = self.max_episode_steps

        # Init model
        self.model = self.create_model()
        if compile_model:
            # Compile the model if not on Windows and if Pytorch version is >= 2.0
            if not platform.system() == "Windows" and torch.__version__ >= "2.0":
                self.model = torch.compile(self.model, dynamic=True)

        # Set model to train mode
        self.model.train()

        # Setup Sampler
        # Instantiate sampler for memory-less / markvoivan policies
        if self.recurrence is None and self.transformer is None:
            self.sampler = TrajectorySampler(configs, worker_id, self.vis_obs_space, self.vec_obs_space,
                                        self.action_space_shape, self.model, self.sample_device, self.train_device)
        # Instantiate sampler for recurrent policies
        elif self.recurrence is not None:
            self.sampler = RecurrentSampler(configs, worker_id, self.vis_obs_space, self.vec_obs_space,
                                        self.action_space_shape, self.model, self.sample_device, self.train_device)
        # Instantiate sampler for transformer policoes
        elif self.transformer is not None:
            self.sampler = TransformerSampler(configs, worker_id, self.vis_obs_space, self.vec_obs_space,
                    self.action_space_shape, self.max_episode_steps, self.model, self.sample_device, self.train_device)

        # List that stores the most recent episodes for training statistics
        self.episode_info = deque(maxlen=100)

    def step(self, update):
        self.current_update = update
        time_start = time.time()

        # 1.: Decay hyperparameters polynomially based on the provided config
        decayed_hyperparameters = self.step_decay_schedules(update)

        # 2.: Sample data from each worker for worker steps
        self.model.to(self.sample_device)
        sample_episode_info = self.sampler.sample()

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
        self.model.to(self.train_device)
        training_stats, formatted_string = self.train()
        # Extend training statistics
        training_stats = {
            **training_stats,
            **decayed_hyperparameters,
            "advantage_mean": (Tag.EPISODE, torch.mean(self.sampler.buffer.advantages).cpu().item()),
            "value_mean": (Tag.EPISODE, torch.mean(self.sampler.buffer.values).cpu().item()),
            "sequence_length": (Tag.OTHER, self.sampler.buffer.actual_sequence_length),
            }

        # Free VRAM
        del(self.sampler.buffer.samples_flat)
        if self.train_device.type == "cuda":
            torch.cuda.empty_cache()
        # mem = torch.cuda.mem_get_info(device=None)
        # print("Optim 2 mem usage: {} MB".format((mem[1] - mem[0]) / 1024 / 1024))
        
        # Store recent episode infos
        self.episode_info.extend(sample_episode_info)

        # Measure seconds needed for a whole update
        time_end = time.time()
        update_duration = int(time_end - time_start)

        return self.episode_info, training_stats, formatted_string, update_duration
    
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
        checkpoint_data["visual_observation_space"] = self.vis_obs_space
        checkpoint_data["vector_observation_space"] = self.vec_obs_space
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

    def save_checkpoint(self, update, path) -> None:
        """Collects data from the base and the trainer to serialize them using torch.save().

        Arguments:
            update {int} -- Current update
        """
        checkpoint_data = self.collect_checkpoint_data(update)
        checkpoint_data["seed"] = self.seed
        torch.save(checkpoint_data, path + ".pt")

    def load_checkpoint(self, path):
        """Loads a checkpoint from a specified file by the config and triggers the process of applying the loaded data."""
        checkpoint = torch.load(path)
        if checkpoint["version"] != neroRL.__version__:
            print("WARNING: The loaded model is created with a different version of neroRL. " +
                "The loaded model might not work properly.")
        self.seed = checkpoint["seed"]
        self.apply_checkpoint_data(checkpoint)

    @staticmethod
    def process_episode_info(episode_info):
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
        try:
            self.sampler.close()
        except:
            pass

    def to(self, device):
        """Args:
            device {torch.device} -- Desired device for all current tensors and models"""
        self.sampler.to(device)
        self.sampler.buffer.to(device)
        self.model.to(device)

    def get_num_trainable_parameters_str(self):
        """
        Returns:
            str -- Formatted number of trainable parameters
        """
        num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        if num_params > 1e6:
            num_params = str(round(num_params / 1e6, 2)) + "M"
        elif num_params > 1e3:
            num_params = str(round(num_params / 1e3, 2)) + "K"
        else:
            num_params = str(num_params)
        return num_params