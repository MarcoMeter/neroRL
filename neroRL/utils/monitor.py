from enum import Enum

class Tag(Enum):
    """The Tag Enum is used to group training and evaluation metrics."""
    DECAY = "decay"
    EPISODE = "episode"
    EVALUATION = "evaluation"
    GRADIENT_NORM = "gradient_norm"
    GRADIENT_MEAN = "gradient_mean"
    LOSS = "loss"
    OTHER = "other"

import logging
import os
import time
from torch.utils.tensorboard import SummaryWriter

class Monitor():
    """The monitor is in charge of logging training statistics to console, file and tensorboard.
    It further arranges all needed directories for saving outputs like model checkpoints.
    """
    def __init__(self, out_path, run_id, worker_id) -> None:
        """

        Arguments:
            out_path {str} -- Determines the target directory for saving summaries, logs and model checkpoints. (default: "./")
            run_id {string} -- The run_id is used to tag the training runs (directory names to store summaries and checkpoints) (default: {"default"})
            worker_id {int} -- Specifies the offset for the port to communicate with the environment, which is needed for Unity ML-Agents environments (default: {1})
        """
        self.timestamp = time.strftime("/%Y%m%d-%H%M%S"+ "_" + str(worker_id) + "/")
        self._create_directories(out_path, run_id)

        # Setup SummaryWriter
        self.writer = SummaryWriter(out_path + "summaries/" + run_id + self.timestamp)

        # Setup logger
        logging.basicConfig(level = logging.INFO, handlers=[])
        self.logger = logging.getLogger("train")

        console = logging.StreamHandler()
        console.setFormatter(logging.Formatter("%(asctime)s: %(message)s", "%Y-%m-%d %H:%M:%S"))
        path = out_path + "logs/" + run_id + self.timestamp[:-1] + ".log"
        logfile = logging.FileHandler(path, mode="w")
        self.logger.addHandler(console)
        self.logger.addHandler(logfile)

    def _create_directories(self, out_path, run_id) -> None:
        """Sets up directories for saving logs, tensorboard summaries and checkpoints.

        Arguments:
            out_path {str}: Determines the target directory for saving summaries, logs and model checkpoints. (default: "./")
            run_id {str}: The run_id is used to tag the training runs (directory names to store summaries and checkpoints) (default: {"default"})
        """
        # Create directories for storing checkpoints, logs and tensorboard summaries based on the current time and provided run_id
        if not os.path.exists(out_path + "summaries"):
            os.makedirs(out_path + "summaries")
        if not os.path.exists(out_path + "checkpoints"):
            os.makedirs(out_path + "checkpoints")
        if not os.path.exists(out_path + "logs") or not os.path.exists(out_path + "logs/" + run_id):
            os.makedirs(out_path + "logs/" + run_id)

        self.checkpoint_path = out_path + "checkpoints/" + run_id + self.timestamp
        os.makedirs(self.checkpoint_path)

    def log(self, message:str) -> None:
        """Prints a message to console and file

        Arguments:
            message {str} -- The message to be logged
        """
        self.logger.info(message)

    def write_eval_summary(self, update, episode_result) -> None:
        """Writes evaluation results to a tensorboard event summary file based on the run-id argument.

        Arguments:
            update {int} -- Current update
            episode_result {dict} -- Dicionary containing episode statistics such as cumulative reward and episode length
        """
        if episode_result:
            for key in episode_result:
                if "std" not in key:
                    self.writer.add_scalar("evaluation/" + key, episode_result[key], update)

    def write_training_summary(self, update, training_stats, episode_result) -> None:
        """Writes training results to a tensorboard event summary file based on the run-id argument.
        
        Arguments:
            update {int} -- Current update
            training_stats {dict} -- Dictionary containing training statistics such as losses
            episode_result {dict} -- Dicionary containing episode statistics such as cumulative reward and episode length
        """
        if episode_result:
            for key in episode_result:
                if "std" not in key:
                    self.writer.add_scalar("episode/" + key, episode_result[key], update)

        for key, (tag, value) in training_stats.items():
            self.writer.add_scalar(tag.value + "/" + key, value, update)

    def write_hyperparameters(self, configs) -> None:
        """Writes hyperparameters to the tensorboard event summary.
        
        Arguments:
            configs {dict} --  The whole set of configurations (e.g. training and environment configs)
        """
        for key, value in configs.items():
            if isinstance(value, dict):
                for k, v in value.items():
                    self.writer.add_text("Hyperparameters", k + " " + str(v))
            else:
                self.writer.add_text("Hyperparameters", key + " " + str(value))#

    def close(self) -> None:
        """Closes the monitor and shuts down the Tensorboard Summary Writer.
        """
        self.logger.info("Terminate: Closing Summary Writer . . .")
        try:
            self.writer.close()
        except:
            pass