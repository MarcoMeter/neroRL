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
    ERROR = "error"

import logging
import os
import re
import time
import sys
import random
from torch.utils.tensorboard import SummaryWriter

class TrainingMonitor():
    """The monitor is in charge of logging training statistics to console, file and tensorboard.
    It further arranges all needed directories for saving outputs like model checkpoints.
    """
    def __init__(self, out_path, run_id, worker_id, checkpoint_path = None) -> None:
        """

        Arguments:
            out_path {str} -- Determines the target directory for saving summaries, logs and model checkpoints. (default: "./")
            run_id {string} -- The run_id is used to tag the training runs (directory names to store summaries and checkpoints) (default: {"default"})
            worker_id {int} -- Specifies the offset for the port to communicate with the environment, which is needed for Unity ML-Agents environments (default: {1})
            checkpoint_path {str} -- The checkpoint path to extract the timestamp from. (default: {None})
        """
        if checkpoint_path is not None:
            self.timestamp = "/" + re.search(r"(\d{8}-\d{6}_\d+)", checkpoint_path).group(1) + "/"
        else: 
            self.timestamp = time.strftime("/%Y%m%d-%H%M%S"+ "_" + str(worker_id) + "/")
        duplicate_suffix = ""
        log_path = out_path + "logs/" + run_id + self.timestamp[:-1] + ".log"
        # Check whether this path setup already exists
        if os.path.isfile(log_path) and checkpoint_path is None:
            # If so, add a random suffix to distinguish duplicate runs
            duplicate_suffix = "_" + str(random.randint(0, 1000))
            log_path = out_path + "logs/" + run_id + self.timestamp[:-1] + duplicate_suffix + ".log"

        # Create directories
        self._create_directories(out_path, run_id, duplicate_suffix)

        # Setup SummaryWriter
        summary_path = out_path + "summaries/" + run_id + self.timestamp[:-1] + duplicate_suffix + "/"
        # If a checkpoint path is provided, then add a suffix to the summary path to distinguish duplicate runs
        if checkpoint_path is not None:
            self.writer = SummaryWriter(log_dir = summary_path, filename_suffix='.v2')
        else:
            self.writer = SummaryWriter(summary_path)

        # Setup logger
        logging.basicConfig(level = logging.INFO, handlers=[])
        self.logger = logging.getLogger("train")
        self.console = logging.StreamHandler()
        self.console.setFormatter(logging.Formatter("%(asctime)s: %(message)s", "%Y-%m-%d %H:%M:%S"))
        
        # If a checkpoint path is provided, then append to the existing log file
        mode_logfile = "a" if checkpoint_path is not None else "w"
        self.logfile = logging.FileHandler(log_path, mode=mode_logfile)
        self.logger.addHandler(self.console)
        self.logger.addHandler(self.logfile)

    def _create_directories(self, out_path, run_id, duplicate_suffix) -> None:
        """Sets up directories for saving logs, tensorboard summaries and checkpoints.

        Arguments:
            out_path {str}: Determines the target directory for saving summaries, logs and model checkpoints. (default: "./")
            run_id {str}: The run_id is used to tag the training runs (directory names to store summaries and checkpoints) (default: {"default"})
            duplicate_suffix {str}: This suffix is added to the end of the path to distinguish duplicate runs.
        """
        # Create directories for storing checkpoints, logs and tensorboard summaries based on the current time and provided run_id
        self.checkpoint_path = out_path + "checkpoints/" + run_id + self.timestamp[:-1] + duplicate_suffix + "/"
        if not os.path.exists(out_path + "summaries"):
            os.makedirs(out_path + "summaries")
        if not os.path.exists(out_path + "checkpoints"):
            os.makedirs(out_path + "checkpoints")
        if not os.path.exists(out_path + "logs") or not os.path.exists(out_path + "logs/" + run_id):
            os.makedirs(out_path + "logs/" + run_id)
        if not os.path.exists(self.checkpoint_path):
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
            self.logger.removeHandler(self.console)
            self.logger.removeHandler(self.logfile)
        except:
            pass
