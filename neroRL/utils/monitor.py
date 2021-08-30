from enum import Enum

class Tag(Enum):
    DECAY = "decay"
    EPISODE = "episode"
    EVALUATION = "evaluation"
    GRADIENT = "gradient"
    LOSS = "loss"
    OTHER = "other"

import logging
import os
import time
from torch.utils.tensorboard import SummaryWriter

class Monitor():
    def __init__(self, out_path, run_id, worker_id) -> None:        
        self.timestamp = time.strftime("/%Y%m%d-%H%M%S"+ "_" + str(worker_id) + "/")
        self._create_directories(out_path, run_id, worker_id)

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

    def _create_directories(self, out_path, run_id, worker_id) -> None:
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
        self.logger.info(message)

    def write_eval_summary(self, update, episode_result):
        """Writes to an event file based on the run-id argument."""
        if episode_result:
            for key in episode_result:
                if "std" not in key:
                    self.writer.add_scalar("evaluation/" + key, episode_result[key], update)

    def write_training_summary(self, update, training_stats, episode_result):
        """Writes to an event file based on the run-id argument."""
        if episode_result:
            for key in episode_result:
                if "std" not in key:
                    self.writer.add_scalar("episode/" + key, episode_result[key], update)

        for key, (tag, value) in training_stats.items():
            self.writer.add_scalar(tag.value + "/" + key, value, update)

    def write_hyperparameters(self, configs):
        """Writes hyperparameters to tensorboard"""
        for key, value in configs.items():
            if isinstance(value, dict):
                for k, v in value.items():
                    self.writer.add_text("Hyperparameters", k + " " + str(v))
            else:
                self.writer.add_text("Hyperparameters", key + " " + str(value))#

    def close(self):
        self.logger.info("Terminate: Closing Summary Writer . . .")
        try:
            self.writer.close()
        except:
            pass