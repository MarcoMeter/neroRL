from enum import Enum
import logging
from torch.utils.tensorboard import SummaryWriter

class Tag(Enum):
    DECAY = "decay"
    EPISODE = "episode"
    EVALUATION = "evaluation"
    GRADIENT = "gradient"
    LOSS = "loss"
    OTHER = "other"

class Monitor():
    def __init__(self, configs, out_path, run_id, timestamp) -> None:

        # Setup SummaryWriter
        self.writer = SummaryWriter(out_path + "summaries/" + run_id + timestamp)
        
        # Log hyperparmeters
        self._write_hyperparameters(configs)

        # Setup logger
        logging.basicConfig(level = logging.INFO, handlers=[])
        self.logger = logging.getLogger("train")

        console = logging.StreamHandler()
        console.setFormatter(logging.Formatter("%(asctime)s: %(message)s", "%Y-%m-%d %H:%M:%S"))
        path = out_path + "logs/" + run_id + timestamp[:-1] + ".log"
        logfile = logging.FileHandler(path, mode="w")
        self.logger.addHandler(console)
        self.logger.addHandler(logfile)

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

        for key, info in training_stats.items():
            (tag, value) = info
            self.writer.add_scalar(tag.value + "/" + key, value, update)

    def _write_hyperparameters(self, configs):
        """Writes hyperparameters to tensorboard"""
        for key, value in configs.items():
            if isinstance(value, dict):
                for k, v in value.items():
                    self.writer.add_text("Hyperparameters", k + " " + str(v))
            else:
                self.writer.add_text("Hyperparameters", key + " " + str(value))

    def close(self):
        self.logger.info("Terminate: Closing Summary Writer . . .")
        try:
            self.writer.close()
        except:
            pass