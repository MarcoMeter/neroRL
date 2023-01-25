"""
Runs the training program using the provided config and arguments.
"""
import random
import sys
import torch

from docopt import docopt
from pathlib import Path
from signal import signal, SIGINT

from neroRL.evaluator import Evaluator
from neroRL.trainers.policy_gradient.ppo_shared import PPOTrainer
from neroRL.trainers.policy_gradient.ppo_decoupled import DecoupledPPOTrainer
from neroRL.utils.monitor import Monitor
from neroRL.utils.monitor import Tag
from neroRL.utils.utils import set_library_seeds
from neroRL.utils.yaml_parser import YamlParser

class Training():
    def __init__(self, configs, run_id, worker_id, out_path, seed) -> None:
        """
        Arguments:
            configs {dict} -- Environment, Model and Training configuration
            run_id {str} -- Short tag that describes the underlying training run
            worker_id {int} -- A different worker id has to be chosen for socket based environments like Unity ML-Agents
            out_path {str} -- Desired output path
            seed {int} -- Seed for all number generators
        """
        # Handle Ctrl + C event, which aborts and shuts down the training process in a controlled manner
        signal(SIGINT, self._close)

        # Sampled seed if a value smaller than 0 was submitted
        if seed < 0:
            self.seed = random.randint(0, 2 ** 31 - 1)
        set_library_seeds(self.seed)

        # Determine cuda availability
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            torch.set_default_tensor_type("torch.cuda.FloatTensor")
        else:
            torch.set_default_tensor_type("torch.FloatTensor")

        # Create training monitor
        self.monitor = Monitor(out_path, run_id, worker_id)
        self.monitor.write_hyperparameters(configs)
        self.monitor.log("Training Seed: " + str(self.seed))
        # Start logging the training setup
        self.monitor.log("Provided config:")
        for key in configs:
            self.monitor.log("\t" + str(key) + ":")
            for k, v in configs[key].items():
                self.monitor.log("\t" * 2 + str(k) + ": " + str(v))

        # Initialize trainer
        if configs["trainer"]["algorithm"] == "PPO":
            self.trainer = PPOTrainer(configs, self.device, worker_id, run_id, out_path, self.seed)
        elif configs["trainer"]["algorithm"] == "DecoupledPPO":
            self.trainer = DecoupledPPOTrainer(configs, self.device, worker_id, run_id, out_path, self.seed)
        else:
            assert(False), "Unsupported algorithm specified"

        self.monitor.log("Environment specs:")
        self.monitor.log("\t" + "Visual Observation Space: " + str(self.trainer.vis_obs_space))
        self.monitor.log("\t" + "Vector Observation Space: " + str(self.trainer.vec_obs_space))
        self.monitor.log("\t" + "Action Space Shape: " + str(self.trainer.action_space_shape))
        self.monitor.log("\t" + "Max Episode Steps: " + str(self.trainer.max_episode_steps))

        # Init evaluator if configured
        self.eval = configs["evaluation"]["evaluate"]
        self.eval_interval = configs["evaluation"]["interval"]
        if self.eval and self.eval_interval > 0:
            self.monitor.log("Initializing evaluator")
            self.evaluator = Evaluator(configs, configs["model"], worker_id, self.trainer.vis_obs_space, self.trainer.vec_obs_space, self.trainer.max_episode_steps)
        else:
            self.evaluator = None

        # Load checkpoint and apply data
        if configs["model"]["load_model"]:
            self.monitor.log("Load checkpoint: " + configs["model"]["model_path"])
            self.trainer.load_checkpoint(configs["model"]["model_path"])

        # Set variables
        self.configs = configs
        self.resume_at = configs["trainer"]["resume_at"]
        self.updates = configs["trainer"]["updates"]
        self.run_id = run_id
        self.worker_id = worker_id
        self.out_path = out_path

    def run(self):
        """Run training loop"""
        if(self.resume_at > 0):
            self.monitor.log("Step 5: Resuming training at step " + str(self.resume_at) + " using " + str(self.device) + " . . .")
        else:
            self.monitor.log("Step 5: Starting training using " + str(self.device) + " . . .")

        for update in range(self.resume_at, self.updates):
            episode_result, training_stats, formatted_string, update_duration, decayed_hyperparameters = self.trainer.step(update)
            # Log to console
            if episode_result:
                self.monitor.log((("{:4} sec={:2} reward={:.2f} std={:.2f} length={:.1f} std={:.2f} ") +
                    (" value={:3f} std={:.3f} adv={:.3f} std={:.3f}")).format(
                    update, update_duration, episode_result["reward_mean"], episode_result["reward_std"],
                    episode_result["length_mean"], episode_result["length_std"], torch.mean(self.trainer.sampler.buffer.values), torch.std(self.trainer.sampler.buffer.values),
                    torch.mean(self.trainer.sampler.buffer.advantages), torch.std(self.trainer.sampler.buffer.advantages)) +
                    " " + formatted_string)
            else:
                self.monitor.log("{:4} sec={:2} value={:3f} std={:.3f} adv={:.3f} std={:.3f}".format(
                    update, update_duration, torch.mean(self.trainer.sampler.buffer.values), torch.std(self.trainer.sampler.buffer.values),
                    torch.mean(self.trainer.sampler.buffer.advantages), torch.std(self.trainer.sampler.buffer.advantages)) +
                    " " + formatted_string)
            # Evaluate agent
            if self.eval:
                if update % self.eval_interval == 0 or update == (self.updates - 1):
                    eval_duration, eval_episode_info = self.evaluator.evaluate(self.trainer.model, self.device)
                    evaluation_result = self.trainer.process_episode_info(eval_episode_info)
                    self.monitor.log("eval: sec={:3} reward={:.2f} length={:.1f}".format(
                        eval_duration, evaluation_result["reward_mean"], evaluation_result["length_mean"]))
                    self.monitor.write_eval_summary(update, evaluation_result)
            # Save checkpoint (update, model, optimizer, configs)
            if update % self.configs["model"]["checkpoint_interval"] == 0 or update == (self.configs["trainer"]["updates"] - 1):
                self.monitor.log("Saving model to " + self.monitor.checkpoint_path + self.run_id + "-" + str(update) + ".pt")
                self.trainer.save_checkpoint(update, self.monitor.checkpoint_path + self.run_id + "-" + str(update))
            # Log to tensorboard
            # Add some more training statistics which should be monitored
            training_stats = {
            **training_stats,
            **decayed_hyperparameters,
            "advantage_mean": (Tag.EPISODE, torch.mean(self.trainer.sampler.buffer.advantages)),
            "value_mean": (Tag.EPISODE, torch.mean(self.trainer.sampler.buffer.values)),
            "sequence_length": (Tag.OTHER, self.trainer.sampler.buffer.actual_sequence_length),
            }
            # Write training statistics to tensorboard
            self.monitor.write_training_summary(update, training_stats, episode_result)

        # Clean up after training
        self._close(None, None)

    def _close(self, signal_received, frame):
        self.monitor.log("Terminating training ...")
        if self.trainer.current_update > 0:
            self.monitor.log("Terminate: Saving model . . .")
            try:
                    self.trainer.save_checkpoint(self.trainer.current_update, self.monitor.checkpoint_path + self.run_id + "-" + str(self.trainer.current_update))
                    self.monitor.log("Terminate: Saved model to: " + self.monitor.checkpoint_path + self.run_id + "-" + str(self.trainer.current_update) + ".pt")
            except:
                pass
        try:
            if self.evaluator is not None:
                self.monitor.log("Terminate: Closing evaluator")
                try:
                    self.evaluator.close()
                except:
                    pass
                self.evaluator = None
        except:
            pass
        self.monitor.log("Terminate: Closing Trainer . . .")
        try:
            self.trainer.close()
        except:
            pass
        self.monitor.log("Terminate: Closing Monitor . . .")
        try:
            self.monitor.close()
        except:
            pass
        exit(0)

def main():
    # Docopt command line arguments
    _USAGE = """
    Usage:
        ntrain [options]
        ntrain --help

    Options:
        --config=<path>            Path to the config file [default: ./configs/default.yaml].
        --worker-id=<n>            Sets the port for each environment instance [default: 2].
        --run-id=<path>            Specifies the tag of the tensorboard summaries [default: default].
        --out=<path>               Specifies the path to output files such as summaries and checkpoints. [default: ./]
        --seed=<n>      	       Specifies the seed to use during training. If set to smaller than 0, use a random seed. [default: -1]
    """
    # Debug CUDA
    # import os
    # os.environ['CUDA_LAUNCH_BLOCKING'] = "1" 
    options = docopt(_USAGE)
    config_path = options["--config"]
    worker_id = int(options["--worker-id"])
    run_id = options["--run-id"]
    out_path = options["--out"]
    seed = int(options["--seed"])

    # If a run-id was not assigned, use the config's name
    for i, arg in enumerate(sys.argv):
        if "--run-id" in arg:
            run_id = options["--run-id"]
            break
        else:
            run_id = Path(config_path).stem

    # Load environment, model, evaluation and training parameters
    configs = YamlParser(config_path).get_config()

    # Training program
    training = Training(configs, run_id, worker_id, out_path, seed)
    # import cProfile, pstats
    # profiler = cProfile.Profile()
    # profiler.enable()
    training.run()
    # profiler.disable()
    # stats = pstats.Stats(profiler).sort_stats('cumtime')
    # stats.print_stats()

if __name__ == "__main__":
    main()
    