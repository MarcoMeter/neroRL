"""
Runs the training program using the provided config and arguments.
"""
import random
import sys
import time
import torch
import re

from docopt import docopt
from pathlib import Path
from signal import signal, SIGINT

from neroRL.evaluator import Evaluator
from neroRL.trainers.policy_gradient.ppo_shared import PPOTrainer
from neroRL.trainers.policy_gradient.ppo_decoupled import DecoupledPPOTrainer
from neroRL.utils.monitor import TrainingMonitor
from neroRL.utils.utils import aggregate_episode_results, set_library_seeds
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from neroRL.utils.yaml_parser import YamlParser

class Training():
    def __init__(self, configs, run_id, worker_id, out_path, seed, compile_model, low_mem, checkpoint_path) -> None:
        """
        Arguments:
            configs {dict} -- Environment, Model and Training configuration
            run_id {str} -- Short tag that describes the underlying training run
            worker_id {int} -- A different worker id has to be chosen for socket based environments like Unity ML-Agents
            out_path {str} -- Desired output path
            seed {int} -- Seed for all number generators
            compile_model {bool} -- Whether to compile the model or not (only PyTorch >= 2.0.0)
            low_mem {bool} -- Whether to use low memory mode or not
            checkpoint_path {str} -- Path to the checkpoint file
        """
        # Start time
        self.start_time = time.time()

        # Handle Ctrl + C event, which aborts and shuts down the training process in a controlled manner
        signal(SIGINT, self.close)

        # Sampled seed if a value smaller than 0 was submitted
        self.seed = seed
        if seed < 0:
            self.seed = random.randint(0, 2 ** 31 - 1)
        set_library_seeds(self.seed)

        # Determine sampling and training devices
        if torch.cuda.is_available():
            # Model optimization on GPU
            self.train_device = torch.device("cuda")
            if low_mem:
                # Sample data on CPU
                self.sample_device = torch.device("cpu")
                torch.set_default_tensor_type("torch.FloatTensor")
            else:
                # Sample data on GPU
                self.sample_device = torch.device("cuda")
                torch.set_default_tensor_type("torch.cuda.FloatTensor")
        else:
            # Sampling and model optimization on CPU
            self.train_device = torch.device("cpu")
            self.sample_device = torch.device("cpu")
            torch.set_default_tensor_type("torch.FloatTensor")

        # Create training monitor
        summaries_path = None
        if checkpoint_path is not None:
            self.timestamp = "/" + re.search(r"(\d{8}-\d{6}_\d+)", checkpoint_path).group(1)
            summaries_path = out_path + "summaries/" + run_id + self.timestamp
            
        self.monitor = TrainingMonitor(out_path, run_id, worker_id, checkpoint_path)
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
            self.trainer = PPOTrainer(configs, self.sample_device, self.train_device, worker_id, run_id, out_path, self.seed, compile_model)
        elif configs["trainer"]["algorithm"] == "DecoupledPPO":
            self.trainer = DecoupledPPOTrainer(configs, self.sample_device, self.train_device, worker_id, run_id, out_path, self.seed, compile_model)
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
        if checkpoint_path is not None:
            self.monitor.log("Load checkpoint: " + checkpoint_path)
            self.trainer.load_checkpoint(checkpoint_path)
        elif configs["model"]["load_model"]:
            self.monitor.log("Load checkpoint: " + configs["model"]["model_path"])
            self.trainer.load_checkpoint(configs["model"]["model_path"])

        # Log the number of trainable parameters of the to-be-optimized model
        self.monitor.log("Number of trainable parameters: " + self.trainer.get_num_trainable_parameters_str())
        if compile_model:
            self.monitor.log("Model will be compiled if applicable")

        # Set variables
        self.configs = configs
        self.resume_at = configs["trainer"]["resume_at"] if summaries_path is None else self._get_last_step(summaries_path) + 1
        self.updates = configs["trainer"]["updates"]
        self.run_id = run_id
        self.worker_id = worker_id
        self.out_path = out_path
        
    def _get_last_step(self, path):
        """
        Returns the last step of the tensorboard event file at the given path.
        
        Arguments:
            path {str} -- Path to the tensorboard event file
        
        Returns:
            {int} -- The last step of the tensorboard event file
        """
        
        # Initialize the EventAccumulator
        event_acc = EventAccumulator(path)

        # Load the EventAccumulator
        event_acc.Reload()

        # List of all scalars available in the EventAccumulator
        # These are referred to as "tags"
        tags = event_acc.Tags()['scalars']

        # Example: Read the last step of the first tag
        # This assumes all tags have the same number of steps
        last_event = event_acc.Scalars(tags[0])[-1]

        # Obtain the last step
        last_step = last_event.step
        
        return last_step


    def run(self):
        """Run training loop"""
        if(self.resume_at > 0):
            self.monitor.log("Step 5: Resuming training at step " + str(self.resume_at) + " using " + str(self.train_device) + " . . .")
        else:
            self.monitor.log("Step 5: Starting training using " + str(self.train_device) + " . . .")

        for update in range(self.resume_at, self.updates):
            # Step trainer
            episode_infos, training_stats, formatted_string, update_duration = self.trainer.step(update)
            # Process training stats to print to console and write summaries
            episode_result = aggregate_episode_results(episode_infos)
            if episode_result:
                self.monitor.log((("{:4} sec={:2} reward={:.2f} std={:.2f} length={:.1f} std={:.2f} ") +
                    (" value={:3f} std={:.3f} adv={:.3f} std={:.3f}")).format(
                    update, update_duration, episode_result["reward_mean"], episode_result["reward_std"],
                    episode_result["length_mean"], episode_result["length_std"], training_stats["value_mean"][1], torch.std(self.trainer.sampler.buffer.values),
                    training_stats["advantage_mean"][1], torch.std(self.trainer.sampler.buffer.advantages)) +
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
            # Write training statistics to tensorboard
            self.monitor.write_training_summary(update, training_stats, episode_result)

    def close(self, signal_received, frame):
        self.monitor.log("Terminating training ...")
        hours, remainder = divmod(time.time() - self.start_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        self.monitor.log("Trainig duration: {:.0f}h {:.0f}m {:.2f}s".format(hours, minutes, seconds))
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
        --config=<path>             Path to the config file [default: ./configs/default.yaml].
        --worker-id=<n>             Sets the port for each environment instance [default: 2].
        --run-id=<path>             Specifies the tag of the tensorboard summaries [default: default].
        --out=<path>                Specifies the path to output files such as summaries and checkpoints. [default: ./]
        --seed=<n>      	        Specifies the seed to use during training. If set to smaller than 0, use a random seed. [default: -1]
        --compile                   Whether to compile the model or not (requires PyTorch >= 2.0.0). [default: False]
        --low-mem                   Whether to move one mini_batch at a time to GPU to save memory [default: False].
        --checkpoint=<path>         Path to a checkpoint to resume training from [default: None].
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
    compile_model = options["--compile"]
    low_mem = options["--low-mem"]
    checkpoint_path = options["--checkpoint"] if options["--checkpoint"] != "None" else None

    # If a run-id was not assigned, use the config's name
    for i, arg in enumerate(sys.argv):
        if "--run-id" in arg:
            run_id = options["--run-id"]
            break
        else:
            run_id = Path(config_path).stem

    # Load environment, model, evaluation and training parameters
    if checkpoint_path is None:
        configs = YamlParser(config_path).get_config()#
    else:
        configs = torch.load(checkpoint_path)["configs"]
        run_id = checkpoint_path.split("/")[2].split("_")[0]

    # Training program
    training = Training(configs, run_id, worker_id, out_path, seed, compile_model, low_mem, checkpoint_path)
    # import cProfile, pstats
    # profiler = cProfile.Profile()
    # profiler.enable()
    try:
        training.run()
    except:
        training.monitor.logger.exception("Exception during training")
    # profiler.disable()
    # stats = pstats.Stats(profiler).sort_stats('cumtime')
    # stats.print_stats()

    # Clean up training
    training.close(None, None)

if __name__ == "__main__":
    main()
    