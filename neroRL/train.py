"""
Runs the training program using the provided config and arguments.
"""

import random

from docopt import docopt

from neroRL.utils.yaml_parser import YamlParser
from neroRL.trainers.policy_gradient.ppo_shared import PPOTrainer
from neroRL.trainers.policy_gradient.ppo_decoupled import DecoupledPPOTrainer

def main():
    # Docopt command line arguments
    _USAGE = """
    Usage:
        train.py [options]
        train.py --help

    Options:
        --config=<path>            Path to the config file [default: ./configs/default.yaml].
        --worker-id=<n>            Sets the port for each environment instance [default: 2].
        --run-id=<path>            Specifies the tag of the tensorboard summaries [default: default].
        --out=<path>               Specifies the path to output files such as summaries and checkpoints. [default: ./]
        --seed=<n>      	       Specifies the seed to use during training. If set to smaller than 0, use a random seed. [default: -1]
    """
    
    options = docopt(_USAGE)
    config_path = options["--config"]
    worker_id = int(options["--worker-id"])
    run_id = options["--run-id"]
    out_path = options["--out"]
    seed = int(options["--seed"])

    # Sampled seed if a value smaller than 0 was submitted
    if seed < 0:
        seed = random.randint(0, 2 ** 31 - 1)

    # Load environment, model, evaluation and training parameters
    configs = YamlParser(config_path).get_config()

    # Initialize trainer
    if configs["trainer"]["algorithm"] == "PPO":
        trainer = PPOTrainer(configs, worker_id, run_id, out_path, seed)
    elif configs["trainer"]["algorithm"] == "DecoupledPPO":
        trainer = DecoupledPPOTrainer(configs, worker_id, run_id, out_path, seed)
    else:
        assert(False), "Unsupported algorithm specified"

    # Start training
    trainer.run_training()

    # Clean up after training
    trainer.close()

if __name__ == "__main__":
    main()
    