"""
This eval script evaluates all available models inside one directory.

The shape of the ouput data is as follows:
(Checkpoint, Seed, Worker)

Each data point is a dictionary given the episode information:
    - reward
    - length
    - seed

Depending on the environment, more information might be available.
For example, Obstacle Tower has a floor key inside that dictionary.
"""
import sys
import torch
import os
import pickle
import numpy as np
from docopt import docopt

from neroRL.utils.utils import get_environment_specs
from neroRL.utils.yaml_parser import YamlParser
from neroRL.evaluator import Evaluator
from neroRL.environments.wrapper import wrap_environment
from neroRL.nn.actor_critic import create_actor_critic_model

def main():
    # Docopt command line arguments
    _USAGE = """
    Usage:
        neval-checkpoints [options]
        neval-checkpoints --help

    Options:
        --worker-id=<n>            Sets the port for each environment instance [default: 2].
        --checkpoints=<path>       Path to the directory containing checkpoints [default: ].
        --config=<path>            Path to the config file [default: ].
        --name=<path>              Specifies the full path to save the output file [default: ./results.res].
        --start-seed=<n>           Specifies the start of the seed range [default: 200000].
        --num-seeds=<n>            Specifies the number of seeds to evaluate [default: 50].
        --repetitions=<n>          Specifies the number of repetitions for each seed [default: 3].
    """
    options = docopt(_USAGE)
    worker_id = int(options["--worker-id"])         # defaults to 2.
    checkpoints_path = options["--checkpoints"]     # defaults to ""
    config_path = options["--config"]               # defaults to ""
    name = options["--name"]                        # defaults to "result.res

    # Determine whether a seed configuration was passed as argument, if not, use the one from the config file
    override_seed_config = False
    args = " ".join(sys.argv)
    if "--start-seed" in args and "--num-seeds" in args and "--repetitions" in args:
        override_seed_config = True

    # Determine cuda availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
    else:
        torch.set_default_tensor_type("torch.FloatTensor")
        
    # Load checkpoint paths
    print("Step 1: Load Checkpoint Paths")
    checkpoints = get_sorted_checkpoints(checkpoints_path)
    print("Step 1: Number of Loaded Checkpoint Paths: " + str(len(checkpoints)))
    
    # Load config, environment, model, evaluation and training parameters
    if len(checkpoints) == 0:
        print("No checkpoints found in the given directory. Exiting...")
        return 0
    checkpoint = torch.load(checkpoints[0], map_location=device)
    model_config = checkpoint["configs"]["model"]
    configs = YamlParser(config_path).get_config() if config_path else checkpoint["configs"]

    # Override seed configuration if necessary
    if override_seed_config:
        if "explicit-seeds" in configs["evaluation"]["seeds"]:
            del configs["evaluation"]["seeds"]["explicit-seeds"]
        configs["evaluation"]["seeds"]["start-seed"] = int(options["--start-seed"])
        configs["evaluation"]["seeds"]["num-seeds"] = int(options["--num-seeds"])
        configs["evaluation"]["n_workers"] = int(options["--repetitions"])

    # Create dummy environment to retrieve the shapes of the observation and action space for further processing
    print("Step 2: Creating dummy environment of type " + configs["environment"]["type"])
    visual_observation_space, vector_observation_space, ground_truth_space, action_space_shape, max_episode_steps = get_environment_specs(configs["environment"], worker_id - 1)
    
    # Init evaluator
    print("Step 2: Environment Config")
    for k, v in configs["environment"].items():
        print("Step 2: " + str(k) + ": " + str(v))
    print("Step 3: Evaluation Config")
    for k, v in configs["evaluation"].items():
        print("Step 3: " + str(k) + ": " + str(v))
    print("Step 3: Init Evaluator")
    evaluator = Evaluator(configs, model_config, worker_id, visual_observation_space, vector_observation_space, max_episode_steps)

    # Init model
    print("Step 3: Initialize model")
    share_parameters = False
    if configs["trainer"]["algorithm"] == "PPO":
        share_parameters = configs["trainer"]["share_parameters"]
    model = create_actor_critic_model(model_config, share_parameters, visual_observation_space,
                            vector_observation_space, ground_truth_space, action_space_shape, device)
    if "DAAC" in configs["trainer"]:
        model.add_gae_estimator_head(action_space_shape, device)
    model.eval()

    if torch.cuda.is_available():
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
        model.cuda()
    else:
        torch.set_default_tensor_type("torch.FloatTensor")
        model.cpu()

    # Evaluate checkpoints
    print("Step 4: Start Evaluation . . .")
    print("Progress:")
    results = []
    current_checkpoint = 0
    for checkpoint in checkpoints:
        loaded_checkpoint = torch.load(checkpoint, map_location=device)
        model.load_state_dict(loaded_checkpoint["model"])
        if "recurrence" in model_config:
            model.set_mean_recurrent_cell_states(loaded_checkpoint["hxs"], loaded_checkpoint["cxs"])
        _, res = evaluator.evaluate(model, device)
        results.append(res)
        current_checkpoint = current_checkpoint + 1
        prog = current_checkpoint / len(checkpoints)
        mean_reward = 0.0
        mean_length = 0.0
        for info in res:
            mean_reward += info["reward"]
            mean_length += info["length"]
        mean_reward = mean_reward / len(res)
        mean_length = mean_length / len(res)
        print(f"\r{prog:.2f} mean reward: {mean_reward:.2f} mean episode length: {mean_length:.1f}", end='', flush=True)
    evaluator.close()

    # Save results to file
    print("\nStep 5: Save to File: " + name)
    if "explicit-seeds" in configs["evaluation"]["seeds"]:
        num_seeds = len(configs["evaluation"]["seeds"]["explicit-seeds"])
    else:
        num_seeds = configs["evaluation"]["seeds"]["num-seeds"]
    results = np.asarray(results).reshape(len(checkpoints), num_seeds, configs["evaluation"]["n_workers"])
    os.makedirs(os.path.dirname(name), exist_ok=True)
    outfile = open(name, "wb")
    pickle.dump(results, outfile)
    outfile.close()

def get_sorted_checkpoints(dirpath):
    """Generates the full file paths to each checkpoint and sorts them alphabetically.

    Arguments:
        dirpath {string} -- Path to the directory containing the checkpoints

    Returns:
        {list} -- List that containts the full file path to each checkpoint
    """
    a = [s for s in os.listdir(dirpath)
         if os.path.isfile(os.path.join(dirpath, s))]
    a.sort(key=lambda s: os.path.getmtime(os.path.join(dirpath, s)))
    for i, f in enumerate(a):
        a[i] = os.path.join(dirpath, f)
    return a

if __name__ == "__main__":
    main()
    