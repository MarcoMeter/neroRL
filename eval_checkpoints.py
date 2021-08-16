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
import torch
import os
import time
import pickle
import numpy as np
from docopt import docopt
from gym import spaces

from neroRL.utils.yaml_parser import YamlParser
from neroRL.trainers.PPO.evaluator import Evaluator
from neroRL.environments.wrapper import wrap_environment
from neroRL.trainers.PPO.models.actor_critic import create_actor_critic_model
from neroRL.utils.serialization import load_checkpoint

def main():
    # Docopt command line arguments
    _USAGE = """
    Usage:
        evaluate.py [options]
        evaluate.py --help

    Options:
        --config=<path>            Path to the config file [default: ./configs/default.yaml].
        --worker-id=<n>            Sets the port for each environment instance [default: 2].
        --path=<path>              Specifies the tag of the tensorboard summaries [default: None].
        --name=<path>              Specifies the full path to save the output file [default: results.res].
    """
    options = docopt(_USAGE)
    config_path = options["--config"]
    worker_id = int(options["--worker-id"])
    path = options["--path"]
    name = options["--name"]

    # Load environment, model, evaluation and training parameters
    configs = YamlParser(config_path).get_config()

    # Determine cuda availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create dummy environment to retrieve the shapes of the observation and action space for further processing
    print("Step 1: Creating dummy environment of type " + configs["environment"]["type"])
    dummy_env = wrap_environment(configs["environment"], worker_id)
    visual_observation_space = dummy_env.visual_observation_space
    vector_observation_space = dummy_env.vector_observation_space
    if isinstance(dummy_env.action_space, spaces.Discrete):
        action_space_shape = (dummy_env.action_space.n,)
    else:
        action_space_shape = tuple(dummy_env.action_space.nvec)
    dummy_env.close()

    # Init evaluator
    print("Step 1: Environment Config")
    for k, v in configs["environment"].items():
        print("Step 1: " + str(k) + ": " + str(v))
    print("Step 2: Evaluation Config")
    for k, v in configs["evaluation"].items():
        print("Step 2: " + str(k) + ": " + str(v))
    print("Step 2: Init Evaluator")
    evaluator = Evaluator(configs, worker_id, visual_observation_space, vector_observation_space)

    # Init model
    print("Step 2: Initialize model")
    model = create_actor_critic_model(configs["model"], visual_observation_space,
                            vector_observation_space, action_space_shape,
                            configs["model"]["recurrence"], device)
    model.eval()

    # Load checkpoint paths
    print("Step 4: Load Checkpoint Paths")
    checkpoints = get_sorted_checkpoints(path)
    print("Step 3: Number of Loaded Checkpoint Paths: " + str(len(checkpoints)))

    # Evaluate checkpoints
    print("Step 5: Start Evaluation . . .")
    print("Progress:")
    results = []
    current_checkpoint = 0
    for checkpoint in checkpoints:
        loaded_checkpoint = load_checkpoint(checkpoint)
        model.load_state_dict(loaded_checkpoint["model_state_dict"])
        if "recurrence" in configs["model"]:
            model.set_mean_recurrent_cell_states(loaded_checkpoint["hxs"], loaded_checkpoint["cxs"])
        _, res = evaluator.evaluate(model, device)
        results.append(res)
        current_checkpoint = current_checkpoint + 1
        prog = current_checkpoint / len(checkpoints)
        print(f"\r{prog:.2f}", end='', flush=True)
    evaluator.close()

    # Save results to file
    print("Step 6: Save to File: " + name)
    results = np.asarray(results).reshape(len(checkpoints), len(configs["evaluation"]["seeds"]), configs["evaluation"]["n_workers"])
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
    