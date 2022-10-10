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
from numpy.core.fromnumeric import mean
import torch
import os
import pickle
import numpy as np
from docopt import docopt
from gym import spaces

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
        --checkpoints=<path>       Path to the directory containing checkpoints [default: "None"].
        --config=<path>            Path to the config file [default: "None"].
        --name=<path>              Specifies the full path to save the output file [default: ./results.res].
    """
    options = docopt(_USAGE)
    worker_id = int(options["--worker-id"])
    checkpoints_path = options["--checkpoints"]
    config_path = options["--config"]
    name = options["--name"]

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
    checkpoint = torch.load(checkpoints[0])
    configs = checkpoint["configs"] if config_path != "None" else YamlParser(config_path).get_config()

    # Create dummy environment to retrieve the shapes of the observation and action space for further processing
    print("Step 2: Creating dummy environment of type " + configs["environment"]["type"])
    dummy_env = wrap_environment(configs["environment"], worker_id)
    visual_observation_space = dummy_env.visual_observation_space
    vector_observation_space = dummy_env.vector_observation_space
    if isinstance(dummy_env.action_space, spaces.Discrete):
        action_space_shape = (dummy_env.action_space.n,)
    else:
        action_space_shape = tuple(dummy_env.action_space.nvec)
    dummy_env.close()
    
    # Init evaluator
    print("Step 2: Environment Config")
    for k, v in configs["environment"].items():
        print("Step 2: " + str(k) + ": " + str(v))
    print("Step 3: Evaluation Config")
    for k, v in configs["evaluation"].items():
        print("Step 3: " + str(k) + ": " + str(v))
    print("Step 3: Init Evaluator")
    evaluator = Evaluator(configs, worker_id, visual_observation_space, vector_observation_space)

    # Init model
    print("Step 3: Initialize model")
    share_parameters = False
    if configs["trainer"]["algorithm"] == "PPO":
        share_parameters = configs["trainer"]["share_parameters"]
    model = create_actor_critic_model(configs["model"], share_parameters, visual_observation_space,
                            vector_observation_space, action_space_shape,
                            configs["model"]["recurrence"] if "recurrence" in configs["model"] else None, device)
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
        loaded_checkpoint = torch.load(checkpoint)
        model.load_state_dict(loaded_checkpoint["model"])
        if "recurrence" in configs["model"]:
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
    results = np.asarray(results).reshape(len(checkpoints), len(configs["evaluation"]["seeds"]), configs["evaluation"]["n_workers"])
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
    