"""
Evaluates an agent based on a configurated environment and evaluation.
"""

import torch
import numpy as np
from docopt import docopt
from gym import spaces

from neroRL.utils.yaml_parser import YamlParser
from neroRL.trainers.PPO.evaluator import Evaluator
from neroRL.environments.wrapper import wrap_environment
from neroRL.trainers.PPO.otc_model import OTCModel

def main():
    # Docopt command line arguments
    _USAGE = """
    Usage:
        evaluate.py [options]
        evaluate.py --help

    Options:
        --config=<path>            Path of the Config file [default: ./configs/default.yaml].
        --untrained                Whether an untrained model should be used [default: False].
        --worker-id=<n>            Sets the port for each environment instance [default: 2].
        --run-id=<path>            Specifies the tag of the tensorboard summaries [default: default].
    """
    options = docopt(_USAGE)
    untrained = options["--untrained"]
    config_path = options["--config"]
    worker_id = int(options["--worker-id"])
    run_id = options["--run-id"]

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

    # Build or load model
    if untrained:
        print("Step 2: Creating model")
        model = OTCModel(configs["model"], visual_observation_space,
                                vector_observation_space, action_space_shape,
                                configs["model"]["use_recurrent"],
                                configs["model"]["hidden_state_size"]).to(device)
    else:
        print("Step 2: Loading model from " + configs["model"]["model_path"])
        model = torch.load(configs["model"]["model_path"]).to(device)
    model.eval()

    # Initialize evaluator
    print("Step 3: Initialize evaluator")
    print("Step 3: Number of Workers: " + str(configs["evaluation"]["n_workers"]))
    print("Step 3: Seeds: " + str(configs["evaluation"]["seeds"]))
    print("Step 3: Number of episodes: " + str(len(configs["evaluation"]["seeds"]) * configs["evaluation"]["n_workers"]))
    evaluator = Evaluator(configs["evaluation"], configs["environment"], worker_id, visual_observation_space, vector_observation_space)

    # Evaluate
    print("Step 4: Run evaluation . . .")
    eval_duration, raw_episode_results = evaluator.evaluate(model, device)
    episode_result = _process_episode_info(raw_episode_results)

    # Print results
    print("RESULT: sec={:3}     mean reward={:.2f} std={:.2f}     mean length={:.1f} std={:.2f}".format(
        eval_duration, episode_result["reward_mean"], episode_result["reward_std"], episode_result["length_mean"], episode_result["length_std"]))

    # Close
    print("Step 5: Closing evaluator . . .")
    evaluator.close()

def _process_episode_info(episode_info):
    """Extracts the mean and std of completed episodes. At minimum the episode length and the collected reward is available.
    
    Arguments:
        episode_info {list} -- List of episode information, each individual item is a dictionary

    Returns:
        result {dict} -- Dictionary that contains the mean, std, min and max of all episode infos        
    """
    result = {}
    if len(episode_info) > 0:
        keys = episode_info[0].keys()
        # Compute mean and std for each information, skip seed
        for key in keys:
            if key == "seed":
                continue
            result[key + "_mean"] = np.mean([info[key] for info in episode_info])
            result[key + "_min"] = np.min([info[key] for info in episode_info])
            result[key + "_max"] = np.max([info[key] for info in episode_info])
            result[key + "_std"] = np.std([info[key] for info in episode_info])
    return result

if __name__ == "__main__":
    main()
    