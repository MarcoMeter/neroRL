"""
Evaluates an agent based on a configured environment and evaluation.
Additionally each evaluation episode can be rendered as a video. 
"""

import logging
from tabnanny import check
import torch
import numpy as np
import sys

from docopt import docopt
from gym import spaces

from neroRL.utils.yaml_parser import YamlParser
from neroRL.evaluator import Evaluator
from neroRL.environments.wrapper import wrap_environment
from neroRL.nn.actor_critic import create_actor_critic_model

# Setup logger
logging.basicConfig(level = logging.INFO, handlers=[])
logger = logging.getLogger("eval")
console = logging.StreamHandler()
console.setFormatter(logging.Formatter("%(asctime)s: %(message)s", "%Y-%m-%d %H:%M:%S"))
logger.addHandler(console)

def main():
    # Docopt command line arguments
    _USAGE = """
    Usage:
        neval [options]
        neval --help

    Options:
        --config=<path>            Path to the config file [default: "None"].
        --checkpoint=<path>        Path to the checkpoint file [default: "None"].
        --untrained                Whether an untrained model should be used [default: False].
        --worker-id=<n>            Sets the port for each environment instance [default: 2].
        --video=<path>             Specify a path for saving videos, if video recording is desired. The files' extension will be set automatically. [default: ./video].
    """
    options = docopt(_USAGE)
    config_path = options["--config"]
    checkpoint_path = options["--checkpoint"]
    untrained = options["--untrained"]
    worker_id = int(options["--worker-id"])
    video_path = options["--video"]

    # Determine whether to record a video. A video is only recorded if the video flag is used.
    record_video = False
    for i, arg in enumerate(sys.argv):
        if "--video" in arg:
            record_video = True
            logger.info("Step 0: Video recording enabled. Video will be saved to " + video_path)
            break
    
    # Determine cuda availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
    else:
        torch.set_default_tensor_type("torch.FloatTensor")

    if config_path == "None" and checkpoint_path == "None":
        raise ValueError("Either a config or a checkpoint must be provided")
    checkpoint = torch.load(checkpoint_path) if checkpoint_path != "None" else None
    configs = checkpoint["configs"] if config_path != "None" else YamlParser(config_path).get_config()

    # Create dummy environment to retrieve the shapes of the observation and action space for further processing
    logger.info("Step 1: Creating dummy environment of type " + configs["environment"]["type"])
    dummy_env = wrap_environment(configs["environment"], worker_id)
    visual_observation_space = dummy_env.visual_observation_space
    vector_observation_space = dummy_env.vector_observation_space
    if isinstance(dummy_env.action_space, spaces.Discrete):
        action_space_shape = (dummy_env.action_space.n,)
    else:
        action_space_shape = tuple(dummy_env.action_space.nvec)
    dummy_env.close()

    # Build or load model
    logger.info("Step 2: Creating model")
    share_parameters = False
    if configs["trainer"]["algorithm"] == "PPO":
        share_parameters = configs["trainer"]["algorithm"]
    model = create_actor_critic_model(configs["model"], share_parameters, visual_observation_space,
                            vector_observation_space, action_space_shape,
                            configs["model"]["recurrence"] if "recurrence" in configs["model"] else None, device)
    if "DAAC" in configs["trainer"]:
        model.add_gae_estimator_head(action_space_shape, device)
    if not untrained:
        logger.info("Step 2: Loading model from " + configs["model"]["model_path"])
        model.load_state_dict(checkpoint["model"])
        if "recurrence" in configs["model"]:
            model.set_mean_recurrent_cell_states(checkpoint["hxs"], checkpoint["cxs"])
    model.eval()

    # Initialize evaluator
    logger.info("Step 3: Initialize evaluator")
    logger.info("Step 3: Number of Workers: " + str(configs["evaluation"]["n_workers"]))
    logger.info("Step 3: Seeds: " + str(configs["evaluation"]["seeds"]))
    logger.info("Step 3: Number of episodes: " + str(len(configs["evaluation"]["seeds"]) * configs["evaluation"]["n_workers"]))
    evaluator = Evaluator(configs, worker_id, visual_observation_space, vector_observation_space, video_path, record_video)

    # Evaluate
    logger.info("Step 4: Run evaluation . . .")
    eval_duration, raw_episode_results = evaluator.evaluate(model, device)
    episode_result = _process_episode_info(raw_episode_results)

    # Print results
    logger.info("RESULT: sec={:3}     mean reward={:.2f} std={:.2f}     mean length={:.1f} std={:.2f}".format(
        eval_duration, episode_result["reward_mean"], episode_result["reward_std"], episode_result["length_mean"], episode_result["length_std"]))

    # Close
    logger.info("Step 5: Closing evaluator . . .")
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
    