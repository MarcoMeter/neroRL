"""
Evaluates an agent based on a configured environment and evaluation.
Additionally each evaluation episode can be rendered as a video.
Either a checkpoint or a config have to be provided. Whenever a checkpoint is provided, its model config is used to
instantiate the model. A checkpoint can be also provided via the config.
"""

import logging
import torch
import numpy as np
import sys

from docopt import docopt

from neroRL.utils.utils import aggregate_episode_results, get_environment_specs, load_and_apply_state_dict
from neroRL.utils.yaml_parser import YamlParser
from neroRL.evaluator import Evaluator
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
        --config=<path>            Path to the config file [default: ].
        --checkpoint=<path>        Path to the checkpoint file [default: ].
        --untrained                Whether an untrained model should be used [default: False].
        --worker-id=<n>            Sets the port for each environment instance [default: 2].
        --video=<path>             Specify a path for saving videos, if video recording is desired. The files' extension will be set automatically. [default: ./video].
        --start-seed=<n>           Specifies the start of the seed range [default: 200000].
        --num-seeds=<n>            Specifies the number of seeds to evaluate [default: 50].
        --repetitions=<n>          Specifies the number of repetitions for each seed [default: 3].
    """
    options = docopt(_USAGE)
    config_path = options["--config"]           # defaults to ""
    checkpoint_path = options["--checkpoint"]   # defaults to ""
    untrained = options["--untrained"]  	    # defaults to False
    worker_id = int(options["--worker-id"])     # defaults to 2
    video_path = options["--video"]             # Defaults to "./video"

    # Determine whether a seed configuration was passed as argument, if not, use the one from the config file
    override_seed_config = False
    args = " ".join(sys.argv)
    if "--start-seed" in args and "--num-seeds" in args and "--repetitions" in args:
        override_seed_config = True

    # Determine whether to record a video. A video is only recorded if the video flag is used.
    record_video = "--video" in args
    if record_video:
        logger.info("Step 0: Video recording enabled. Video will be saved to " + video_path)
    
    # Determine cuda availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
    else:
        torch.set_default_tensor_type("torch.FloatTensor")

    if not config_path and not checkpoint_path:
        raise ValueError("Either a config or a checkpoint must be provided")
    checkpoint = torch.load(checkpoint_path, map_location=device) if checkpoint_path else None
    configs = YamlParser(config_path).get_config() if config_path else checkpoint["configs"]
    model_config = checkpoint["configs"]["model"] if checkpoint else configs["model"]

    # Override seed configuration if necessary
    if override_seed_config:
        if "explicit-seeds" in configs["evaluation"]["seeds"]:
            del configs["evaluation"]["seeds"]["explicit-seeds"]
        configs["evaluation"]["seeds"]["start-seed"] = int(options["--start-seed"])
        configs["evaluation"]["seeds"]["num-seeds"] = int(options["--num-seeds"])
        configs["evaluation"]["n_workers"] = int(options["--repetitions"])

    # Create dummy environment to retrieve the shapes of the observation and action space for further processing
    logger.info("Step 1: Creating dummy environment of type " + configs["environment"]["type"])
    visual_observation_space, vector_observation_space, ground_truth_space, action_space_shape, max_episode_steps = get_environment_specs(configs["environment"], worker_id - 1)

    # Build or load model
    logger.info("Step 2: Creating model")
    model = create_actor_critic_model(model_config, visual_observation_space,
                            vector_observation_space, ground_truth_space, action_space_shape, device)
    if not untrained:
        if not checkpoint:
            # If a checkpoint is not provided as an argument, it shall be retrieved from the config
            logger.info("Step 2: Loading model from " + model_config["model_path"])
            checkpoint = torch.load(model_config["model_path"], map_location=device)
        model = load_and_apply_state_dict(model, checkpoint["model"])
        if "recurrence" in model_config:
            model.set_mean_recurrent_cell_states(checkpoint["hxs"], checkpoint["cxs"])
    model.eval()

    # Initialize evaluator
    logger.info("Step 3: Initialize evaluator")
    logger.info("Step 3: Number of Workers: " + str(configs["evaluation"]["n_workers"]))
    logger.info("Step 3: Seeds: " + str(configs["evaluation"]["seeds"]))
    logger.info("Step 3: Number of episodes: " + str(configs["evaluation"]["seeds"]["num-seeds"] * configs["evaluation"]["n_workers"]))
    evaluator = Evaluator(configs, model_config, worker_id, visual_observation_space, vector_observation_space,
                            max_episode_steps, video_path, record_video)

    # Evaluate
    logger.info("Step 4: Run evaluation . . .")
    eval_duration, raw_episode_results = evaluator.evaluate(model, device)
    episode_result = aggregate_episode_results(raw_episode_results)

    # Print results
    logger.info("RESULT: sec={:3}     mean reward={:.2f} std={:.2f}     mean length={:.1f} std={:.2f}".format(
        eval_duration, episode_result["reward_mean"], episode_result["reward_std"], episode_result["length_mean"], episode_result["length_std"]))

    # Close
    logger.info("Step 5: Closing evaluator . . .")
    evaluator.close()

if __name__ == "__main__":
    main()
    