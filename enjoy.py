"""
Instantiates an environment and loads a trained model based on the provided config.
The agent environment interaction is then shown in realtime for one episode on a specified or random seed.
TODO: Additionally a video can be rendered.
"""

import logging
import torch
import numpy as np
from docopt import docopt
from gym import spaces

from neroRL.utils.yaml_parser import YamlParser
from neroRL.trainers.PPO.otc_model import OTCModel
from neroRL.environments.unity_wrapper import UnityWrapper
from neroRL.environments.obstacle_tower_wrapper import ObstacleTowerWrapper
from neroRL.environments.minigrid_wrapper import MinigridWrapper
from neroRL.environments.procgen_wrapper import ProcgenWrapper
from neroRL.environments.cartpole_wrapper import CartPoleWrapper
from neroRL.environments.wrapper import wrap_environment
from neroRL.utils.serialization import load_checkpoint

# Setup logger
logging.basicConfig(level = logging.INFO, handlers=[])
logger = logging.getLogger("enjoy")
console = logging.StreamHandler()
console.setFormatter(logging.Formatter("%(asctime)s: %(message)s", "%Y-%m-%d %H:%M:%S"))
logger.addHandler(console)

def main():
    # Docopt command line arguments
    _USAGE = """
    Usage:
        enjoy.py [options]
        enjoy.py --help

    Options:
        --config=<path>            Path of the Config file [default: ./configs/default.yaml].
        --untrained                Whether an untrained model should be used [default: False].
        --worker-id=<n>            Sets the port for each environment instance [default: 2].
        --seed=<n>                 The to be played seed of an episode [default: 0].
    """
    options = docopt(_USAGE)
    untrained = options["--untrained"]
    config_path = options["--config"]
    worker_id = int(options["--worker-id"])
    seed = int(options["--seed"])

    # Load environment, model, evaluation and training parameters
    configs = YamlParser(config_path).get_config()

    # Determine cuda availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Launch environment
    logger.info("Step 1: Launching environment")
    env = wrap_environment(configs["environment"], worker_id, realtime_mode = True)
    # Retrieve observation space
    visual_observation_space = env.visual_observation_space
    vector_observation_space = env.vector_observation_space
    if isinstance(env.action_space, spaces.Discrete):
        action_space_shape = (env.action_space.n,)
    else:
        action_space_shape = tuple(env.action_space.nvec)

    # Build or load model
    logger.info("Step 2: Creating model")
    model = OTCModel(configs["model"], visual_observation_space,
                            vector_observation_space, action_space_shape,
                            configs["model"]["recurrence"] if "recurrence" in configs["model"] else None).to(device)
    if not untrained:
        logger.info("Step 2: Loading model from " + configs["model"]["model_path"])
        checkpoint = load_checkpoint(configs["model"]["model_path"])
        model.load_state_dict(checkpoint["model_state_dict"])
        if "recurrence" in configs["model"]:
            model.set_mean_recurrent_cell_states(checkpoint["hxs"], checkpoint["cxs"])
    model.eval()

    # Reset environment
    logger.info("Step 3: Resetting the environment")
    logger.info("Step 3: Using seed " + str(seed))
    reset_params = configs["environment"]["reset_params"]
    reset_params["start-seed"] = seed
    reset_params["num-seeds"] = 1
    vis_obs, vec_obs = env.reset(reset_params)
    done = False
    
    # Init hidden state (None if not available)
    if "recurrence" in configs["model"]:
        hxs, cxs = model.init_recurrent_cell_states(1, device)
        if configs["model"]["recurrence"]["layer_type"] == "gru":
            recurrent_cell = hxs
        elif configs["model"]["recurrence"]["layer_type"] == "lstm":
            recurrent_cell = (hxs, cxs)
    else:
        recurrent_cell = None

    # Play episode
    logger.info("Step 4: Run single episode in realtime . . .")
    with torch.no_grad():
        while not done:
            # Sample action
            policy, _, recurrent_cell = model(np.expand_dims(vis_obs, 0) if vis_obs is not None else None,
                                np.expand_dims(vec_obs, 0) if vec_obs is not None else None,
                                recurrent_cell,
                                device)

            actions = []
            for action_branch in policy:
                action = action_branch.sample()
                actions.append(action.item())

            vis_obs, vec_obs, _, done, info = env.step(actions)

    logger.info("Episode Reward: " + str(info["reward"]))

if __name__ == "__main__":
    main()
    