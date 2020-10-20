"""
Instantiates an environment and loads a trained model based on the provided config.
The agent environment interaction is then shown in realtime for one episode on a specified or random seed.
TODO: Additionally a video can be rendered.
"""

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
    print("Step 1: Launching environment")
    env = wrap_environment(configs["environment"], worker_id, realtime_mode = True)
    
    # Retrieve observation space
    visual_observation_space = env.visual_observation_space
    vector_observation_space = env.vector_observation_space
    if isinstance(env.action_space, spaces.Discrete):
        action_space_shape = (env.action_space.n,)
    else:
        action_space_shape = tuple(env.action_space.nvec)

    # Build or load model
    print("Step 2: Creating model")
    model = OTCModel(configs["model"], visual_observation_space,
                            vector_observation_space, action_space_shape,
                            configs["model"]["use_recurrent"],
                            configs["model"]["hidden_state_size"]).to(device)
    if not untrained:
        print("Step 2: Loading model from " + configs["model"]["model_path"])
        checkpoint = load_checkpoint(configs["model"]["model_path"])
        model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Reset environment
    print("Step 3: Resetting the environment")
    print("Step 3: Using seed " + str(seed))
    vis_obs, vec_obs = env.reset({"start-seed": seed, "num-seeds": 1})
    done = False
    # Init hidden state (None if not available)
    if model.use_recurrent:
        hidden_state = torch.zeros((1, model.hidden_state_size), dtype=torch.float32, device=device)
    else:
        hidden_state = None

    # Play episode
    print("Step 4: Run single episode in realtime . . .")
    with torch.no_grad():
        while not done:
            # Sample action
            policy, _, hidden_state = model(np.expand_dims(vis_obs, 0) if vis_obs is not None else None,
                                np.expand_dims(vec_obs, 0) if vec_obs is not None else None,
                                hidden_state,
                                device)

            actions = []
            for action_branch in policy:
                action = action_branch.sample()
                actions.append(action.item())

            vis_obs, vec_obs, _, done, info = env.step(actions)

    print("Episode Reward: " + str(info["reward"]))

if __name__ == "__main__":
    main()
    