"""
Instantiates an environment and loads a trained model based on the provided config.
The agent environment interaction is then shown in realtime for one episode on a specified seed.
Additionally a video can be rendered.
"""

import logging
import torch
import numpy as np
from docopt import docopt
from gym import spaces
import sys

from neroRL.utils.yaml_parser import YamlParser
from neroRL.environments.wrapper import wrap_environment
from neroRL.utils.video_recorder import VideoRecorder
from neroRL.nn.actor_critic import create_actor_critic_model

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
        --video=<path>             Specify a path for saving a video, if video recording is desired. The file's extension will be set automatically. [default: ./video].
        --framerate=<n>            Specifies the frame rate of a video shall be rendered. [default: 6]
    """
    options = docopt(_USAGE)
    untrained = options["--untrained"]
    config_path = options["--config"]
    worker_id = int(options["--worker-id"])
    seed = int(options["--seed"])
    video_path = options["--video"]
    frame_rate = options["--framerate"]

    # Determine whether to record a video. A video is only recorded if the video flag is used.
    record_video = False
    for i, arg in enumerate(sys.argv):
        if "--video" in arg:
            record_video = True
            logger.info("Step 0: Video recording enabled. Video will be saved to " + video_path)
            break

    # Load environment, model, evaluation and training parameters
    configs = YamlParser(config_path).get_config()

    # Determine cuda availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Launch environment
    logger.info("Step 1: Launching environment")
    env = wrap_environment(configs["environment"], worker_id, realtime_mode = True, record_trajectory = record_video)
    # Retrieve observation space
    visual_observation_space = env.visual_observation_space
    vector_observation_space = env.vector_observation_space
    if isinstance(env.action_space, spaces.Discrete):
        action_space_shape = (env.action_space.n,)
    else:
        action_space_shape = tuple(env.action_space.nvec)

    # Build or load model
    logger.info("Step 2: Creating model")
    share_parameters = False
    if configs["trainer"] == "PPO":
        share_parameters = configs["trainers"]["share_parameters"]
    model = create_actor_critic_model(configs["model"], share_parameters, visual_observation_space,
                            vector_observation_space, action_space_shape,
                            configs["model"]["recurrence"] if "recurrence" in configs["model"] else None, device)
    if not untrained:
        logger.info("Step 2: Loading model from " + configs["model"]["model_path"])
        checkpoint = torch.load(configs["model"]["model_path"])
        model.load_state_dict(checkpoint["model"])
        if "recurrence" in configs["model"]:
            model.set_mean_recurrent_cell_states(checkpoint["hxs"], checkpoint["cxs"])
    model.eval()

    # Reset environment
    logger.info("Step 3: Resetting the environment")
    logger.info("Step 3: Using seed " + str(seed))
    reset_params = configs["environment"]["reset_params"]
    reset_params["seed"] = seed
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

    # Store data for video recording
    log_probs = []
    entropies = []
    values = []
    actions = []

    with torch.no_grad():
        while not done:
            # Forward the neural net
            policy, value, recurrent_cell = model(np.expand_dims(vis_obs, 0) if vis_obs is not None else None,
                                np.expand_dims(vec_obs, 0) if vec_obs is not None else None,
                                recurrent_cell,
                                device)

            _actions = []
            probs = []
            entropy = []
            # Sample action
            for action_branch in policy:
                action = action_branch.sample()
                _actions.append(action.item())
                probs.append(action_branch.probs)
                entropy.append(action_branch.entropy().item())

            # Store data for video recording
            actions.append(_actions)
            log_probs.append(probs)
            entropies.append(entropy)
            values.append(value)

            # Step environment
            vis_obs, vec_obs, _, done, info = env.step(_actions)

    logger.info("Episode Reward: " + str(info["reward"]))

    # Complete video data
    if record_video:
        trajectory_data = env.get_episode_trajectory
        trajectory_data["action_names"] = env.action_names
        trajectory_data["actions"] = actions
        trajectory_data["log_probs"] = log_probs
        trajectory_data["entropies"] = entropies
        trajectory_data["values"] = values
        trajectory_data["episode_reward"] = info["reward"]
        trajectory_data["seed"] = seed
        # Init video recorder
        video_recorder = VideoRecorder(video_path, frame_rate)
        # Render and serialize video
        video_recorder.render_video(trajectory_data)

    env.close()

if __name__ == "__main__":
    main()
    