"""
Instantiates an environment and loads a trained model based on the provided config.
The agent environment interaction is then shown in realtime for one episode on a specified seed.
Additionally a video can be rendered. Alternatively a website visualizing more properties, such as the value function,
can be generated.
"""

import logging
import torch
import numpy as np
import sys

from docopt import docopt
from gym import spaces

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
        nenjoy [options]
        nenjoy --help

    Options:
        --config=<path>            Path to the config file [default: "./configs/default.yaml"].
        --checkpoint=<path>        Path to the checkpoint file [default: "None"].
        --untrained                Whether an untrained model should be used [default: False].
        --worker-id=<n>            Sets the port for each environment instance [default: 2].
        --seed=<n>                 The to be played seed of an episode [default: 0].
        --num-episodes=<n>         The number of to be played episodes [default: 1].
        --video=<path>             Specify a path for saving a video, if video recording is desired. The file's extension will be set automatically. [default: ./video].
        --framerate=<n>            Specifies the frame rate of the to be rendered video. [default: 6]
        --generate_website         Specifies wether a website shall be generated. [default: False]
    """
    options = docopt(_USAGE)
    untrained = options["--untrained"]
    config_path = options["--config"]
    checkpoint_path = options["--checkpoint"]
    worker_id = int(options["--worker-id"])
    seed = int(options["--seed"])
    num_episodes = int(options["--num-episodes"])
    video_path = options["--video"]
    frame_rate = options["--framerate"]
    generate_website = options["--generate_website"]

    # Determine whether to record a video. A video is only recorded if the video flag is used.
    record_video = False
    for i, arg in enumerate(sys.argv):
        if "--video" in arg:
            record_video = True
            logger.info("Step 0: Video recording enabled. Video will be saved to " + video_path)
            logger.info("Step 0: Only 1 episode will be played")
            num_episodes = 1
            break

    if generate_website:
        logger.info("Step 0: Only 1 episode will be played")
        num_episodes = 1

    # Determine cuda availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
    else:
        torch.set_default_tensor_type("torch.FloatTensor")

    # Load config, environment, model, evaluation and training parameters
    checkpoint = torch.load(checkpoint_path) if checkpoint_path != "None" else None
    configs = checkpoint["configs"] if checkpoint_path != "None" else YamlParser(config_path).get_config()

    # Launch environment
    logger.info("Step 1: Launching environment")
    configs["environment"]["reset_params"]["start-seed"] = seed
    configs["environment"]["reset_params"]["num-seeds"] = 1
    configs["environment"]["reset_params"]["seed"] = seed
    env = wrap_environment(configs["environment"], worker_id, realtime_mode = True, record_trajectory = record_video or generate_website)
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
    if configs["trainer"]["algorithm"] == "PPO":
        share_parameters = configs["trainer"]["share_parameters"]
    model = create_actor_critic_model(configs["model"], share_parameters, visual_observation_space,
                            vector_observation_space, action_space_shape,
                            configs["model"]["recurrence"] if "recurrence" in configs["model"] else None, device)
    if "DAAC" in configs["trainer"]:
        model.add_gae_estimator_head(action_space_shape, device)
    if not untrained:
        logger.info("Step 2: Loading model from " + configs["model"]["model_path"])
        checkpoint = torch.load(configs["model"]["model_path"])
        model.load_state_dict(checkpoint["model"])
        if "recurrence" in configs["model"]:
            model.set_mean_recurrent_cell_states(checkpoint["hxs"], checkpoint["cxs"])
    model.eval()

    # Run all desired episodes
    # Note: Only one episode is run upon generating a result website or rendering a video
    for _ in range(num_episodes):
        # Reset environment
        logger.info("Step 3: Resetting the environment")
        logger.info("Step 3: Using seed " + str(seed))
        vis_obs, vec_obs = env.reset(configs["environment"]["reset_params"])
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
        logger.info("Step 4: Run " + str(num_episodes) + " episode(s) in realtime . . .")

        # Store data for video recording
        probs = []
        entropies = []
        values = []
        actions = []

        # Play one episode
        with torch.no_grad():
            while not done:
                # Forward the neural net
                vis_obs = torch.tensor(np.expand_dims(vis_obs, 0), dtype=torch.float32, device=device) if vis_obs is not None else None
                vec_obs = torch.tensor(np.expand_dims(vec_obs, 0), dtype=torch.float32, device=device) if vec_obs is not None else None
                policy, value, recurrent_cell, _ = model(vis_obs, vec_obs, recurrent_cell)

                _actions = []
                _probs = []
                entropy = []
                # Sample action
                for action_branch in policy:
                    action = action_branch.sample()
                    _actions.append(action.item())
                    _probs.append(action_branch.probs)
                    entropy.append(action_branch.entropy().item())

                # Store data for video recording
                actions.append(_actions)
                probs.append(torch.stack(_probs))
                entropies.append(entropy)
                values.append(value.cpu().numpy())

                # Step environment
                vis_obs, vec_obs, _, done, info = env.step(_actions)

        logger.info("Episode Reward: " + str(info["reward"]))
        logger.info("Episode Length: " + str(info["length"]))

        # Complete video data
        if record_video or generate_website:
            trajectory_data = env.get_episode_trajectory
            trajectory_data["action_names"] = env.action_names
            trajectory_data["actions"] = actions
            trajectory_data["probs"] = probs
            trajectory_data["entropies"] = entropies
            trajectory_data["values"] = values
            trajectory_data["episode_reward"] = info["reward"]
            trajectory_data["seed"] = seed
            # Init video recorder
            video_recorder = VideoRecorder(video_path, frame_rate)
            # Render and serialize video
            if record_video:
                video_recorder.render_video(trajectory_data)
            # Generate website
            if generate_website:
                video_recorder.generate_website(trajectory_data, configs)

    env.close()

if __name__ == "__main__":
    main()
    