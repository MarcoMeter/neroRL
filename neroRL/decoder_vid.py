"""
Instantiates an environment and loads a trained model based on the provided config and checkpoint.
Either a checkpoint or a config have to be provided. Whenever a checkpoint is provided, its model config is used to
instantiate the model. A checkpoint can be also provided via the config.
The agent environment interaction is then shown in realtime for one episode on a specified seed.
Additionally a video can be rendered. Alternatively a website visualizing more properties, such as the value function,
can be generated.
"""

import logging
import torch
import numpy as np
import sys
import pygame
import gymnasium as gym
import memory_gym

from docopt import docopt
from gymnasium import spaces

from neroRL.utils.yaml_parser import YamlParser
from neroRL.utils.utils import get_environment_specs
from neroRL.environments.wrapper import wrap_environment
from neroRL.utils.video_recorder import VideoRecorder
from neroRL.nn.actor_critic import create_actor_critic_model

import cv2
import pickle

# Setup logger
logging.basicConfig(level = logging.INFO, handlers=[])
logger = logging.getLogger("enjoy")
console = logging.StreamHandler()
console.setFormatter(logging.Formatter("%(asctime)s: %(message)s", "%Y-%m-%d %H:%M:%S"))
logger.addHandler(console)

def init_recurrent_cell(recurrence_config, model, device):
    hxs, cxs = model.init_recurrent_cell_states(1, device)
    if recurrence_config["layer_type"] == "gru":
        recurrent_cell = hxs
    elif recurrence_config["layer_type"] == "lstm":
        recurrent_cell = (hxs, cxs)
    return recurrent_cell

def init_transformer_memory(trxl_conf, model):
    memory_mask = torch.tril(torch.ones((trxl_conf["memory_length"], trxl_conf["memory_length"])), diagonal=-1)
    memory = model.init_transformer_memory(1, trxl_conf["max_episode_steps"], trxl_conf["num_blocks"], trxl_conf["embed_dim"])
    # Setup memory window indices
    repetitions = torch.repeat_interleave(torch.arange(0, trxl_conf["memory_length"]).unsqueeze(0), trxl_conf["memory_length"] - 1, dim = 0).long()
    memory_indices = torch.stack([torch.arange(i, i + trxl_conf["memory_length"]) for i in range(trxl_conf["max_episode_steps"] - trxl_conf["memory_length"] + 1)]).long()
    memory_indices = torch.cat((repetitions, memory_indices))
    return memory, memory_mask, memory_indices

def main():
    # Docopt command line arguments
    _USAGE = """
    Usage:
            nenjoy [options]
            nenjoy --help

    Options:
        --config=<path>            Path to the config file [default: ].
        --checkpoint=<path>        Path to the checkpoint file [default: ].
        --untrained                Whether an untrained model should be used [default: False].
        --worker-id=<n>            Sets the port for each environment instance [default: 2].
        --seed=<n>                 The to be played seed of an episode [default: 0].
        --video=<path>             Specify a path for saving a video, if video recording is desired. The file's extension will be set automatically. [default: ./video].
        --framerate=<n>            Specifies the frame rate of the to be rendered video. [default: 6]
        --play                     Play a the checkpoint environment by yourself to collect the data. [default: False]
    """
    options = docopt(_USAGE)
    untrained = options["--untrained"]                              # defaults to False
    config_path = options["--config"]                               # defaults to an empty string
    checkpoint_path = options["--checkpoint"]                       # defaults to an empty string
    worker_id = int(options["--worker-id"])                         # defaults to 2
    seed = int(options["--seed"])                                   # defaults to 0
    video_path = options["--video"]                                 # defaults to "video"
    frame_rate = options["--framerate"]                             # defaults to 6
    play_env = options["--play"]                                        # defaults to False
    

    # Determine cuda availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
    else:
        torch.set_default_tensor_type("torch.FloatTensor")

    # Load config, environment, model, evaluation and training parameters
    if not config_path and not checkpoint_path:
        raise ValueError("Either a config or a checkpoint must be provided")
    checkpoint = torch.load(checkpoint_path, map_location=device) if checkpoint_path else None
    configs = YamlParser(config_path).get_config() if config_path else checkpoint["configs"]
    model_config = checkpoint["configs"]["model"] if checkpoint else configs["model"]

    if play_env:
        play(configs["environment"], seed)
        
    # Launch environment
    logger.info("Step 1: Launching environment")
    configs["environment"]["reset_params"]["start-seed"] = seed
    configs["environment"]["reset_params"]["num-seeds"] = 1
    configs["environment"]["reset_params"]["seed"] = seed
    visual_observation_space, vector_observation_space, action_space_shape, max_episode_steps = get_environment_specs(configs["environment"], worker_id + 1, True)
    env = wrap_environment(configs["environment"], worker_id, realtime_mode = True, record_trajectory = True)

    # Build or load model
    logger.info("Step 2: Creating model")
    share_parameters = False
    if configs["trainer"]["algorithm"] == "PPO":
        share_parameters = configs["trainer"]["share_parameters"]
    if "transformer" in model_config:
        model_config["transformer"]["max_episode_steps"] = max_episode_steps
    use_obs_reconstruction = configs["trainer"]["obs_reconstruction_schedule"]["initial"] > 0.0
    model = create_actor_critic_model(model_config, share_parameters, visual_observation_space,
                            vector_observation_space, action_space_shape, device, use_obs_reconstruction)
    if "DAAC" in configs["trainer"]:
        model.add_gae_estimator_head(action_space_shape, device)
    if not untrained:
        if not checkpoint:
            # If a checkpoint is not provided as an argument, it shall be retrieved from the config
            logger.info("Step 2: Loading model from " + model_config["model_path"])
            checkpoint = torch.load(model_config["model_path"], map_location=device)
        model.load_state_dict(checkpoint["model"])
        if "recurrence" in model_config:
            model.set_mean_recurrent_cell_states(checkpoint["hxs"], checkpoint["cxs"])
    model.eval()

    # Run all one episode
    env_data =load_env_data("result.pkl")
    # Reset environment
    t = 0
    logger.info("Step 3: Resetting the environment")
    logger.info("Step 3: Using seed " + str(seed))
    vis_obs, vec_obs = env_data["vis_obs"][t], env_data["vec_obs"][t]
    done = False
    
    # Init memory if applicable
    memory, memory_mask, memory_indices, mask, indices = None, None, None, None, None
    # Init hidden state (None if not available)
    if "recurrence" in model_config:
        memory = init_recurrent_cell(model_config["recurrence"], model, device)
    # Init transformer memory
    if "transformer" in model_config:
        memory, memory_mask, memory_indices = init_transformer_memory(model_config["transformer"], model)
        memory_length = model_config["transformer"]["memory_length"]

    # Play episode
    logger.info("Step 4: Run " + str(1) + " episode(s) in realtime . . .")

    # Store data for video recording
    decoder_obs = []

    # Play one episode
    with torch.no_grad():
        while not done:
            # Forward the neural net
            vis_obs = torch.tensor(np.expand_dims(vis_obs, 0), dtype=torch.float32, device=device) if vis_obs is not None else None
            vec_obs = torch.tensor(np.expand_dims(vec_obs, 0), dtype=torch.float32, device=device) if vec_obs is not None else None
            # Prepare transformer memory
            if "transformer" in model_config:
                in_memory = memory[0, memory_indices[t].unsqueeze(0)]
                t_ = max(0, min(t, memory_length - 1))
                mask = memory_mask[t_].unsqueeze(0)
                indices = memory_indices[t].unsqueeze(0)
            else:
                in_memory = memory

            _, _, new_memory, _ = model(vis_obs, vec_obs, in_memory, mask, indices)
            
            # Set memory if used
            if "recurrence" in model_config:
                memory = new_memory
            if "transformer" in model_config:
                memory[:, t] = new_memory

            # Store data for video recording
            decoder_obs.append(model.reconstruct_observation().squeeze(0).cpu().numpy().transpose(2, 1, 0) * 255.0)

            # Step environment
            t += 1
            vis_obs, vec_obs, _, done, info = env_data["vis_obs"][t], env_data["vec_obs"][t], env_data["reward"][t], env_data["done"][t], env_data["info"][t]

        vis_obs = torch.tensor(np.expand_dims(vis_obs, 0), dtype=torch.float32, device=device) if vis_obs is not None else None
        vec_obs = torch.tensor(np.expand_dims(vec_obs, 0), dtype=torch.float32, device=device) if vec_obs is not None else None
        model(vis_obs, vec_obs, in_memory, mask, indices)
        decoder_obs.append(model.reconstruct_observation().detach().squeeze(0).cpu().numpy().transpose(2, 1, 0) * 255.0)
        
        logger.info("Episode Reward: " + str(info["reward"]))
        logger.info("Episode Length: " + str(info["length"]))

        # Complete video data
        trajectory_data = {}
        trajectory_data["episode_reward"] = info["reward"]
        trajectory_data["seed"] = seed
        trajectory_data["decoder_obs"] = decoder_obs
        trajectory_data["vis_obs"] = env_data["imgs"]
        trajectory_data["rewards"] = env_data["reward"]
        
        # Render and serialize video
        render_video(trajectory_data, video_path, frame_rate)

    env.close()
    
def play(config, seed):
    if config["name"] == "SearingSpotlights-v0":
        play_ss(config, seed)
    
def load_env_data(file_name:str):
    with open(file_name, 'rb') as handle:
        b = pickle.load(handle)
    return b
    
def play_ss(config, seed):

    result = []
    env = wrap_environment(config, realtime_mode = True, record_trajectory = True, worker_id=None)
    vis_obs, vec_obs = env.reset()
    result = {"vis_obs": [vis_obs], "vec_obs": [vec_obs], "reset_info": [None], "actions": [None], "reward": [0], "done": [False], "info": [None], "seed": seed}
    done = False

    while not done:
        actions = [0, 0]
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            actions[1] = 1
        if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            actions[0] = 2
        if keys[pygame.K_DOWN] or keys[pygame.K_s]:
            actions[1] = 2
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            actions[0] = 1

        vis_obs, vec_obs, reward, done, info = env.step(actions)
        result["vis_obs"].append(vis_obs)
        result["vec_obs"].append(vec_obs)
        result["reward"].append(reward)
        result["done"].append(done)
        result["info"].append(info)

        # Process event-loop
        for event in pygame.event.get():
        # Quit
            if event.type == pygame.QUIT:
                done = True

    print("episode reward: " + str(info["reward"]))
    print("episode length: " + str(info["length"]))
    print("agent health: " + str(info["agent_health"]))
    print("coins collected: " + str(info["coins_collected"]))
    print("exit success: " + str(bool(info["exit_success"])))

    result["imgs"] = env.get_episode_trajectory["vis_obs"]
    env.close()
    with open('result.pkl', 'wb') as handle:
        pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)
    

def render_video(trajectory_data, video_path, frame_rate):
    """Triggers the process of rendering the trajectory data to a video.
    The rendering is done with the help of OpenCV.
    
    Arguments:
        trajectory_data {dift} -- This dictionary provides all the necessary information to render one episode of an agent behaving in its environment.
    """
    # Init video recorder
    video_recorder = VideoRecorder(video_path, frame_rate)
    # Init VideoWriter, the frame rate is defined by each environment individually
    out = cv2.VideoWriter(video_recorder.video_path + "_seed_" + str(trajectory_data["seed"]) + ".mp4",
                            video_recorder.fourcc, video_recorder.frame_rate, (video_recorder.width * 2, video_recorder.height + video_recorder.info_height))
    # Render each frame of the episode
    for i in range(len(trajectory_data["vis_obs"])):
        # Setup environment frame
        env_frame = trajectory_data["vis_obs"][i][...,::-1].astype(np.uint8) # Convert RGB to BGR, OpenCV expects BGR
        env_frame = cv2.resize(env_frame, (video_recorder.width, video_recorder.height), interpolation=cv2.INTER_AREA)
        
        decoder_frame = trajectory_data["decoder_obs"][i][...,::-1].astype(np.uint8) # Convert RGB to BGR, OpenCV expects BGR
        decoder_frame = cv2.resize(decoder_frame, (video_recorder.width, video_recorder.height), interpolation=cv2.INTER_AREA)
        info_frame = np.zeros((video_recorder.info_height, video_recorder.width, 3), dtype=np.uint8)
        decoder_frame = np.vstack((info_frame, decoder_frame))

        # Setup info frame
        info_frame = np.zeros((video_recorder.info_height, video_recorder.width, 3), dtype=np.uint8)
        # Seed
        video_recorder.draw_text_overlay(info_frame, 8, 20, trajectory_data["seed"], "seed")
        # Current step
        video_recorder.draw_text_overlay(info_frame, 108, 20, i, "step")
        # Collected rewards so far
        video_recorder.draw_text_overlay(info_frame, 208, 20, round(sum(trajectory_data["rewards"][0:i]), 3), "total reward")

        # Concatenate environment and debug frames
        output_image = np.vstack((info_frame, env_frame))
        # Concatenate decoder frame if available
        output_image = np.hstack((output_image, decoder_frame))

        # Write frame
        out.write(output_image)
    # Finish up the video
    out.release()


if __name__ == "__main__":
    main()
    