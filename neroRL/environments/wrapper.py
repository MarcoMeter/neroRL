from neroRL.environments.wrappers.frame_skip import FrameSkipEnv
from neroRL.environments.wrappers.stacked_observation import StackedObservationEnv
from neroRL.environments.wrappers.scaled_visual_observation import ScaledVisualObsEnv
from neroRL.environments.wrappers.grayscale_visual_observation import GrayscaleVisualObsEnv
from neroRL.environments.wrappers.positional_encoding import PositionalEncodingEnv
from neroRL.environments.wrappers.pytorch_shape import PyTorchEnv
from neroRL.environments.wrappers.last_action_to_obs import LastActionToObs
from neroRL.environments.wrappers.last_reward_to_obs import LastRewardToObs
from neroRL.environments.wrappers.reward_normalization import RewardNormalizer

def wrap_environment(config, worker_id, realtime_mode = False, record_trajectory = False):
    """This function instantiates an environment and applies wrappers based on the specified config.

    Arguments:
        config {dict} -- The to be applied wrapping configuration
        worker_id {int} -- The worker id that sets off the port for communication with Unity environments
        realtime_mode {bool} -- Whether to render and run the environment in realtime
        record_trajectory {bool} -- Whether to record the trajectory of an entire episode. This can be used for video recording. (default: {False})

    Returns:
        {Env} -- The wrapped environment
    """
    # Instantiate environment
    if config["type"] == "MemoryGym":
        from neroRL.environments.memory_gym_wrapper import MemoryGymWrapper
        env = MemoryGymWrapper(config["name"], config["reset_params"], realtime_mode=realtime_mode, record_trajectory=record_trajectory)
    elif config["type"] == "Unity":
        from neroRL.environments.unity_wrapper import UnityWrapper
        env = UnityWrapper(config["name"], config["reset_params"], worker_id, realtime_mode=realtime_mode, record_trajectory=record_trajectory)
    elif config["type"] == "ObstacleTower":
        from neroRL.environments.obstacle_tower_wrapper import ObstacleTowerWrapper
        env = ObstacleTowerWrapper(config["name"], config["reset_params"], worker_id, realtime_mode=realtime_mode, record_trajectory=record_trajectory)
    elif config["type"] == "Minigrid":
        from neroRL.environments.minigrid_wrapper import MinigridWrapper
        env = MinigridWrapper(config["name"], config["reset_params"], realtime_mode=realtime_mode, record_trajectory=record_trajectory)
    elif config["type"] == "MinigridVec":
        from neroRL.environments.minigrid_vec_wrapper import MinigridVecWrapper
        env = MinigridVecWrapper(config["name"], config["reset_params"], realtime_mode=realtime_mode, record_trajectory=record_trajectory)
    elif config["type"] == "Procgen":
        from neroRL.environments.procgen_wrapper import ProcgenWrapper
        env = ProcgenWrapper(config["name"], config["reset_params"], realtime_mode=realtime_mode, record_trajectory=record_trajectory)
    elif config["type"] == "CartPole":
        from neroRL.environments.cartpole_wrapper import CartPoleWrapper
        env = CartPoleWrapper(config["name"], config["reset_params"], realtime_mode=realtime_mode, record_trajectory=record_trajectory)
    elif config["type"] == "PoCMemoryEnv":
        from neroRL.environments.poc_memory_env_wrapper import PocMemoryEnvWrapper
        return PocMemoryEnvWrapper(config["reset_params"], realtime_mode=realtime_mode, record_trajectory=record_trajectory)
    elif config["type"] == "PokeRedV2":
        from neroRL.environments.poke_red_wrapper import PokeRedV2Wrapper
        env = PokeRedV2Wrapper(config["name"], config["reset_params"], realtime_mode=realtime_mode, record_trajectory=record_trajectory)
    else:
        raise ValueError(f"Environment type {config['type']} not supported.")
    
    # Wrap environment
    # Frame Skip
    if config["frame_skip"] > 1:
        env = FrameSkipEnv(env, config["frame_skip"])
    # Last action to obs
    if config["last_action_to_obs"]:
        env = LastActionToObs(env)
    # Last reward to obs
    if config["last_reward_to_obs"]:
        env = LastRewardToObs(env)
    # Grayscale
    if config["grayscale"] and env.observation_space is not None:
        env = GrayscaleVisualObsEnv(env)
    # Rescale Visual Observation
    is_vis_obs = "vis_obs" in env.observation_space.spaces
    is_visual_observation = "visual_observation" in env.observation_space.spaces
    if is_vis_obs or is_visual_observation:
        if is_vis_obs:
            shape = env.observation_space.spaces["vis_obs"].shape
        elif is_visual_observation:
            shape = env.observation_space.spaces["visual_observation"].shape
        if shape[0] != config["resize_vis_obs"][0] or shape[1] != config["resize_vis_obs"][1]:
            env = ScaledVisualObsEnv(env, config["resize_vis_obs"][0], config["resize_vis_obs"][1])
    # Stack Observation
    if config["obs_stacks"] > 1:
        env = StackedObservationEnv(env, config["obs_stacks"])
    # Positional Encoding
    if config["positional_encoding"]:
        env = PositionalEncodingEnv(env)
    # Normalize reward
    if config["reward_normalization"] > 1:
        env = RewardNormalizer(env, config["reward_normalization"])
        
    return PyTorchEnv(env)