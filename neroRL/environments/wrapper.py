from neroRL.environments.wrappers.frame_skip import FrameSkipEnv
from neroRL.environments.wrappers.stacked_observation import StackedObservationEnv
from neroRL.environments.wrappers.scaled_visual_observation import ScaledVisualObsEnv
from neroRL.environments.wrappers.grayscale_visual_observation import GrayscaleVisualObsEnv
from neroRL.environments.wrappers.spotlights import SpotlightsEnv
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
    elif config["type"] == "Ballet":
        from neroRL.environments.ballet_wrapper import BalletWrapper
        env = BalletWrapper(config["reset_params"], realtime_mode=realtime_mode, record_trajectory=record_trajectory)
    elif config["type"] == "RandomMaze":
        from neroRL.environments.maze_wrapper import MazeWrapper
        env = MazeWrapper(config["reset_params"], realtime_mode=realtime_mode, record_trajectory=record_trajectory)
        

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
    if config["grayscale"] and env.visual_observation_space is not None:
        env = GrayscaleVisualObsEnv(env)
    # Spotlight perturbation
    if "spotlight_perturbation" in config:
        env = SpotlightsEnv(env, config["spotlight_perturbation"])
    # Rescale Visual Observation
    if env.visual_observation_space is not None:
        env = ScaledVisualObsEnv(env, config["resize_vis_obs"][0], config["resize_vis_obs"][1])
    # Stack Observation
    if config["obs_stacks"] > 1:
        env = StackedObservationEnv(env, config["obs_stacks"])
    if config["reward_normalization"] > 1:
        env = RewardNormalizer(env, config["reward_normalization"])
        
    return PyTorchEnv(env)