import numpy as np
import os
import torch
import random

from gymnasium import spaces

from neroRL.environments.wrapper import wrap_environment
from neroRL.utils.monitor import Tag

def set_library_seeds(seed:int) -> None:
    """Applies the submitted seed to PyTorch, Numpy and Python

    Arguments:
        int {seed} -- The to be applied seed
    """
    random.seed(seed)
    random.SystemRandom().seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)

def compute_gradient_stats(modules_dict, prefix = ""):
    """Computes the gradient norm and the gradient mean for each parameter of the model and the entire model itself.

    Arguments:
        model_parameters {dict} -- Main modules of the models
        tag {string} -- To distinguish entire models from each other, a tag can be supplied

    Returns:
        {dict}: Returns all results as a dictionary
    """
    results = {}
    all_grads = []

    for module_name, module in modules_dict.items():
        if module is not None:
            grads = []
            for param in module.parameters():
                grads.append(param.grad.view(-1))
            results[module_name + "_norm"] = (Tag.GRADIENT_NORM, module.grad_norm())
            # results[module_name + "_mean"] = (Tag.GRADIENT_MEAN, module.grad_mean())
            all_grads = all_grads + grads
    results[prefix + "_model_norm"] = (Tag.GRADIENT_NORM, torch.linalg.norm(torch.cat(all_grads)).item())
    # results[prefix + "_model_mean"] = (Tag.GRADIENT_MEAN, torch.mean(torch.cat(all_grads)).item())
    return results

def batched_index_select(input, dim, index):
    for ii in range(1, len(input.shape)):
        if ii != dim:
            index = index.unsqueeze(ii)
    expanse = list(input.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.expand(expanse)
    return torch.gather(input, dim, index)

def get_environment_specs(env_config, worker_id):
    """_summary_

    Arguments:
        env_config {_type_} -- _description_
        worker_id {_type_} -- _description_

    Returns:
        _type_ -- _description_
    """
    dummy_env = wrap_environment(env_config, worker_id)
    vis_obs, vec_obs = dummy_env.reset(env_config["reset_params"])
    max_episode_steps = dummy_env.max_episode_steps
    visual_observation_space = dummy_env.visual_observation_space
    vector_observation_space = dummy_env.vector_observation_space
    if isinstance(dummy_env.action_space, spaces.Discrete):
        action_space_shape = (dummy_env.action_space.n,)
    else:
        action_space_shape = tuple(dummy_env.action_space.nvec)
    dummy_env.close()
    return visual_observation_space, vector_observation_space, action_space_shape, max_episode_steps