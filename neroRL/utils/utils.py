import numpy as np
import torch
import random
from neroRL.utils.monitor import Tag

def set_library_seeds(seed:int) -> None:
    """Applies the submitted seed to PyTorch, Numpy and Python

    Arguments:
        int {seed} -- The to be applied seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def masked_mean(tensor:torch.Tensor, mask:torch.Tensor) -> torch.Tensor:
    """
    Returns the mean of the tensor but ignores the values specified by the mask.
    This is used for masking out the padding of the loss functions.

    Args:
        tensor {Tensor} -- The to be masked tensor
        mask {Tensor} -- The mask that is used to mask out padded values of a loss function

    Returns:
        {Tensor} -- Returns the mean of the masked tensor.
    """
    return (tensor.T * mask).sum() / torch.clamp((torch.ones_like(tensor.T) * mask).float().sum(), min=1.0)

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
