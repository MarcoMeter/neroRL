import torch
from neroRL.utils.monitor import Tag

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

def compute_gradient_stats(model_parameters, tag = ""):
    """Computes the gradient norm and the gradient mean for each parameter of the model and the entire model itself.

    Arguments:
        model_parameters {dict} -- Named parameters of the model
        tag {str} -- The tag is added to the key of the norm and the mean for the entire model.
                    This can be usefull to seperate actor from critic parameters (default: "")

    Returns:
        {dict}: Returns all results as a dictionary
    """
    grad_output = {}
    grads = []
    for name, param in model_parameters:
        if param.grad is not None:
            grad = param.grad.data.cpu()
            grads.append(grad.view(-1))
            grad_output["n_" + name] = (Tag.GRADIENT_NORM, torch.linalg.norm(grad).item())
            grad_output["m_" + name] = (Tag.GRADIENT_MEAN, torch.mean(grad).item())
    grad_output["n_" + tag + "_model"] = (Tag.GRADIENT_NORM, torch.linalg.norm(torch.cat(grads)).item())
    grad_output["m_" + tag + "_model"] = (Tag.GRADIENT_MEAN, torch.mean(torch.cat(grads)).item())
    return grad_output
