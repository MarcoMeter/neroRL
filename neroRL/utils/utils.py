import torch

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