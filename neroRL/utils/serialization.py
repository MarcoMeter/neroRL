import torch

def save_checkpoint(path, checkpoint_data):
    """Saves a checkpoint to file

    Arguments:
        path {string} -- Path and filename of the desired destination
        checkpoint_data {dict} -- Checkpoint data like the model's and optimizer's state dict...
    """
    torch.save(checkpoint_data, path)

def load_checkpoint(path):
    """Loads a checkpoint from file

    Arguments:
        path {string} -- Path to the checkpoint file

    Returns:
        {dict} -- Checkpoint that contains the update index, the model's state dict, the optimizer's state dict and the config dict.
    """
    return torch.load(path)
