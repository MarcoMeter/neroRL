import torch

def save_checkpoint(path, update: int, model_state_dict, optimizer_state_dict, config):
    """Saves a checkpoint to file

    Arguments:
        path {string}: Path and filename of the desired destination
        update {int}: Index of the latest update cycle
        model_state_dict {dict}: State dicitonary of the model
        optimizer_state_dict {dict}: State dictionary of the optimizer
        config {dict}: The utilized config containing all training parameters and properties
    """
    torch.save({
        'update': update,
        'model_state_dict': model_state_dict,
        'optimizer_state_dict': optimizer_state_dict,
        'config': config
    }, path)

def load_checkpoint(path):
    """Loads a checkpoint from file

    Arguments:
        path {string}: Path to the checkpoint file

    Returns:
        {dict} -- Checkpoint that contains the update index, the model's state dict, the optimizer's state dict and the config dict.
    """
    return torch.load(path)
