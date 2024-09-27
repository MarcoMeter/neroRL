from torch import nn

activation_map = {
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "sigmoid": nn.Sigmoid(),
            "leaky_relu": nn.LeakyReLU(),
            "elu": nn.ELU(),
            "swish": nn.SiLU(),
            "gelu": nn.GELU(),
        }

def get_activation(activation_name):
    """Returns the activation function by name

    Arguments:
        activation_name {str} -- Name of the activation function

    Returns:
        {torch.nn.Module} -- The activation function
    """
    if activation_name not in activation_map:
        raise ValueError("Activation function not found")
    return activation_map[activation_name]