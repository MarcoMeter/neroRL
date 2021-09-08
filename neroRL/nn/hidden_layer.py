from torch import nn
import numpy as np

from neroRL.nn.module import Module

class HiddenLayer(Module):
    """
    A flexibal hidden layer with a variable number of layers.
    """
    def __init__(self, activ_fn, num_hidden_layers, in_features, out_features):
        """Initializes the hidden layer.
        
        Arguments:
            activ_fn {activation} -- Activation function
            num_hidden_layers {int} -- The number of hidden layers
            in_features {int} -- The number of input features
            out_features {int} -- The number of output features

        Raises:
            ValueError -- Raises a value error if the number of hidden layers is below one.
        """
        super().__init__()
        if num_hidden_layers <= 0: # Raise an error if the number of hidden layers is below one
            raise ValueError("The number of hidden layers should be greater than zero") 

        # Collect the (possible) linear layer(s) of the model in a list
        hidden_layer_modules = []
        for _ in range(num_hidden_layers):
            linear_layer = nn.Linear(in_features=in_features, out_features=out_features)
            nn.init.orthogonal_(linear_layer.weight, np.sqrt(2))
            hidden_layer_modules.append(linear_layer)
            hidden_layer_modules.append(activ_fn) # add activation function
            in_features = out_features

        # Build the hidden layer
        self.hidden_layer = nn.Sequential(*hidden_layer_modules)

    def forward(self, input):
        """Forward pass of the model

        Arguments:
            input {torch.tensor} -- Input feature tensor

        Returns:
            {torch.tensor} -- Output feature tensor
        """
        return self.hidden_layer(input)