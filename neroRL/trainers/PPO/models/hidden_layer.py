from torch import nn
import numpy as np

class Hidden_Layer(nn.Module):
    def __init__(self, activ_fn, num_hidden_layers, in_features, out_features):
        super().__init__()
        if num_hidden_layers == 0:
            raise ValueError("The number of hidden layers should be greater than 0") 

        hidden_layer_modules = []

        single_hidden_layer = nn.Linear(in_features=in_features, out_features=out_features)
        nn.init.orthogonal_(single_hidden_layer.weight, np.sqrt(2))
        hidden_layer_modules.append(single_hidden_layer)

        for _ in range(num_hidden_layers):
            hidden_layer_modules.append(activ_fn)

            single_hidden_layer = nn.Linear(in_features=out_features, out_features=out_features)
            nn.init.orthogonal_(single_hidden_layer.weight, np.sqrt(2))
            hidden_layer_modules.append(single_hidden_layer)

        self.hidden_layer = nn.Sequential(*hidden_layer_modules)

    def forward(self, input):
        return self.hidden_layer(input)