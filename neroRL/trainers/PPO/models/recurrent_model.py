import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from torch.nn import functional as F


class RecurrentModel(nn.Module):
    def __init__(self, layer_type, input_shape, hidden_state_size):
        super().__init__()
        if layer_type == "gru":
            self.recurrent_layer = nn.GRU(input_shape, hidden_state_size, batch_first=True)
        elif layer_type == "lstm":
            self.recurrent_layer = nn.LSTM(input_shape, hidden_state_size, batch_first=True)
        # Init recurrent layer
        for name, param in self.recurrent_layer.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            elif 'weight' in name:
                nn.init.orthogonal_(param, np.sqrt(2))

    def forward(self, h, recurrent_cell, sequence_length):
        if sequence_length == 1:
                # Case: sampling training data or model optimization using fake recurrence
                h, recurrent_cell = self.recurrent_layer(h.unsqueeze(1), recurrent_cell)
                h = h.squeeze(1) # Remove sequence length dimension
        else:
            # Case: Model optimization
            # Reshape the to be fed data to batch_size, sequence_length, data
            h_shape = tuple(h.size())
            h = h.reshape((h_shape[0] // sequence_length), sequence_length, h_shape[1])

            # Forward recurrent layer
            h, recurrent_cell = self.recurrent_layer(h, recurrent_cell)

            # Reshape to the original tensor size
            h_shape = tuple(h.size())
            h = h.reshape(h_shape[0] * h_shape[1], h_shape[2])
        return h, recurrent_cell

class Base():
    def __init__(self, recurrence):
        self.recurrence = recurrence

        self.mean_hxs = np.zeros(self.recurrence["hidden_state_size"], dtype=np.float32) if recurrence is not None else None
        self.mean_cxs = np.zeros(self.recurrence["hidden_state_size"], dtype=np.float32) if recurrence is not None else None

        # Set the activation function for most layers of the neural net
        self.available_activ_fns = {
            "elu": F.elu,
            "leaky_relu": F.leaky_relu,
            "relu": F.relu,
            "swish": F.silu
        }

    def init_recurrent_cell_states(self, num_sequences, device):
        """Initializes the recurrent cell states (hxs, cxs) based on the configured method and the used recurrent layer type.
        These states can be initialized in 4 ways:

        - zero
        - one
        - mean (based on the recurrent cell states of the sampled training data)
        - sample (based on the mean of all recurrent cell states of the sampled training data, the std is set to 0.01)

        Arugments:
            num_sequences {int}: The number of sequences determines the number of the to be generated initial recurrent cell states.
            device {torch.device}: Target device.

        Returns:
            {tuple}: Depending on the used recurrent layer type, just hidden states (gru) or both hidden states and cell states are returned using initial values.
        """
        hxs, cxs = None, None
        if self.recurrence["hidden_state_init"] == "zero":
            hxs = torch.zeros((num_sequences), self.recurrence["hidden_state_size"], dtype=torch.float32, device=device, requires_grad=True).unsqueeze(0)
            if self.recurrence["layer_type"] == "lstm":
                cxs = torch.zeros((num_sequences), self.recurrence["hidden_state_size"], dtype=torch.float32, device=device, requires_grad=True).unsqueeze(0)
        elif self.recurrence["hidden_state_init"] == "one":
            hxs = torch.ones((num_sequences), self.recurrence["hidden_state_size"], dtype=torch.float32, device=device, requires_grad=True).unsqueeze(0)
            if self.recurrence["layer_type"] == "lstm":
                cxs = torch.ones((num_sequences), self.recurrence["hidden_state_size"], dtype=torch.float32, device=device, requires_grad=True).unsqueeze(0)
        elif self.recurrence["hidden_state_init"] == "mean":
            mean = [self.mean_hxs for i in range(num_sequences)]
            hxs = torch.tensor(mean, device=device, requires_grad=True).unsqueeze(0)
            if self.recurrence["layer_type"] == "lstm":
                mean = [self.mean_cxs for i in range(num_sequences)]
                cxs = torch.tensor(mean, device=device, requires_grad=True).unsqueeze(0)
        elif self.recurrence["hidden_state_init"] == "sample":
            mean = [self.mean_hxs for i in range(num_sequences)]
            hxs = torch.normal(np.mean(mean), 0.01, size=(1, num_sequences, self.recurrence["hidden_state_size"]), requires_grad=True).to(device)
            if self.recurrence["layer_type"] == "lstm":
                mean = [self.mean_cxs for i in range(num_sequences)]
                cxs = torch.normal(np.mean(mean), 0.01, size=(1, num_sequences, self.recurrence["hidden_state_size"]), requires_grad=True).to(device)
        return hxs, cxs

    def set_mean_recurrent_cell_states(self, mean_hxs, mean_cxs):
        """Sets the mean values (hidden state size) for recurrent cell statres.

        Args:
            mean_hxs {np.ndarray}: Mean hidden state
            mean_cxs {np.ndarray}: Mean cell state (in the case of using an LSTM layer)
        """
        self.mean_hxs = mean_hxs
        self.mean_cxs = mean_cxs
