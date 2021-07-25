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