import numpy as np
from torch import nn

from neroRL.nn.module import Module

class GRU(Module):
    """
    A single-layer gated recurrent unit (GRU) module.
    """
    def __init__(self, input_shape, hidden_state_size, num_layers):
        """
        Initializes the gated recurrent unit.

        Arguments:
            input_shape {int} -- Input size
            hidden_state_size {int} -- The number of features in the hidden state
        """
        super().__init__()
        self.recurrent_layer = nn.GRU(input_shape, hidden_state_size, num_layers, batch_first=True)
        # Init recurrent layer
        for name, param in self.recurrent_layer.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param, np.sqrt(2))

    def forward(self, h, recurrent_cell, sequence_length):
        """Forward pass of the model

        Arguments:
            h {numpy.ndarray/torch.tensor} -- Feature input tensor
            recurrent_cell {torch.tensor} -- Memory cell of the recurrent layer
            sequence_length {int} -- Length of the fed sequences

        Returns:
            {numpy.ndarray/torch.tensor} -- Feature output tensor
            {torch.tensor} -- Memory cell of the recurrent layer
        """
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

class ResGRU(Module):
    """
    A single-layer residual based gated recurrent unit (GRU) module.
    """
    def __init__(self, input_shape, hidden_state_size, num_layers):
        """
        Initializes the gated recurrent unit.

        Arguments:
            input_shape {int} -- Input size
            hidden_state_size {int} -- The number of features in the hidden state
        """
        super().__init__()
        self.preprocessing_layer = nn.Linear(input_shape, hidden_state_size)
        nn.init.orthogonal_(self.preprocessing_layer.weight, np.sqrt(2))
        self.recurrent_layer = nn.GRU(hidden_state_size, hidden_state_size, num_layers, batch_first=True)
        # Init recurrent layer
        for name, param in self.recurrent_layer.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param, np.sqrt(2))

    def forward(self, h, recurrent_cell, sequence_length):
        """Forward pass of the model

        Arguments:
            h {numpy.ndarray/torch.tensor} -- Feature input tensor
            recurrent_cell {torch.tensor} -- Memory cell of the recurrent layer
            sequence_length {int} -- Length of the fed sequences

        Returns:
            {numpy.ndarray/torch.tensor} -- Feature output tensor
            {torch.tensor} -- Memory cell of the recurrent layer
        """
        h = self.preprocessing_layer(h)
        h_identity = h
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

        # Residual connection
        h = h + h_identity
        return h, recurrent_cell

class LSTM(Module):
    """
    A single-layer long short-term memory (LSTM) module.
    """
    def __init__(self, input_shape, hidden_state_size, num_layers):
        """
        Initializes the long short-term memory network.

        Arguments:
            input_shape {int} -- Size of input
            hidden_state_size {int} -- The number of features in the hidden state
        """
        super().__init__()
        self.recurrent_layer = nn.LSTM(input_shape, hidden_state_size, num_layers, batch_first=True)
        # Init recurrent layer
        for name, param in self.recurrent_layer.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param, np.sqrt(2))

    def forward(self, h, recurrent_cell, sequence_length):
        """Forward pass of the model

        Arguments:
            h {numpy.ndarray/torch.tensor} -- Feature input tensor
            recurrent_cell {torch.tensor} -- Memory cell of the recurrent layer
            sequence_length {int} -- Length of the fed sequences
            
        Returns:
            {numpy.ndarray/torch.tensor} -- Feature output tensor
            {torch.tensor} -- Memory cell of the recurrent layer
        """
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

class ResLSTM(Module):
    """
    A single-layer residual based long short-term memory (LSTM) module.
    """
    def __init__(self, input_shape, hidden_state_size, num_layers):
        """
        Initializes the long short-term memory network.

        Arguments:
            input_shape {int} -- Size of input
        """
        super().__init__()
        self.preprocessing_layer = nn.Linear(input_shape, hidden_state_size)
        nn.init.orthogonal_(self.preprocessing_layer.weight, np.sqrt(2))
        self.recurrent_layer = nn.LSTM(hidden_state_size, hidden_state_size, num_layers, batch_first=True)
        # Init recurrent layer
        for name, param in self.recurrent_layer.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param, np.sqrt(2))

    def forward(self, h, recurrent_cell, sequence_length):
        """Forward pass of the model

        Arguments:
            h {numpy.ndarray/torch.tensor} -- Feature input tensor
            recurrent_cell {torch.tensor} -- Memory cell of the recurrent layer
            sequence_length {int} -- Length of the fed sequences
            
        Returns:
            {numpy.ndarray/torch.tensor} -- Feature output tensor
            {torch.tensor} -- Memory cell of the recurrent layer
        """

        h = self.preprocessing_layer(h)
        h_identity = h
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

        # Residual connection
        h = h + h_identity
        return h, recurrent_cell