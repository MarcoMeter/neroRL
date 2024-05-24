import numpy as np
from torch import nn

from neroRL.nn.module import Module

class GRU(Module):
    """
    A gated recurrent unit (GRU) module. Before feeding the input to the GRU layer, it is first transformed by a linear layer.
    A residual connection can be added around the GRU layer.
    """
    def __init__(self, input_dim, hidden_dim, num_layers, activ_fn, residual = False, embed = True):
        """
        Initializes the gated recurrent unit.

        Arguments:
            input_dim {int} -- Input size
            hidden_dim {int} -- The number of features in the hidden state
            num_layers {int} -- The number of GRU layers
            activ_fn {torch.nn} -- Activation function
            residual {bool} -- Whether to use a residual connection around the GRU layer
            embed {bool} -- Whether to use a linear layer before feeding the input to the GRU layer
        """
        super().__init__()
        self.activ_fn  = activ_fn
        self.residual = residual
        self.embed = embed

        if embed:
            # Linear input transformation layer
            self.linear_transform = nn.Linear(input_dim, hidden_dim)
            # Init linear layer
            nn.init.orthogonal_(self.linear_transform.weight, np.sqrt(2))
            in_dim = hidden_dim
        else:
            in_dim = input_dim

        # GRU layer
        self.recurrent_layer = nn.GRU(in_dim, hidden_dim, num_layers, batch_first=True)
        # Init GRU
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
        if self.embed:
            # Feed and activate innput transformation layer
            h = self.activ_fn(self.linear_transform(h))
        
        h_identity = h
        
        # (batch_size, num_layers, hidden_size) => (num_layers, batch_size, hidden_size) 
        recurrent_cell = recurrent_cell.swapaxes(0, 1).contiguous()
        if sequence_length == 1:
                # Case: sampling training data (inference)
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
        if self.residual:
            h = h + h_identity

        # Transform the recurrent cell back to its original shape s.t. it can be stored in the buffer
        recurrent_cell = recurrent_cell.swapaxes(0, 1)

        return h, recurrent_cell

class LSTM(Module):
    """
    A long short-term memory (LSTM) module. Before feeding the input to the LSTM layer, it is transformed by a linear layer.
    A residual connection can be added around the LSTM layer.
    """
    def __init__(self, input_dim, hidden_dim, num_layers, activ_fn, residual = False, embed = True):
        """
        Initializes the long short-term memory network.

        Arguments:
            input_dim {int} -- Size of input
            hidden_dim {int} -- The number of features in the hidden state
            num_layers {int} -- The number of GRU layers
            activ_fn {torch.nn.functional} -- Activation function
            residual {bool} -- Whether to use a residual connection
            embed {bool} -- Whether to use a linear layer before feeding the input to the LSTM layer
        """
        super().__init__()
        self.activ_fn  = activ_fn
        self.residual = residual
        self.embed = embed
        
        if embed:
            # Linear input transformation layer
            self.linear_transform = nn.Linear(input_dim, hidden_dim)
            # Init linear layer
            nn.init.orthogonal_(self.linear_transform.weight, np.sqrt(2))
            in_dim = hidden_dim
        else:
            in_dim = input_dim

        # LSTM layer
        self.recurrent_layer = nn.LSTM(in_dim, hidden_dim, num_layers, batch_first=True)
        # Init LSTM layer
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
        if self.embed:
            # Feed and activate input transformation layer
            h = self.activ_fn(self.linear_transform(h))
        
        h_identity = h

        # (batch_size, num_layers, hidden_size) => (num_layers, batch_size, hidden_size) 
        recurrent_cell = (recurrent_cell[0].swapaxes(0, 1).contiguous(), recurrent_cell[1].swapaxes(0, 1).contiguous())
        if sequence_length == 1:
                # Case: sampling training data (inference)
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
        if self.residual:
            h = h + h_identity

        # Transform the recurrent cell back to its original shape s.t. it can be stored in the buffer
        recurrent_cell = (recurrent_cell[0].swapaxes(0, 1), recurrent_cell[1].swapaxes(0, 1))

        return h, recurrent_cell
