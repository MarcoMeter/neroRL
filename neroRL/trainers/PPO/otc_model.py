import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from torch.nn import functional as F

class OTCModel(nn.Module):
    """A flexible actor-critic model that supports:
            - Multi-discrete action spaces
            - visual & vector observation spaces
            - Recurrent polices (either GRU or LSTM)

        Originally, this model has been used for the Obstacle Tower Challenge without a short-term memory.
    """
    def __init__(self, config, vis_obs_space, vec_obs_shape, action_space_shape, recurrence):
        """Model setup

        Arguments:
            config {dict} -- Model config that is not used yet
            vis_obs_space {box} -- Dimensions of the visual observation space (None if not available)
            vec_obs_shape {tuple} -- Dimensions of the vector observation space (None if not available)
            action_space_shape {tuple} -- Dimensions of the action space
            recurrence {dict} -- None if no recurrent policy is used, otherwise contains relevant detais:
                - layer type {stirng}, sequence length {int}, hidden state size {int}, hiddens state initialization {string}, fake recurrence {bool}
        """
        super().__init__()
        self.recurrence = recurrence
        
        self.activ_fn = F.relu

        # Observation encoder
        if vis_obs_space is not None:
            # Case: visual observation available
            vis_obs_shape = vis_obs_space.shape
            # Visual Encoder made of 3 convolutional layers
            self.conv1 = nn.Conv2d(in_channels=vis_obs_shape[0],
                                out_channels=32,
                                kernel_size=8,
                                stride=4,
                                padding=0)
            nn.init.orthogonal_(self.conv1.weight, np.sqrt(2))

            self.conv2 = nn.Conv2d(in_channels=32,
                                out_channels=64,
                                kernel_size=4,
                                stride=2,
                                padding=0)
            nn.init.orthogonal_(self.conv2.weight, np.sqrt(2))

            self.conv3 = nn.Conv2d(in_channels=64,
                                out_channels=64,
                                kernel_size=3,
                                stride=1,
                                padding=0)
            nn.init.orthogonal_(self.conv3.weight, np.sqrt(2))

            # Compute output size of convolutional layers
            self.conv_out_size = self.get_conv_output(vis_obs_shape)
            in_features_next_layer = self.conv_out_size

            # Determine number of features for the next layer's input
            if vec_obs_shape is not None:
                # Case: vector observation is also available
                in_features_next_layer = in_features_next_layer + vec_obs_shape[0]
        else:
            # Case: only vector observation is available
            in_features_next_layer = vec_obs_shape[0]

        # Recurrent Layer (GRU or LSTM)
        if self.recurrence is not None:
            if self.recurrence["type"] == "gru":
                self.recurrent_layer = nn.GRU(in_features_next_layer, self.recurrence["hidden_state_size"])
            elif self.recurrence["type"] == "lstm":
                self.recurrent_layer = nn.LSTM(in_features_next_layer, self.recurrence["hidden_state_size"])
            # Init recurrent layer
            for name, param in self.recurrent_layer.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0)
                elif 'weight' in name:
                    nn.init.orthogonal_(param)
            # Hidden layer
            self.lin_hidden = nn.Linear(in_features=self.recurrence["hidden_state_size"], out_features=512)
        else:
            # Hidden layer
            self.lin_hidden = nn.Linear(in_features=in_features_next_layer, out_features=512)

        # Init Hidden layer
        nn.init.orthogonal_(self.lin_hidden.weight, np.sqrt(2))

        # Decouple policy from value
        # Hidden layer of the policy
        self.lin_policy = nn.Linear(in_features=512, out_features=512)
        nn.init.orthogonal_(self.lin_policy.weight, np.sqrt(2))

        # Hidden layer of the value function
        self.lin_value = nn.Linear(in_features=512, out_features=512)
        nn.init.orthogonal_(self.lin_value.weight, np.sqrt(2))

        # Outputs / Model Heads
        # Policy Branches
        self.policy_branches = nn.ModuleList()
        for num_actions in action_space_shape:
            policy_branch = nn.Linear(in_features=512, out_features=num_actions)
            nn.init.orthogonal_(policy_branch.weight, np.sqrt(0.01))
            self.policy_branches.append(policy_branch)

        # Value Function
        self.value = nn.Linear(in_features=512,
                               out_features=1)
        nn.init.orthogonal_(self.value.weight, 1)

    def forward(self, vis_obs, vec_obs, recurrent_cell, device, sequence_length = 1):
        """Forward pass of the model

        Arguments:
            vis_obs {numpy.ndarray/torch,tensor} -- Visual observation (None if not available)
            vec_obs {numpy.ndarray/torch.tensor} -- Vector observation (None if not available)
            recurrent_cell {torch.tensor} -- Memory cell of the recurrent layer (None if not available)
            device {torch.device} -- Current device
            sequence_length {int} -- Length of the fed sequences

        Returns:
            {list} -- Policy: List featuring categorical distributions respectively for each policy branch
            {torch.tensor} -- Value Function: Value
            {tuple} -- Recurrent cell
        """
        h: torch.Tensor

        if vis_obs is not None:
            vis_obs = torch.tensor(vis_obs, dtype=torch.float32, device=device)      # Convert vis_obs to tensor
            # Propagate input through the visual encoder
            h = self.activ_fn(self.conv1(vis_obs))
            h = self.activ_fn(self.conv2(h))
            h = self.activ_fn(self.conv3(h))
            # Flatten the output of the convolutional layers
            h = h.reshape((-1, self.conv_out_size))
            if vec_obs is not None:
                vec_obs = torch.tensor(vec_obs, dtype=torch.float32, device=device)    # Convert vec_obs to tensor
                # Add vector observation to the flattened output of the visual encoder if available
                h = torch.cat((h, vec_obs), 1)
        else:
            h = torch.tensor(vec_obs, dtype=torch.float32, device=device)        # Convert vec_obs to tensor

        # Forward reccurent layer (GRU or LSTM) if available
        if self.recurrence is not None:
            if sequence_length == 1:
                # Case: sampling training data
                h, recurrent_cell = self.recurrent_layer(h.unsqueeze(0), recurrent_cell)
                h = h.squeeze(0) # Remove sequence length dimension
            else:
                # Case: Model optimization
                # Reshape the to be fed data to sequence_length, batch_size, Data
                h_shape = tuple(h.size())
                h = h.view(sequence_length, (h_shape[0] // sequence_length), h_shape[1])

                # Initialize hidden states to zero
                # recurrent_cell = torch.zeros((h_shape[0] // sequence_length), self.recurrence["hidden_state_size"], dtype=torch.float32, device=device, requires_grad=True)
                # h, recurrent_cell = self.recurrent_layer(h, recurrent_cell.unsqueeze(0))

                # Init hidden states based on the mean of the latest hidden states of the training data sampling phase
                # recurrent_cell = torch.cat((h_shape[0] // sequence_length) * [torch.mean(recurrent_cell, 1)]).unsqueeze(0)

                # Sample initial hidden state based on the mean and std of the previous hidden states of the training data sampling phase
                if self.recurrence["type"] == "gru":
                    mean = torch.mean(recurrent_cell)
                    std = torch.std(recurrent_cell)
                    recurrent_cell = torch.normal(mean, std, size=(1, (h_shape[0] // sequence_length), self.recurrence["hidden_state_size"])).to(device)
                elif self.recurrence["type"] == "lstm":
                    # Hidden state
                    mean = torch.mean(recurrent_cell[0])
                    std = torch.std(recurrent_cell[0])
                    hxs = torch.normal(mean, std, size=(1, (h_shape[0] // sequence_length), self.recurrence["hidden_state_size"])).to(device)
                    # Hidden cell
                    mean = torch.mean(recurrent_cell[1])
                    std = torch.std(recurrent_cell[1])
                    cxs = torch.normal(mean, std, size=(1, (h_shape[0] // sequence_length), self.recurrence["hidden_state_size"])).to(device)
                    recurrent_cell = (hxs, cxs)
                # Forward recurrent layer
                h, recurrent_cell = self.recurrent_layer(h, recurrent_cell)
                # Reshape to the original tensor size
                h_shape = tuple(h.size())
                h = h.view(h_shape[0] * h_shape[1], h_shape[2])

        # Feed hidden layer
        h = self.activ_fn(self.lin_hidden(h))

        # Decouple policy from value
        # Feed hidden layer (policy)
        h_policy = self.activ_fn(self.lin_policy(h))
        # Feed hidden layer (value function)
        h_value = self.activ_fn(self.lin_value(h))
        # Output: Value Function
        value = self.value(h_value).reshape(-1)
        # Output: Policy Branches
        pi = []
        for i, branch in enumerate(self.policy_branches):
            pi.append(Categorical(logits=self.policy_branches[i](h_policy)))

        return pi, value, recurrent_cell

    def get_conv_output(self, shape):
        """Computes the output size of the convolutional layers by feeding a dumy tensor.

        Arguments:
            shape {tuple} -- Input shape of the data feeding the first convolutional layer

        Returns:
            {int} -- Number of output features returned by the utilized convolutional layers
        """
        o = self.conv1(torch.zeros(1, *shape))
        o = self.conv2(o)
        o = self.conv3(o)
        return int(np.prod(o.size()))

class Swish(nn.Module):
    def forward(self, data: torch.Tensor) -> torch.Tensor:
        return torch.mul(data, torch.sigmoid(data))