import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from torch.nn import functional as F

class OTCModel(nn.Module):
    """A flexible actor-critic model that supports:
            - Multi-discrete action spaces
            - visual & vector observation spaces
            - Short-term memory (GRU)

        Originally, this model has been used for the Obstacle Tower Challenge without a short-term memory.
    """
    def __init__(self, config, vis_obs_space, vec_obs_shape, action_space_shape, use_recurrent, hidden_state_size):
        """Model setup


        Arguments:
            config {dict} -- Model config that is not used yet
            vis_obs_space {box} -- Dimensions of the visual observation space (None if not available)
            vec_obs_shape {tuple} -- Dimensions of the vector observation space (None if not available)
            action_space_shape {tuple} -- Dimensions of the action space
            use_recurrent {bool} -- Whether to use short-term memory
            hidden_state_size {int} -- Size of the memory's hidden state
        """
        super().__init__()
        self.use_recurrent = use_recurrent
        self.hidden_state_size = hidden_state_size
        
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

        # Recurrent Layer (GRU)
        if use_recurrent:
            self.gru = nn.GRU(in_features_next_layer, hidden_state_size)
            # Init GRU layer
            for name, param in self.gru.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0)
                elif 'weight' in name:
                    nn.init.orthogonal_(param)
            # Hidden layer
            self.lin_hidden = nn.Linear(in_features=hidden_state_size, out_features=512)
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

    def forward(self, vis_obs, vec_obs, hxs, device):
        """Forward pass of the model

        Arguments:
            vis_obs {numpy.ndarray/torch,tensor} -- Visual observation (None if not available)
            vec_obs {numpy.ndarray/torch.tensor} -- Vector observation (None if not available)
            hxs {torch.tensor} -- Hidden state (None if not available)
            device {torch.device} -- Current device

        Returns:
            {list} -- Policy: List featuring categorical distributions respectively for each policy branch
            {torch.tensor} -- Value Function: Value
            {torch.tensor} -- Hidden state
        """
        h: torch.Tensor

        if vis_obs is not None:
            vis_obs = torch.tensor(vis_obs, dtype=torch.float32, device=device)      # Convert vis_obs to tensor
            # Propagate input through the visual encoder
            h = F.relu(self.conv1(vis_obs))
            h = F.relu(self.conv2(h))
            h = F.relu(self.conv3(h))
            # Flatten the output of the convolutional layers
            h = h.reshape((-1, self.conv_out_size))
            if vec_obs is not None:
                vec_obs = torch.tensor(vec_obs, dtype=torch.float32, device=device)    # Convert vec_obs to tensor
                # Add vector observation to the flattened output of the visual encoder if available
                h = torch.cat((h, vec_obs), 1)
        else:
            h = torch.tensor(vec_obs, dtype=torch.float32, device=device)        # Convert vec_obs to tensor

        # Forward reccurent layer (GRU) if available
        if self.use_recurrent:
            h, hxs = self.gru(h.unsqueeze(0), hxs.unsqueeze(0))
            h = h.squeeze(0)
            hxs = hxs.squeeze(0)

        # Feed hidden layer
        h = F.relu(self.lin_hidden(h))

        # Decouple policy from value
        # Feed hidden layer (policy)
        h_policy = F.relu(self.lin_policy(h))
        # Feed hidden layer (value function)
        h_value = F.relu(self.lin_value(h))
        # Output: Value Function
        value = self.value(h_value).reshape(-1)
        # Output: Policy Branches
        pi = []
        for i, branch in enumerate(self.policy_branches):
            pi.append(Categorical(logits=self.policy_branches[i](h_policy)))

        return pi, value, hxs

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
