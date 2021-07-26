import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from torch.nn import functional as F

from neroRL.trainers.PPO.models.encoder_model import CNNEncoder
from neroRL.trainers.PPO.models.recurrent_model import RecurrentModel, Base


class OTCModel(nn.Module, Base):
    """A flexible actor-critic model that supports:
            - Multi-discrete action spaces
            - Visual & vector observation spaces
            - Recurrent polices (either GRU or LSTM)

        Originally, this model has been used for the Obstacle Tower Challenge without a recurrent layer.
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

        # Members for using a recurrent policy
        self.recurrence = recurrence

        nn.Module.__init__(self)
        Base.__init__(self, self.recurrence)
        
        self.activ_fn = self.available_activ_fns[config["activation"]]

        # Observation encoder
        if vis_obs_space is not None:
            self.encoder = CNNEncoder(vis_obs_space, config)

            # Case: visual observation available
            vis_obs_shape = vis_obs_space.shape
            # Compute output size of convolutional layers
            conv_out_size = self.get_enc_output(vis_obs_shape)
            in_features_next_layer = conv_out_size

            # Determine number of features for the next layer's input
            if vec_obs_shape is not None:
                # Case: vector observation is also available
                in_features_next_layer = in_features_next_layer + vec_obs_shape[0]

        else:
            # Case: only vector observation is available
            in_features_next_layer = vec_obs_shape[0]

        # Recurrent Layer (GRU or LSTM)
        if self.recurrence is not None:
            self.recurrent_layer = RecurrentModel(self.recurrence["layer_type"], in_features_next_layer, self.recurrence["hidden_state_size"])
            # Hidden layer
            self.lin_hidden = nn.Linear(in_features=self.recurrence["hidden_state_size"], out_features=512)

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

        # Forward observation encoder
        if vis_obs is not None:
            h = self.encoder(vis_obs, device)
            if vec_obs is not None:
                vec_obs = torch.tensor(vec_obs, dtype=torch.float32, device=device)    # Convert vec_obs to tensor
                # Add vector observation to the flattened output of the visual encoder if available
                h = torch.cat((h, vec_obs), 1)
        else:
            h = torch.tensor(vec_obs, dtype=torch.float32, device=device)        # Convert vec_obs to tensor

        # Forward reccurent layer (GRU or LSTM) if available
        if self.recurrence is not None:
            h, recurrent_cell = self.recurrent_layer(h, recurrent_cell, sequence_length)
            

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

    def get_enc_output(self, shape):
        """Computes the output size of the convolutional layers by feeding a dummy tensor.

        Arguments:
            shape {tuple} -- Input shape of the data feeding the first convolutional layer

        Returns:
            {int} -- Number of output features returned by the utilized convolutional layers
        """
        o = self.encoder(torch.zeros(1, *shape), "cpu")
        return int(np.prod(o.size()))
 
    
