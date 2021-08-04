import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from torch.nn import functional as F

from neroRL.trainers.PPO.models.base import ActorCriticBase

class ActorCriticSeperateWeights(ActorCriticBase):
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
        ActorCriticBase.__init__(self, recurrence, config)
        self.actor_encoder, self.actor_recurrent_layer, self.actor_hidden = self.create_base_model(config, vis_obs_space, vec_obs_shape)
        self.critic_encoder, self.critic_recurrent_layer, self.critic_hidden = self.create_base_model(config, vis_obs_space, vec_obs_shape)

        # Decouple policy from value
        # Hidden layer of the policy
        self.lin_policy = nn.Linear(in_features=self.out_hidden_layer, out_features=512)
        nn.init.orthogonal_(self.lin_policy.weight, np.sqrt(2))

        # Hidden layer of the value function
        self.lin_value = nn.Linear(in_features=self.out_hidden_layer, out_features=512)
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
            h_actor, h_critic = self.actor_encoder(vis_obs, device), self.critic_encoder(vis_obs, device)
            if vec_obs is not None:
                vec_obs = torch.tensor(vec_obs, dtype=torch.float32, device=device)    # Convert vec_obs to tensor
                # Add vector observation to the flattened output of the visual encoder if available
                h_actor, h_critic = torch.cat((h_actor, vec_obs), 1), torch.cat((h_critic, vec_obs), 1)
        else:
            h_actor, h_critic = torch.tensor(vec_obs, dtype=torch.float32, device=device), torch.tensor(vec_obs, dtype=torch.float32, device=device) # Convert vec_obs to tensor

        # Forward reccurent layer (GRU or LSTM) if available
        if self.recurrence is not None:
            (actor_recurrent_cell, critic_recurrent_cell) = self.unpack_recurrent_cell(recurrent_cell)

            h_actor, actor_recurrent_cell = self.actor_recurrent_layer(h_actor, actor_recurrent_cell, sequence_length)
            h_critic, critic_recurrent_cell = self.critic_recurrent_layer(h_critic, critic_recurrent_cell, sequence_length)
            

        # Feed hidden layer
        h_actor, h_critic = self.activ_fn(self.actor_hidden(h_actor)), self.activ_fn(self.critic_hidden(h_critic))

        # Decouple policy from value
        # Feed hidden layer (policy)
        h_policy = self.activ_fn(self.lin_policy(h_actor))
        # Feed hidden layer (value function)
        h_value = self.activ_fn(self.lin_value(h_critic))
        # Output: Value Function
        value = self.value(h_value).reshape(-1)
        # Output: Policy Branches
        pi = []
        for i, branch in enumerate(self.policy_branches):
            pi.append(Categorical(logits=self.policy_branches[i](h_policy)))

        recurrent_cell = self.pack_recurrent_cell(actor_recurrent_cell, critic_recurrent_cell, device)

        return pi, value, recurrent_cell

    def init_recurrent_cell_states(self, num_sequences, device):
        actor_recurrent_cell = ActorCriticBase.init_recurrent_cell_states(self, num_sequences, device)
        critic_recurrent_cell = ActorCriticBase.init_recurrent_cell_states(self, num_sequences, device)

        packed_recurrent_cell = self.pack_recurrent_cell(actor_recurrent_cell, critic_recurrent_cell, device)
        recurrent_cell = packed_recurrent_cell if self.recurrence["layer_type"] == "lstm" else (packed_recurrent_cell, None)

        return recurrent_cell 

    def pack_recurrent_cell(self, actor_recurrent_cell, critic_recurrent_cell, device):
        # Unpack recurrent cell, if GRU is used then unpacking might not be possible, so zip the recurrent cell with None and then unpack it
        actor_hxs, actor_cxs = actor_recurrent_cell if isinstance(actor_recurrent_cell, tuple) else (actor_recurrent_cell, None)
        critic_hxs, critic_cxs = critic_recurrent_cell if isinstance(critic_recurrent_cell, tuple) else (critic_recurrent_cell, None)

        hxs = torch.zeros((*actor_hxs.shape, 2), dtype=torch.float32, device=device)
        cxs = torch.zeros((*actor_cxs.shape, 2), dtype=torch.float32, device=device) if actor_cxs is not None else None

        hxs[:, :, :, 0], hxs[:, :, :, 1] = actor_hxs, critic_hxs
        if cxs is not None: # check if LSTM network is used, if it's used then unpack the cell state
            cxs[:, :, :, 0], cxs[:, :, :, 1] = actor_cxs, critic_cxs

        # return the packed recurrent_cell based on the recurrent layer_type
        recurrent_cell = (hxs, cxs) if self.recurrence["layer_type"] == "lstm" else hxs

        return recurrent_cell

    def unpack_recurrent_cell(self, recurrent_cell):
        # Unpack recurrent cell, if GRU is used then unpacking might not be possible, so zip the recurrent cell with None and then unpack it
        (hxs, cxs) = recurrent_cell if isinstance(recurrent_cell, tuple) else (recurrent_cell, None)

        actor_hxs, critic_hxs = hxs[:, :, :, 0], hxs[:, :, :, 1]
        if cxs is not None: # check if LSTM network is used, if it's used then unpack the cell state
            actor_cxs, critic_cxs = cxs[:, :, :, 0], cxs[:, :, :, 1]

        # return the packed recurrent_cell based on the recurrent layer_type
        actor_recurrent_cell = (actor_hxs, actor_cxs) if self.recurrence["layer_type"] == "lstm" else actor_hxs
        critic_recurrent_cell = (critic_hxs, critic_cxs) if self.recurrence["layer_type"] == "lstm" else critic_hxs

        return actor_recurrent_cell, critic_recurrent_cell



class ActorCriticSharedWeights(ActorCriticBase):
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
        ActorCriticBase.__init__(self, recurrence, config)
        self.encoder, self.recurrent_layer, self.hidden_layer = self.create_base_model(config, vis_obs_space, vec_obs_shape)
        

        # Decouple policy from value
        # Hidden layer of the policy
        self.lin_policy = nn.Linear(in_features=self.out_hidden_layer, out_features=512)
        nn.init.orthogonal_(self.lin_policy.weight, np.sqrt(2))

        # Hidden layer of the value function
        self.lin_value = nn.Linear(in_features=self.out_hidden_layer, out_features=512)
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
        h = self.activ_fn(self.hidden_layer(h))

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

