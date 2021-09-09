import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from neroRL.nn.base import ActorCriticBase

class ActorCriticSeperateWeights(ActorCriticBase):
    """A flexible actor-critic model with separate actor and critic weights that supports:
            - Multi-discrete action spaces
            - Visual & vector observation spaces
            - Recurrent polices (either GRU or LSTM)
    """
    def __init__(self, config, vis_obs_space, vec_obs_shape, action_space_shape, recurrence):
        """Model setup
        
        Arguments:
            config {dict} -- Model config
            vis_obs_space {box} -- Dimensions of the visual observation space (None if not available)
            vec_obs_shape {tuple} -- Dimensions of the vector observation space (None if not available)
            action_space_shape {tuple} -- Dimensions of the action space
            recurrence {dict} -- None if no recurrent policy is used, otherwise contains relevant details:
                - layer type {str}, sequence length {int}, hidden state size {int}, hiddens state initialization {str}, reset hidden state {bool}
        """
        ActorCriticBase.__init__(self, recurrence, config)

        # Lists to group modules to distinguish actor from critic modules
        self.actor_modules = []
        self.critic_modules = []

        # Members for using a recurrent policy
        self.mean_hxs = np.zeros((self.recurrence["hidden_state_size"], 2), dtype=np.float32) if recurrence is not None else None
        self.mean_cxs = np.zeros((self.recurrence["hidden_state_size"], 2), dtype=np.float32) if recurrence is not None else None

        # Create the base model
        self.actor_vis_encoder, self.actor_vec_encoder, self.actor_recurrent_layer, self.actor_body = self.create_base_model(config, vis_obs_space, vec_obs_shape)
        self.critic_vis_encoder, self.critic_vec_encoder, self.critic_recurrent_layer, self.critic_body = self.create_base_model(config, vis_obs_space, vec_obs_shape)

        # Decouple policy from value
        # Hidden layer of the policy
        self.actor_linear = nn.Linear(in_features=self.out_features_body, out_features=512)
        nn.init.orthogonal_(self.actor_linear.weight, np.sqrt(2))

        # Hidden layer of the value function
        self.critic_linear = nn.Linear(in_features=self.out_features_body, out_features=512)
        nn.init.orthogonal_(self.critic_linear.weight, np.sqrt(2))

        # Outputs / Model heads
        # Policy branches
        self.actor_branches = nn.ModuleList()
        for num_actions in action_space_shape:
            actor_branch = nn.Linear(in_features=512, out_features=num_actions)
            self._add_actor_modules(actor_branch)
            nn.init.orthogonal_(actor_branch.weight, np.sqrt(0.01))
            self.actor_branches.append(actor_branch)

        # Value function (i.e. critic)
        self.critic = nn.Linear(in_features=512,
                               out_features=1)
        nn.init.orthogonal_(self.critic.weight, 1)

        # Append the just created modules to their respective list
        self._add_actor_modules([self.actor_vis_encoder, self.actor_vec_encoder, self.actor_recurrent_layer,
                                self.actor_body, self.actor_linear])
        self._add_critic_modules([self.critic_vis_encoder, self.critic_vec_encoder, self.critic_recurrent_layer,
                                self.critic_body, self.critic_linear, self.critic])

    def forward(self, vis_obs, vec_obs, recurrent_cell, device, sequence_length = 1):
        """Forward pass of the model

        Arguments:
            vis_obs {numpy.ndarray/torch.tensor} -- Visual observation (None if not available)
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
            h_actor, h_critic = self.actor_vis_encoder(vis_obs, device), self.critic_vis_encoder(vis_obs, device)
            if vec_obs is not None:
                # Convert vec_obs to tensor and forward vector observation encoder
                vec_obs = torch.tensor(vec_obs, dtype=torch.float32, device=device)
                h_vec_actor, h_vec_critic = self.actor_vec_encoder(vec_obs), self.critic_vec_encoder(vec_obs)
                # Add vector observation to the flattened output of the visual encoder if available
                h_actor, h_critic = torch.cat((h_actor, h_vec_actor), 1), torch.cat((h_critic, h_vec_critic), 1)
        else:
            # Convert vec_obs to tensor and forward vector observation encoder
            h_actor, h_critic = torch.tensor(vec_obs, dtype=torch.float32, device=device), torch.tensor(vec_obs, dtype=torch.float32, device=device)
            h_actor, h_critic = self.actor_vec_encoder(h_actor), self.critic_vec_encoder(h_critic)

        # Forward reccurent layer (GRU or LSTM) if available
        if self.recurrence is not None:
            (actor_recurrent_cell, critic_recurrent_cell) = self._unpack_recurrent_cell(recurrent_cell)

            h_actor, actor_recurrent_cell = self.actor_recurrent_layer(h_actor, actor_recurrent_cell, sequence_length)
            h_critic, critic_recurrent_cell = self.critic_recurrent_layer(h_critic, critic_recurrent_cell, sequence_length)

        # Feed network body
        h_actor, h_critic = self.actor_body(h_actor), self.critic_body(h_critic)

        # Decouple policy from value
        # Feed hidden layer (policy)
        h_policy = self.activ_fn(self.actor_linear(h_actor))
        # Feed hidden layer (value function)
        h_value = self.activ_fn(self.critic_linear(h_critic))
        # Output: Value function
        value = self.critic(h_value).reshape(-1)
        # Output: Policy branches
        pi = []
        for i, branch in enumerate(self.actor_branches):
            pi.append(Categorical(logits=self.actor_branches[i](h_policy)))

        if self.recurrence is not None:
            recurrent_cell = self._pack_recurrent_cell(actor_recurrent_cell, critic_recurrent_cell, device)

        return pi, value, recurrent_cell

    def init_recurrent_cell_states(self, num_sequences, device):
        """Initializes the recurrent cell states (hxs, cxs) based on the configured method and the used recurrent layer type.
        These states can be initialized in 4 ways:
        - zero
        - one
        - mean (based on the recurrent cell states of the sampled training data)
        - sample (based on the mean of all recurrent cell states of the sampled training data, the std is set to 0.01)

        Arguments:
            num_sequences {int} -- The number of sequences determines the number of the to be generated initial recurrent cell states.
            device {torch.device} -- Target device.

        Returns:
            {tuple} -- Depending on the used recurrent layer type, just hidden states (gru) or both hidden and cell states are returned using initial values.
        """
        actor_recurrent_cell = ActorCriticBase.init_recurrent_cell_states(self, num_sequences, device)
        critic_recurrent_cell = ActorCriticBase.init_recurrent_cell_states(self, num_sequences, device)

        packed_recurrent_cell = self._pack_recurrent_cell(actor_recurrent_cell, critic_recurrent_cell, device)
        # (hxs, cxs) is expected to be returned. But if we use GRU then pack_recurrent_cell just returns hxs so we need to zip the recurrent cell with None to return (hxs, None)
        recurrent_cell = packed_recurrent_cell if self.recurrence["layer_type"] == "lstm" else (packed_recurrent_cell, None)

        return recurrent_cell 

    def _pack_recurrent_cell(self, actor_recurrent_cell, critic_recurrent_cell, device):
        """ 
        This method packs the recurrent cell states in such a way s.t. it's possible to be stored in the to be used buffer.
        The returned recurrent cell has the form (hxs, cxs) if an lstm is used or hxs if gru is used.
        In the last dimension the hidden state/cell state of the actor and critic are concatenated to a single tensor in such a way that it yields to the form:
        hxs = (actor_hxs, critic_hxs)
        cxs = (actor_cxs, critic_cxs)

        Arguments:
            actor_recurrent_cell {tuple} -- Depending on the used recurrent layer type, just hidden states (gru) or both hidden states and cell states
            critic_recurrent_cell {tuple} -- Depending on the used recurrent layer type, just hidden states (gru) or both hidden states and cell states
            device {torch.device} -- Target device

        Returns:
            {tuple} -- Depending on the used recurrent layer type, just hidden states (gru) or both hidden states and cell states are returned.
        """
        # Unpack recurrent cell
        # If GRU is used then unpacking might not be possible, so zip the recurrent cell with None and then unpack it
        actor_hxs, actor_cxs = actor_recurrent_cell if isinstance(actor_recurrent_cell, tuple) else (actor_recurrent_cell, None)
        critic_hxs, critic_cxs = critic_recurrent_cell if isinstance(critic_recurrent_cell, tuple) else (critic_recurrent_cell, None)

        hxs = torch.zeros((*actor_hxs.shape, 2), dtype=torch.float32, device=device)
        cxs = torch.zeros((*actor_cxs.shape, 2), dtype=torch.float32, device=device) if actor_cxs is not None else None

        hxs[:, :, :, 0], hxs[:, :, :, 1] = actor_hxs, critic_hxs
        if cxs is not None: # check if LSTM network is used, if it's used then unpack the cell state
            cxs[:, :, :, 0], cxs[:, :, :, 1] = actor_cxs, critic_cxs

        # return the packed recurrent_cell based on the recurrent layer_type
        recurrent_cell = (hxs, cxs) if self.recurrence["layer_type"] == "lstm" else hxs
        # return ((actor_hxs, critic_hxs), (actor_cxs, critic_cxs))
        return recurrent_cell

    def _unpack_recurrent_cell(self, recurrent_cell):
        """ 
        This method unpacks the recurrent cell states back to its original form, so that a recurrent cell has the form (hxs, cxs).
        
        Arguments:
            actor_recurrent_cell {tuple} -- Depending on the used recurrent layer type, just hidden states (gru) or both hidden states and cell states
            critic_recurrent_cell {tuple} -- Depending on the used recurrent layer type, just hidden states (gru) or both hidden states and cell states
            device {torch.device} -- Target device.

        Returns:
            {tuple} -- Depending on the used recurrent layer type, just hidden states (gru) or both hidden states and cell states are returned.
        """
        # Unpack recurrent cell
        # If GRU is used then unpacking might not be possible, so zip the recurrent cell with None and then unpack it
        (hxs, cxs) = recurrent_cell if isinstance(recurrent_cell, tuple) else (recurrent_cell, None)

        actor_hxs, critic_hxs = hxs[:, :, :, 0], hxs[:, :, :, 1]
        if cxs is not None: # check if LSTM network is used, if it's used then unpack the cell state
            actor_cxs, critic_cxs = cxs[:, :, :, 0], cxs[:, :, :, 1]

        # return the packed recurrent_cell based on the recurrent layer_type
        actor_recurrent_cell = (actor_hxs.contiguous(), actor_cxs.contiguous()) if self.recurrence["layer_type"] == "lstm" else actor_hxs.contiguous()
        critic_recurrent_cell = (critic_hxs.contiguous(), critic_cxs.contiguous()) if self.recurrence["layer_type"] == "lstm" else critic_hxs.contiguous()

        # return (actor_hxs, actor_cxs), (critic_hxs, critic_cxs)
        return actor_recurrent_cell, critic_recurrent_cell

    def _add_actor_modules(self, modules):
        if isinstance(modules, nn.Module):
            self.actor_modules.append(modules)
        elif isinstance(modules, list):
            for module in modules:
                if module is not None:
                    self.actor_modules.append(module)

    def _add_critic_modules(self, modules):
        if isinstance(modules, nn.Module):
            self.critic_modules.append(modules)
        elif isinstance(modules, list):
            for module in modules:
                if module is not None:
                    self.critic_modules.append(module)

    def get_actor_params(self):
        params = []
        for module in self.actor_modules:
            try:
                params.append(list(module.parameters())[0])
            except:
                pass
        return params
        
    def get_critic_params(self):
        params = []
        for module in self.critic_modules:
            try:
                params.append(list(module.parameters())[0])
            except:
                pass
        return params

class ActorCriticSharedWeights(ActorCriticBase):
    """A flexible shared weights actor-critic model that supports:
            - Multi-discrete action spaces
            - Visual & vector observation spaces
            - Recurrent polices (either GRU or LSTM)
    """
    def __init__(self, config, vis_obs_space, vec_obs_shape, action_space_shape, recurrence):
        """Model setup

        Arguments:
            config {dict} -- Model config
            vis_obs_space {box} -- Dimensions of the visual observation space (None if not available)
            vec_obs_shape {tuple} -- Dimensions of the vector observation space (None if not available)
            action_space_shape {tuple} -- Dimensions of the action space
            recurrence {dict} -- None if no recurrent policy is used, otherwise contains relevant detais:
                - layer type {str}, sequence length {int}, hidden state size {int}, hiddens state initialization {str}, reset hidden state {bool}
        """
        ActorCriticBase.__init__(self, recurrence, config)

        # Whether the model uses shared parameters (i.e. weights) or not
        self.share_parameters = True

        # Create the base model
        self.vis_encoder, self.vec_encoder, self.recurrent_layer, self.body = self.create_base_model(config, vis_obs_space, vec_obs_shape)

        # Decouple policy from value
        # Hidden layer of the policy
        self.actor_linear = nn.Linear(in_features=self.out_features_body, out_features=512)
        nn.init.orthogonal_(self.actor_linear.weight, np.sqrt(2))

        # Hidden layer of the value function
        self.critic_linear = nn.Linear(in_features=self.out_features_body, out_features=512)
        nn.init.orthogonal_(self.critic_linear.weight, np.sqrt(2))

        # Outputs / Model heads
        # Policy branches
        self.actor_branches = nn.ModuleList()
        for num_actions in action_space_shape:
            actor_branch = nn.Linear(in_features=512, out_features=num_actions)
            nn.init.orthogonal_(actor_branch.weight, np.sqrt(0.01))
            self.actor_branches.append(actor_branch)

        # Value function
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
            h = self.vis_encoder(vis_obs, device)
            if vec_obs is not None:
                vec_obs = torch.tensor(vec_obs, dtype=torch.float32, device=device)    # Convert vec_obs to tensor
                h_vec = self.vec_encoder(vec_obs)
                # Add vector observation to the flattened output of the visual encoder if available
                h = torch.cat((h, h_vec), 1)
        else:
            h = torch.tensor(vec_obs, dtype=torch.float32, device=device)        # Convert vec_obs to tensor
            h = self.vec_encoder(h)

        # Forward reccurent layer (GRU or LSTM) if available
        if self.recurrence is not None:
            h, recurrent_cell = self.recurrent_layer(h, recurrent_cell, sequence_length)
            
        # Feed network body
        h = self.body(h)

        # Decouple policy from value
        # Feed hidden layer (policy)
        h_policy = self.activ_fn(self.actor_linear(h))
        # Feed hidden layer (value function)
        h_value = self.activ_fn(self.critic_linear(h))
        # Output: Value function
        value = self.value(h_value).reshape(-1)
        # Output: Policy branches
        pi = []
        for i, branch in enumerate(self.actor_branches):
            pi.append(Categorical(logits=self.actor_branches[i](h_policy)))

        return pi, value, recurrent_cell

def create_actor_critic_model(model_config, share_parameters, visual_observation_space, vector_observation_space, action_space_shape, recurrence, device):
    """Creates a shared or non-shared weights actor critic model.

    Arguments:
        model_config {dict} -- Model config
        vis_obs_space {box} -- Dimensions of the visual observation space (None if not available)
        vec_obs_shape {tuple} -- Dimensions of the vector observation space (None if not available)
        action_space_shape {tuple} -- Dimensions of the action space
        recurrence {dict} -- None if no recurrent policy is used, otherwise contains relevant details:
                - layer type {str}, sequence length {int}, hidden state size {int}, hiddens state initialization {str}, reset hidden state {bool}
        device {torch.device} -- Current device

    Raises:
        ValueError -- Raises an error if conflicting model parameters are used.

    Returns:
        {nn.Module} -- The created actor critic model
    """
    if share_parameters: # check if the actor critic model should share its weights
        return ActorCriticSharedWeights(model_config, visual_observation_space, vector_observation_space,
                            action_space_shape, recurrence).to(device)
    else:
        return ActorCriticSeperateWeights(model_config, visual_observation_space, vector_observation_space,
                            action_space_shape, recurrence).to(device)