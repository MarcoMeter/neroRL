import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from torch.nn import functional as F

from neroRL.trainers.PPO.models.encoder import CNNEncoder
from neroRL.trainers.PPO.models.recurrent import GRU, LSTM


class ActorCriticBase(nn.Module):
    def __init__(self, recurrence, config):
        super().__init__()
        self.recurrence = recurrence

        self.mean_hxs = np.zeros(self.recurrence["hidden_state_size"], dtype=np.float32) if recurrence is not None else None
        self.mean_cxs = np.zeros(self.recurrence["hidden_state_size"], dtype=np.float32) if recurrence is not None else None

        self.activ_fn = self.getActivationFunc(config)


    def create_base_model(self, config, vis_obs_space, vec_obs_shape):
        # Observation encoder
        if vis_obs_space is not None:
            encoder = self.get_encoder(config, vis_obs_space)

            # Case: visual observation available
            vis_obs_shape = vis_obs_space.shape
            # Compute output size of convolutional layers
            conv_out_size = self.get_enc_output(encoder, vis_obs_shape)
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
            recurrent_layer = self.get_recurrent_layer(self.recurrence, in_features_next_layer)
            # Hidden layer
            hidden_layer = self.get_hidden_layer(config, self.recurrence["hidden_state_size"], 512)

        # Init Hidden layer
        nn.init.orthogonal_(hidden_layer.weight, np.sqrt(2))

        return encoder, recurrent_layer, hidden_layer



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

    def getActivationFunc(self, config):
        # Set the activation function for most layers of the neural net
        if config["activation"] == "elu":
            return F.elu
        elif config["activation"] == "leaky_relu":
            return F.leaky_relu
        elif config["activation"] == "relu":
            return F.relu
        elif config["activation"] == "elu":
            return F.silu

    def get_encoder(self, config, vis_obs_space):
        if config["encoder"] == "cnn":
            return CNNEncoder(vis_obs_space, config)

    def get_hidden_layer(self, config, in_features, out_features):
        self.out_hidden_layer = out_features
        if config["hidden_layer"] == "default":
            return nn.Linear(in_features=in_features, out_features=self.out_hidden_layer)

    
    def get_recurrent_layer(self, recurrence, input_shape):
        if recurrence["layer_type"] == "gru":
            return GRU(input_shape, recurrence["hidden_state_size"])
        elif recurrence["layer_type"] == "lstm":
            return LSTM(input_shape, recurrence["hidden_state_size"])

    def get_enc_output(self, encoder, shape):
        """Computes the output size of the convolutional layers by feeding a dummy tensor.
        Arguments:
            shape {tuple} -- Input shape of the data feeding the first convolutional layer
        Returns:
            {int} -- Number of output features returned by the utilized convolutional layers
        """
        o = encoder(torch.zeros(1, *shape), "cpu")
        return int(np.prod(o.size()))


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
            h_actor, actor_recurrent_cell = self.actor_recurrent_layer(h_actor, recurrent_cell, sequence_length)
            h_critic, critic_recurrent_cell = self.critic_recurrent_layer(h_critic, recurrent_cell, sequence_length)
            

        # Feed hidden layer
        h_actor, h_critic = self.activ_fn(self.lin_hidden(h_actor)), self.activ_fn(self.lin_hidden(h_critic))

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

        return pi, value, actor_recurrent_cell, critic_recurrent_cell


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

