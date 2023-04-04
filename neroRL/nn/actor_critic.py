import numpy as np
import torch

from neroRL.nn.base import ActorCriticBase
from neroRL.nn.heads import MultiDiscreteActionPolicy, ValueEstimator, AdvantageEstimator

class ActorCriticSeperateWeights(ActorCriticBase):
    """A flexible actor-critic model with separate actor and critic weights that supports:
            - Multi-discrete action spaces
            - Visual & vector observation spaces
            - Recurrent polices (either GRU or LSTM)
    """
    def __init__(self, config, vis_obs_space, vec_obs_shape, action_space_shape):
        """Model setup
        
        Arguments:
            config {dict} -- Model config
            vis_obs_space {box} -- Dimensions of the visual observation space (None if not available)
            vec_obs_shape {tuple} -- Dimensions of the vector observation space (None if not available)
            action_space_shape {tuple} -- Dimensions of the action space
        """
        ActorCriticBase.__init__(self, config)

        # Members for using a recurrent policy
        # The mean hxs and cxs can be used to sample a recurrent cell state upon initialization
        self.mean_hxs = np.zeros((self.recurrence_config["hidden_state_size"], 2), dtype=np.float32) if self.recurrence_config is not None else None
        self.mean_cxs = np.zeros((self.recurrence_config["hidden_state_size"], 2), dtype=np.float32) if self.recurrence_config is not None else None

        # Create the base models
        self.actor_vis_encoder, self.actor_vec_encoder, self.actor_helm_encoder, self.actor_recurrent_layer, self.actor_transformer, self.actor_body = self.create_base_model(config, vis_obs_space, vec_obs_shape)
        self.critic_vis_encoder, self.critic_vec_encoder, self.critic_helm_encoder, self.critic_recurrent_layer, self.critic_transformer, self.critic_body = self.create_base_model(config, vis_obs_space, vec_obs_shape)

        # Policy head/output
        self.actor_policy = MultiDiscreteActionPolicy(in_features = self.out_features_body, action_space_shape = action_space_shape, activ_fn = self.activ_fn)

        # Value function head/output
        self.critic = ValueEstimator(in_features = self.out_features_body, activ_fn = self.activ_fn)

        # Organize all modules inside a dictionary
        # This will be used for collecting gradient statistics inside the trainer
        self.actor_modules = {
            "actor_vis_encoder": self.actor_vis_encoder,
            "actor_vec_encoder": self.actor_vec_encoder,
            "actor_recurrent_layer": self.actor_recurrent_layer,
            "actor_body": self.actor_body,
            "actor_head": self.actor_policy
        }
        if self.actor_transformer is not None:
            for b, block in enumerate(self.actor_transformer.transformer_blocks):
                self.actor_modules["transformer_" + str(b)] = block

        self.critic_modules = {
            "critic_vis_encoder": self.critic_vis_encoder,
            "critic_vec_encoder": self.critic_vec_encoder,
            "critic_recurrent_layer": self.critic_recurrent_layer,
            "critic_body": self.critic_body,
            "critic_head": self.critic
        }
        if self.critic_transformer is not None:
            for b, block in enumerate(self.critic_transformer.transformer_blocks):
                self.critic_modules["transformer_" + str(b)] = block

    def forward(self, vis_obs, vec_obs, memory = None, mask = None, sequence_length = 1, actions = None):
        """Forward pass of the model

        Arguments:
            vis_obs {numpy.ndarray/torch.tensor} -- Visual observation (None if not available)
            vec_obs {numpy.ndarray/torch.tensor} -- Vector observation (None if not available)
            memory {torch.tensor} -- Reucrrent cell state or episodic memory (None if not available)
            mask {torch.tensor} -- Memory mask (None if the model is not transformer-based)
            sequence_length {int} -- Length of the fed sequences
            actions {torch.tensor} -- The agent's actions (None if DAAC is not used)

        Returns:
            {list} -- Policy: List featuring categorical distributions respectively for each policy branch
            {torch.tensor} -- Value function: Value
            {torch.tensor or tuple} -- Current memory representation or recurrent cell state (None if memory is not used)
            {torch.tensor} -- Advantage function: Advantage (None if DAAC is not used)
        """
        # Process memory
        actor_memory, critic_memory = None, None
        # Unpack recurrent cell or episodic memory if applicable
        if self.recurrence_config is not None:
            (actor_memory, critic_memory) = self.unpack_recurrent_cell(memory)
        if self.transformer_config is not None:
            actor_memory = memory[..., 0]
            critic_memory = memory[..., 1]

        # Feed actor model
        policy, actor_memory, gae = self.forward_actor(vis_obs, vec_obs, actor_memory, mask, sequence_length, actions)

        # Feed critic model
        value, critic_memory = self.forward_critic(vis_obs, vec_obs, critic_memory, mask, sequence_length, actions)

        # Pack recurrent cell
        if self.recurrence_config is not None:
            memory = self.pack_recurrent_cell(actor_memory, critic_memory)
        if self.transformer_config is not None:
            memory = torch.stack((actor_memory, critic_memory),  axis=-1)

        return policy, value, memory, gae

    def forward_actor(self, vis_obs, vec_obs, actor_memory, mask = None, sequence_length = 1, actions = None):
        # Forward observation encoder
        if vis_obs is not None:
            h_actor = self.actor_vis_encoder(vis_obs)
            if vec_obs is not None:
                h_vec_actor = self.actor_vec_encoder(vec_obs)
                # Add vector observation to the flattened output of the visual encoder if available
                h_actor = torch.cat((h_actor, h_vec_actor), 1)
        else:
            h_actor = self.actor_vec_encoder(vec_obs)

        # Forward reccurent layer (GRU or LSTM) if available
        if self.recurrence_config is not None:
            h_actor, actor_memory = self.actor_recurrent_layer(h_actor, actor_memory, sequence_length)

        # Forward transformer if available
        if self.transformer_config is not None:
            h_actor, actor_memory = self.actor_transformer(h_actor, actor_memory, mask)

        # Feed network body
        h_actor = self.actor_body(h_actor)

        # Head: GAE
        if hasattr(self, "actor_gae"):
            gae = self.actor_gae(h_actor, actions)
        else:
            gae = None
        # Head: Policy branches
        pi = self.actor_policy(h_actor)

        return pi, actor_memory, gae

    def forward_critic(self, vis_obs, vec_obs, critic_memory, mask = None, sequence_length = 1, actions = None):
        # Forward observation encoder
        if vis_obs is not None:
            h_critic = self.critic_vis_encoder(vis_obs)
            if vec_obs is not None:
                h_vec_critic = self.critic_vec_encoder(vec_obs)
                # Add vector observation to the flattened output of the visual encoder if available
                h_critic = torch.cat((h_critic, h_vec_critic), 1)
        else:
            h_critic = self.critic_vec_encoder(vec_obs)

        # Forward reccurent layer (GRU or LSTM) if available
        if self.recurrence_config is not None:
            h_critic, critic_memory = self.critic_recurrent_layer(h_critic, critic_memory, sequence_length)

        # Forward transformer if available
        if self.transformer_config is not None:
            h_critic, critic_memory = self.critic_transformer(h_critic, critic_memory, mask)

        # Feed network body
        h_critic = self.critic_body(h_critic)

        # Head: Value function
        value = self.critic(h_critic)

        return value, critic_memory

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

        packed_recurrent_cell = self.pack_recurrent_cell(actor_recurrent_cell, critic_recurrent_cell)
        # (hxs, cxs) is expected to be returned. But if we use GRU then pack_recurrent_cell just returns hxs so we need to zip the recurrent cell with None to return (hxs, None)
        recurrent_cell = packed_recurrent_cell if self.recurrence_config["layer_type"] == "lstm" else (packed_recurrent_cell, None)

        return recurrent_cell 

    def pack_recurrent_cell(self, actor_recurrent_cell, critic_recurrent_cell):
        """ 
        This method packs the recurrent cell states in such a way s.t. it's possible to be stored in the to be used buffer.
        The returned recurrent cell has the form (hxs, cxs) if an lstm is used or hxs if gru is used.
        In the last dimension the hidden state/cell state of the actor and critic are concatenated to a single tensor in such a way that it yields to the form:
        hxs = (actor_hxs, critic_hxs)
        cxs = (actor_cxs, critic_cxs)

        Arguments:
            actor_recurrent_cell {tuple} -- Depending on the used recurrent layer type, just hidden states (gru) or both hidden states and cell states
            critic_recurrent_cell {tuple} -- Depending on the used recurrent layer type, just hidden states (gru) or both hidden states and cell states

        Returns:
            {tuple} -- Depending on the used recurrent layer type, just hidden states (gru) or both hidden states and cell states are returned.
        """
        # Unpack recurrent cell
        # If GRU is used then unpacking might not be possible, so zip the recurrent cell with None and then unpack it
        actor_hxs, actor_cxs = actor_recurrent_cell if isinstance(actor_recurrent_cell, tuple) else (actor_recurrent_cell, None)
        critic_hxs, critic_cxs = critic_recurrent_cell if isinstance(critic_recurrent_cell, tuple) else (critic_recurrent_cell, None)

        hxs = torch.stack((actor_hxs, critic_hxs), dim = 3)
        if actor_cxs is not None: # check if LSTM network is used, if it's used then unpack the cell state
            cxs = torch.stack((actor_cxs, critic_cxs), dim = 3)

        # return the packed recurrent_cell based on the recurrent layer_type
        recurrent_cell = (hxs, cxs) if self.recurrence_config["layer_type"] == "lstm" else hxs
        # return ((actor_hxs, critic_hxs), (actor_cxs, critic_cxs))
        return recurrent_cell

    def unpack_recurrent_cell(self, recurrent_cell):
        """ 
        This method unpacks the recurrent cell states back to its original form, so that a recurrent cell has the form (hxs, cxs).
        
        Arguments:
            actor_recurrent_cell {tuple} -- Depending on the used recurrent layer type, just hidden states (gru) or both hidden states and cell states
            critic_recurrent_cell {tuple} -- Depending on the used recurrent layer type, just hidden states (gru) or both hidden states and cell states

        Returns:
            {tuple} -- Depending on the used recurrent layer type, just hidden states (gru) or both hidden states and cell states are returned.
        """
        # Unpack recurrent cell
        # If GRU is used then unpacking might not be possible, so zip the recurrent cell with None and then unpack it
        (hxs, cxs) = recurrent_cell if isinstance(recurrent_cell, tuple) else (recurrent_cell, None)

        actor_hxs, critic_hxs = hxs.unbind(dim = 3)
        if cxs is not None: # check if LSTM network is used, if it's used then unpack the cell state
            actor_cxs, critic_cxs = cxs.unbind(dim = 3)

        # return the packed recurrent_cell based on the recurrent layer_type
        actor_recurrent_cell = (actor_hxs.contiguous(), actor_cxs.contiguous()) if self.recurrence_config["layer_type"] == "lstm" else actor_hxs.contiguous()
        critic_recurrent_cell = (critic_hxs.contiguous(), critic_cxs.contiguous()) if self.recurrence_config["layer_type"] == "lstm" else critic_hxs.contiguous()

        # return (actor_hxs, actor_cxs), (critic_hxs, critic_cxs)
        return actor_recurrent_cell, critic_recurrent_cell

    def init_transformer_memory(self, num_sequences, memory_length, num_layers, layer_size, deivce):
        """Initializes the transformer-based episodic memory as zeros.

        Arguments:
            num_sequences {int} -- Number of batches / sequences
            memory_length {int} -- Sequence / memory length of the transformer
            num_layers {int} -- Number of transformer blocks
            layer_size {int} -- Dimension of the transformber layers
            deivce {torch.device} -- Tensor device

        Returns:
            {torch.tensor} -- Transformer-based episodic memory as zeros for actor and critic
        """
        return torch.zeros((num_sequences, memory_length, num_layers, layer_size, 2), dtype=torch.float32)

    def add_gae_estimator_head(self, action_space_shape, device) -> None:
        """Adds the generalized advantage estimation head to the model

        Arguments:
            action_space_shape {tuple} -- Shape of the action space
        """
        # Generalized Advantage Estimate head
        self.actor_gae = AdvantageEstimator(in_features = self.out_features_body, action_space_shape = action_space_shape)
        self.actor_gae.to(device)
        self.actor_modules["actor_gae"] = self.actor_gae

    def get_actor_params(self):
        """Collects and returns the parameters of the modules that are related to the actor model

        Returns:
           {list} -- List of actor model parameters
        """
        params = []
        for key, value in self.actor_modules.items():
            if value is not None:
                for param in value.parameters():
                    params.append(param)       
        return params
        
    def get_critic_params(self):
        """Collects and returns the parameters of the modules that are related to the critic model

        Returns:
           {list} -- List of critic model parameters
        """
        params = []
        for key, value in self.critic_modules.items():
            if value is not None:
                for param in value.parameters():
                    params.append(param)       
        return params

class ActorCriticSharedWeights(ActorCriticBase):
    """A flexible shared weights actor-critic model that supports:
            - Multi-discrete action spaces
            - Visual & vector observation spaces
            - Recurrent polices (either GRU or LSTM)
    """
    def __init__(self, config, vis_obs_space, vec_obs_shape, action_space_shape):
        """Model setup

        Arguments:
            config {dict} -- Model config
            vis_obs_space {box} -- Dimensions of the visual observation space (None if not available)
            vec_obs_shape {tuple} -- Dimensions of the vector observation space (None if not available)
            action_space_shape {tuple} -- Dimensions of the action space
            recurrence {dict} -- None if no recurrent policy is used, otherwise contains relevant detais:
                - layer type {str}, sequence length {int}, hidden state size {int}, hiddens state initialization {str}, reset hidden state {bool}
        """
        ActorCriticBase.__init__(self, config)

        # Whether the model uses shared parameters (i.e. weights) or not
        self.share_parameters = True

        # Create the base model
        self.vis_encoder, self.vec_encoder, self.helm_encoder, self.recurrent_layer, self.transformer, self.body = self.create_base_model(config, vis_obs_space, vec_obs_shape)

        # Policy head/output
        self.actor_policy = MultiDiscreteActionPolicy(self.out_features_body, action_space_shape, self.activ_fn)

        # Value function head/output
        self.critic = ValueEstimator(self.out_features_body, self.activ_fn)

        # Organize all modules inside a dictionary
        # This will be used for collecting gradient statistics inside the trainer
        self.actor_critic_modules = {
            "vis_encoder": self.vis_encoder,
            "vec_encoder": self.vec_encoder,
            "recurrent_layer": self.recurrent_layer,
            "body": self.body,
            "actor_head": self.actor_policy,
            "critic_head": self.critic
        }

        if self.transformer is not None:
            for b, block in enumerate(self.transformer.transformer_blocks):
                self.actor_critic_modules["transformer_" + str(b)] = block

    def forward(self, vis_obs, vec_obs, h_helm = None, memory = None, mask = None, memory_indices = None, sequence_length = 1):
        """Forward pass of the model

            vis_obs {numpy.ndarray/torch.tensor} -- Visual observation (None if not available)
            vec_obs {numpy.ndarray/torch.tensor} -- Vector observation (None if not available)
            memory {torch.tensor} -- Reucrrent cell state or episodic memory (None if not available)
            mask {torch.tensor} -- Memory mask (None if the model is not transformer-based)
            memory_indices {torch.tesnor} -- Indices to select the positional encoding that mathes the memory window (None of the model is not transformer-based)
            sequence_length {int} -- Length of the fed sequences

        Returns:
            {list} -- Policy: List featuring categorical distributions respectively for each policy branch
            {torch.tensor} -- Value function: Value
            {torch.tensor or tuple} -- Current memory representation or recurrent cell state (None if memory is not used)
        """
        # Forward observation encoder
        if vis_obs is not None:
            h = self.vis_encoder(vis_obs)
            if vec_obs is not None:
                h_vec = self.vec_encoder(vec_obs)
                # Add vector observation to the flattened output of the visual encoder if available
                h = torch.cat((h, h_vec), 1)
        else:
            h = self.vec_encoder(vec_obs)

        # Forward reccurent layer (GRU or LSTM) if available
        if self.recurrence_config is not None:
            h, memory = self.recurrent_layer(h, memory, sequence_length)

        # Forward transformer if available
        if self.transformer is not None:
            h, memory = self.transformer(h, memory, mask, memory_indices)
            
        # Forward HELM encoder
        if self.helm_encoder is not None:
            if h_helm is None:
                # Sampling
                h_helm = self.helm_encoder(vis_obs)
            h = torch.cat((h, h_helm), dim=-1)

        # Feed network body
        h = self.body(h)

        # Head: Value function
        value = self.critic(h)
        # Head: Policy branches
        pi = self.actor_policy(h)

        return pi, value, memory, None, h_helm

def create_actor_critic_model(model_config, share_parameters, visual_observation_space, vector_observation_space, action_space_shape, device):
    """Creates a shared or non-shared weights actor critic model.

    Arguments:
        model_config {dict} -- Model config
        share_parameters {bool} -- Whether a model with shared parameters or none-shared parameters shall be created
        visual_observation_space {box} -- Dimensions of the visual observation space (None if not available)
        vector_observation_space {tuple} -- Dimensions of the vector observation space (None if not available)
        action_space_shape {tuple} -- Dimensions of the action space
        device {torch.device} -- Current device

    Returns:
        {ActorCriticBase} -- The created actor critic model
    """
    if share_parameters: # check if the actor critic model should share its weights
        return ActorCriticSharedWeights(model_config, visual_observation_space, vector_observation_space,
                            action_space_shape).to(device)
    else:
        return ActorCriticSeperateWeights(model_config, visual_observation_space, vector_observation_space,
                            action_space_shape).to(device)