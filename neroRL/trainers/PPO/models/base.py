import numpy as np
import torch
from torch import nn

from neroRL.trainers.PPO.models.encoder import CNNEncoder
from neroRL.trainers.PPO.models.recurrent import GRU, LSTM
from neroRL.trainers.PPO.models.hidden_layer import Hidden_Layer

class ActorCriticBase(nn.Module):
    """An actor-critic base model which defines the basic components and functionality of the final model:
            - Components: encoder, recurrent layer, hidden layer
            - Functionality: Initialization of the recurrent cells and basic model
    """
    def __init__(self, recurrence, config):
        """Model setup

        Args:
            recurrence {dict}: None if no recurrent policy is used, otherwise contains relevant detais:
                - layer type {string}, sequence length {int}, hidden state size {int}, hiddens state initialization {string}, fake recurrence {bool}
            config {dict}: Model config
        """
        super().__init__()
        self.recurrence = recurrence

        self.mean_hxs = np.zeros(self.recurrence["hidden_state_size"], dtype=np.float32) if recurrence is not None else None
        self.mean_cxs = np.zeros(self.recurrence["hidden_state_size"], dtype=np.float32) if recurrence is not None else None

        self.activ_fn = ActorCriticBase.get_activation_function(config)

    def create_base_model(self, config, vis_obs_space, vec_obs_shape):
        """
        Creates and returns the components of a base model, which consists of an encoder, recurrent layer and hidden layer specified by the model config.

        Arguments:
            config {dict}: Model config
            vis_obs_space {box}: Dimensions of the visual observation space
            vec_obs_shape {tuple}: Dimensions of the vector observation space (None if not available)

        Returns:
            tuple: Encoder, Recurrent layer, Hidden layer
        """
        encoder, recurrent_layer, hidden_layer = None, None, None

        # Observation encoder
        if vis_obs_space is not None:
            encoder = self.create_encoder(config, vis_obs_space)

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

        # Recurrent layer (GRU or LSTM)
        if self.recurrence is not None:
            recurrent_layer = self.create_recurrent_layer(self.recurrence, in_features_next_layer)
            in_features_next_layer = self.recurrence["hidden_state_size"]
        
        # Hidden layer
        out_features = config["num_hidden_units"]
        hidden_layer = self.create_hidden_layer(config, in_features_next_layer, out_features)

        return encoder, recurrent_layer, hidden_layer

    def init_recurrent_cell_states(self, num_sequences, device):
        """Initializes the recurrent cell states (hxs, cxs) based on the configured method and the used recurrent layer type.
        These states can be initialized in 4 ways:
        - zero
        - one
        - mean (based on the recurrent cell states of the sampled training data)
        - sample (based on the mean of all recurrent cell states of the sampled training data, the std is set to 0.01)
        Arguments:
            num_sequences {int}: The number of sequences determines the number of the to be generated initial recurrent cell states.
            device {torch.device}: Target device.
        Returns:
            {tuple}: Depending on the used recurrent layer type, just hidden states (gru) or both hidden states and cell states are returned using initial values.
        """
        hxs, cxs = None, None
        if self.recurrence["hidden_state_init"] == "zero":
            hxs = torch.zeros((num_sequences), self.recurrence["hidden_state_size"], dtype=torch.float32, device=device).unsqueeze(0)
            if self.recurrence["layer_type"] == "lstm":
                cxs = torch.zeros((num_sequences), self.recurrence["hidden_state_size"], dtype=torch.float32, device=device).unsqueeze(0)
        elif self.recurrence["hidden_state_init"] == "one":
            hxs = torch.ones((num_sequences), self.recurrence["hidden_state_size"], dtype=torch.float32, device=device).unsqueeze(0)
            if self.recurrence["layer_type"] == "lstm":
                cxs = torch.ones((num_sequences), self.recurrence["hidden_state_size"], dtype=torch.float32, device=device).unsqueeze(0)
        elif self.recurrence["hidden_state_init"] == "mean":
            mean = [self.mean_hxs for i in range(num_sequences)]
            hxs = torch.tensor(mean, device=device).unsqueeze(0)
            if self.recurrence["layer_type"] == "lstm":
                mean = [self.mean_cxs for i in range(num_sequences)]
                cxs = torch.tensor(mean, device=device).unsqueeze(0)
        elif self.recurrence["hidden_state_init"] == "sample":
            mean = [self.mean_hxs for i in range(num_sequences)]
            hxs = torch.normal(np.mean(mean), 0.01, size=(1, num_sequences, self.recurrence["hidden_state_size"])).to(device)
            if self.recurrence["layer_type"] == "lstm":
                mean = [self.mean_cxs for i in range(num_sequences)]
                cxs = torch.normal(np.mean(mean), 0.01, size=(1, num_sequences, self.recurrence["hidden_state_size"])).to(device)
        return hxs, cxs

    def set_mean_recurrent_cell_states(self, mean_hxs, mean_cxs):
        """Sets the mean values (hidden state size) for recurrent cell states.
        Arguments:
            mean_hxs {np.ndarray}: Mean hidden state
            mean_cxs {np.ndarray}: Mean cell state (in the case of using an LSTM layer)
        """
        self.mean_hxs = mean_hxs
        self.mean_cxs = mean_cxs

    @staticmethod
    def get_activation_function(config):
        """Returns the chosen activation function based on the model config.
        Arguments:
            config {dict}: Model config
        Returns:
            {torch.nn.modules.activation}: activation function
        """
        if config["activation"] == "elu":
            return nn.ELU()
        elif config["activation"] == "leaky_relu":
            return nn.LeakyReLU()
        elif config["activation"] == "relu":
            return nn.ReLU()
        elif config["activation"] == "swish":
            return nn.SiLU()

    def create_encoder(self, config, vis_obs_space):
        """Creates and returns a new instance of the encoder based on the model config.
        Arguments:
            config {dict}: Model config
            vis_obs_space {box}: Dimensions of the visual observation space
        Returns:
            {torch.nn.Module}: The created encoder
        """
        if config["encoder"] == "cnn":
            return CNNEncoder(vis_obs_space, config)

    def create_hidden_layer(self, config, in_features, out_features):
        """Creates and returns a new instance of the hidden layer based on the model config.
        Arguments:
            config {dict}: Model config
            in_features {int}: Size of input
            out_features {int}: Size of output
        Returns:
            {torch.nn.Module}: The created hidden layer
        """
        self.out_hidden_layer = out_features
        if config["hidden_layer"] == "default":
            return Hidden_Layer(self.activ_fn, config["num_hidden_layers"], in_features, out_features)
    
    def create_recurrent_layer(self, recurrence, input_shape):
        """Creates and returns a new instance of the recurrent layer based on the recurrence config.
        Arguments:
            recurrence {dict}: Recurrence config
            input_shape {int}: Size of input
        Returns:
            {torch.nn.Module}: The created recurrent layer
        """
        if recurrence["layer_type"] == "gru":
            return GRU(input_shape, recurrence["hidden_state_size"])
        elif recurrence["layer_type"] == "lstm":
            return LSTM(input_shape, recurrence["hidden_state_size"])

    def get_enc_output(self, encoder, shape):
        """Computes the output size of the encoder by feeding a dummy tensor.
        Arguments:
            encoder{torch.nn.Module} -- The to be used encoder
            shape {tuple} -- Input shape of the data feeding to the encoder
            {int} -- Number of output features returned by the utilized convolutional layers
        """
        o = encoder(torch.zeros(1, *shape), "cpu")
        return int(np.prod(o.size()))