import numpy as np
import torch
from torch import nn

from neroRL.nn.encoder import CNNEncoder, ResCNN, SmallImpalaCNN, LinVecEncoder
from neroRL.nn.recurrent import GRU, LSTM
from neroRL.nn.transformer import Transformer
from neroRL.nn.body import HiddenLayer
from neroRL.nn.module import Module, Sequential

class ActorCriticBase(Module):
    """An actor-critic base model which defines the basic components and functionality of the final model:
            - Components: Visual encoder, vector encoder, recurrent layer, transformer, body, heads (value, policy, gae)
            - Functionality: Initialization of the recurrent cells and basic model
    """
    def __init__(self, config):
        """Model setup

        Arguments:
            config {dict} -- Model config
        """
        super().__init__()

        # Whether the model uses shared parameters (i.e. weights) or not
        self.share_parameters = False

        # Members for using a recurrent policy
        # The mean hxs and cxs can be used to sample a recurrent cell state upon initialization
        self.recurrence_config = config["recurrence"] if "recurrence" in config else None
        self.mean_hxs = np.zeros(self.recurrence_config["hidden_state_size"], dtype=np.float32) if self.recurrence_config is not None else None
        self.mean_cxs = np.zeros(self.recurrence_config["hidden_state_size"], dtype=np.float32) if self.recurrence_config is not None else None

        # Members for using a transformer-based policy
        self.transformer_config = config["transformer"] if "transformer" in config else None

        # Set activation function
        self.activ_fn = self.get_activation_function(config)

    def create_base_model(self, config, vis_obs_space, vec_obs_shape):
        """
        Creates and returns the components of a base model, which consists of:
            - a visual encoder,
            - a vector encoder
            - a recurrent layer (optional)
            - a transformer architecture (optinal)
            - and a body
        specified by the model config.

        Arguments:
            config {dict} -- Model config
            vis_obs_space {box} -- Dimensions of the visual observation space
            vec_obs_shape {tuple} -- Dimensions of the vector observation space (None if not available)
        
        Returns:
            {tuple} -- visual encoder, vector encoder, recurrent layer, transformer, body
        """
        vis_encoder, vec_encoder, recurrent_layer, transformer, body = None, None, None, None, None

        # Observation encoder
        if vis_obs_space is not None:
            vis_encoder = self.create_vis_encoder(config, vis_obs_space)

            # Case: visual observation available
            vis_obs_shape = vis_obs_space.shape
            # Compute output size of the encoder
            conv_out_size = self.get_vis_enc_output(vis_encoder, vis_obs_shape)
            in_features_next_layer = conv_out_size

            # Determine number of features for the next layer's input
            if vec_obs_shape is not None:
                # Case: vector observation is also available
                out_features = config["num_vec_encoder_units"] if config["vec_encoder"] != "none" else vec_obs_shape[0]
                vec_encoder = self.create_vec_encoder(config, vec_obs_shape[0], out_features)
                in_features_next_layer = in_features_next_layer + out_features
        else:
            # Case: only vector observation is available
            # Vector observation encoder
            out_features = config["num_vec_encoder_units"] if config["vec_encoder"] != "none" else vec_obs_shape[0]
            vec_encoder = self.create_vec_encoder(config, vec_obs_shape[0], out_features)
            in_features_next_layer = out_features

        # Recurrent layer (GRU or LSTM)
        if self.recurrence_config is not None:
            out_features = self.recurrence_config["hidden_state_size"]
            recurrent_layer = self.create_recurrent_layer(self.recurrence_config, in_features_next_layer, out_features)
            in_features_next_layer = out_features

        # Transformer
        if self.transformer_config is not None:
            out_features = self.transformer_config["embed_dim"]
            transformer = self.create_transformer_layer(self.transformer_config, in_features_next_layer)
            in_features_next_layer = out_features

        # Network body
        out_features = config["num_hidden_units"]
        body = self.create_body(config, in_features_next_layer, out_features)

        return vis_encoder, vec_encoder, recurrent_layer, transformer, body

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
            {tuple} -- Depending on the used recurrent layer type, just hidden states (gru) or both hidden states and cell states are returned using initial values.
        """
        hxs, cxs = None, None
        if self.recurrence_config["hidden_state_init"] == "zero":
            hxs = torch.zeros((num_sequences), self.recurrence_config["num_layers"], self.recurrence_config["hidden_state_size"])
            if self.recurrence_config["layer_type"] == "lstm":
                cxs = torch.zeros((num_sequences), self.recurrence_config["num_layers"], self.recurrence_config["hidden_state_size"])
        elif self.recurrence_config["hidden_state_init"] == "one":
            hxs = torch.ones((num_sequences), self.recurrence_config["num_layers"], self.recurrence_config["hidden_state_size"])
            if self.recurrence_config["layer_type"] == "lstm":
                cxs = torch.ones((num_sequences), self.recurrence_config["num_layers"], self.recurrence_config["hidden_state_size"])
        elif self.recurrence_config["hidden_state_init"] == "mean":
            mean = [self.mean_hxs for i in range(num_sequences)]
            hxs = torch.tensor(mean).unsqueeze(0)
            if self.recurrence_config["layer_type"] == "lstm":
                mean = [self.mean_cxs for i in range(num_sequences)]
                cxs = torch.tensor(mean).unsqueeze(0)
        elif self.recurrence_config["hidden_state_init"] == "sample":
            mean = [self.mean_hxs for i in range(num_sequences)]
            hxs = torch.normal(np.mean(mean), 0.01, size=(1, num_sequences, self.recurrence_config["num_layers"], self.recurrence_config["hidden_state_size"]))
            if self.recurrence_config["layer_type"] == "lstm":
                mean = [self.mean_cxs for i in range(num_sequences)]
                cxs = torch.normal(np.mean(mean), 0.01, size=(1, num_sequences, self.recurrence_config["num_layers"], self.recurrence_config["hidden_state_size"]))
        return hxs, cxs

    def init_transformer_memory(self, num_sequences, episode_length, num_layers, layer_size):
        """Initializes the transformer-based episodic memory as zeros.

        Arguments:
            num_sequences {int} -- Number of batches / sequences
            episode_length {int} -- Max number of steps in an episode
            num_layers {int} -- Number of transformer blocks
            layer_size {int} -- Dimension of the transformber layers

        Returns:
            {torch.tensor} -- Transformer-based episodic memory as zeros
        """
        return torch.zeros((num_sequences, episode_length, num_layers, layer_size), dtype=torch.float32)

    def set_mean_recurrent_cell_states(self, mean_hxs, mean_cxs):
        """Sets the mean values (hidden state size) for recurrent cell states.

        Arguments:
            mean_hxs {np.ndarray} -- Mean hidden state
            mean_cxs {np.ndarray} -- Mean cell state (in the case of using an LSTM layer)
        """
        self.mean_hxs = mean_hxs
        self.mean_cxs = mean_cxs

    def get_activation_function(self, config):
        """Returns the chosen activation function based on the model config.

        Arguments:
            config {dict} -- Model config

        Returns:
            {torch.nn.modules.activation} -- Activation function
        """
        if config["activation"] == "elu":
            return nn.ELU()
        elif config["activation"] == "leaky_relu":
            return nn.LeakyReLU()
        elif config["activation"] == "relu":
            return nn.ReLU()
        elif config["activation"] == "swish":
            return nn.SiLU()
        elif config["activation"] == "gelu":
            return nn.GELU()

    def create_vis_encoder(self, config, vis_obs_space):
        """Creates and returns a new instance of the visual encoder based on the model config.

        Arguments:
            config {dict} -- Model config
            vis_obs_space {box} -- Dimensions of the visual observation space

        Returns:
            {Module} -- The created visual encoder
        """
        if config["vis_encoder"] == "cnn":
            return CNNEncoder(vis_obs_space, config, self.activ_fn)
        elif config["vis_encoder"] == "rescnn":
            return ResCNN(vis_obs_space, config, self.activ_fn)
        elif config["vis_encoder"] == "smallimpala":
            return SmallImpalaCNN(vis_obs_space, config, self.activ_fn)

    def create_vec_encoder(self, config, in_features, out_features):
        """Creates and returns a new instance of the vector encoder based on the model config.

        Arguments:
            config {dict} -- Model config
            in_features {int} -- Size of input
            out_features {int} -- Size of output

        Returns:
            {Module} -- The created vector encoder
        """
        if config["vec_encoder"] == "linear":
            return LinVecEncoder(in_features, out_features, self.activ_fn)
        elif config["vec_encoder"] == "none":
            return Sequential()

    def create_body(self, config, in_features, out_features):
        """Creates and returns a new instance of the model body based on the model config.

        Arguments:
            config {dict} -- Model config
            in_features {int} -- Size of input
            out_features {int} -- Size of output

        Returns:
            {Module} -- The created model body
        """
        self.out_features_body = out_features
        if config["hidden_layer"] == "default":
            return HiddenLayer(self.activ_fn, config["num_hidden_layers"], in_features, out_features)
    
    def create_recurrent_layer(self, config, input_shape, hidden_state_size):
        """Creates and returns a new instance of the recurrent layer based on the recurrence config.

        Arguments:
            recurrence {dict} -- Recurrence config
            input_shape {int} -- Size of input
            hidden_state_size {int} -- Size of the hidden state

        Returns:
            {Module} -- The created recurrent layer
        """
        if config["layer_type"] == "gru":
            return GRU(input_shape, hidden_state_size, config["num_layers"], self.activ_fn, config["residual"])
        elif config["layer_type"] == "lstm":
            return LSTM(input_shape, hidden_state_size, config["num_layers"], self.activ_fn, config["residual"])

    def create_transformer_layer(self, config, input_shape):
        """Creates and returns a transformer module based on the provided config.

        Arguments:
            config {dict} -- Transformer config
            input_shape {int} -- Size of the input (i.e. output size of the preceding layer)

        Returns:
            {nn.Module} -- The entire transformer
        """
        return Transformer(config, input_shape, self.activ_fn)

    def get_vis_enc_output(self, vis_encoder, shape):
        """Computes the output size of the visual encoder by feeding a dummy tensor.

        Arguments:
            encoder{torch.nn.Module} -- The to be used encoder
            shape {tuple} -- Input shape of the data feeding to the encoder

        Returns:
            {int} -- Number of output features returned by the utilized convolutional layers
        """
        o = vis_encoder(torch.zeros(1, *shape))
        return int(np.prod(o.size()))