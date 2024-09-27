import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from neroRL.nn.activation_map import get_activation
from neroRL.nn.module import Module

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class AtariCNN(Module):
    """
    A simple three layer CNN which serves as a visual encoder. Also known as Atari CNN.
    """
    def __init__(self, vis_obs_space, config):
        """Initializes a three layer convolutional neural network.

        Arguments:
            config {dict} -- Model config
            vis_obs_space {box} -- Dimensions of the visual observation space
        """
        super().__init__()

        self.cnn = nn.Sequential(
            layer_init(nn.Conv2d(vis_obs_space.shape[0], 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute the output size of the encoder
        self.conv_enc_size = self.get_enc_output(vis_obs_space.shape)

    def forward(self, vis_obs):
        """Forward pass of the model

        Arguments:
            vis_obs {numpy.ndarray/torch.tensor} -- Visual observation
            
        Returns:
            {torch.tensor} -- Feature tensor
        """
        return self.cnn(vis_obs)

    def get_enc_output(self, shape):
        """Computes the output size of the encoder by feeding a dummy tensor.

        Arguments:
            shape {tuple} -- Input shape of the data feeding the first encoder
            
        Returns:
            {int} -- Number of output features returned by the utilized encoder
        """
        h = self.cnn(torch.zeros(1, *shape))
        return int(np.prod(h.size()))

class ResCNN(Module):
    """
    A residual convolutional network which serves as a visual encoder.
    Used by the DAAC Algorithm by Raileanu & Fergus, 2021, https://arxiv.org/abs/2102.10330
    """
    def __init__(self, vis_obs_space, config, activ_fn):
        """Initializes a three layer convolutional based neural network.

        Arguments:
            config {dict} -- Model config
            vis_obs_space {box} -- Dimensions of the visual observation space
            activ_fn {activation} -- activation function
        """
        super().__init__()

        # Set the activation function
        self.activ_fn = activ_fn

        vis_obs_shape = vis_obs_space.shape

        self.layer1 = self._make_layer(in_channels=vis_obs_shape[0], out_channels=16)
        self.layer2 = self._make_layer(in_channels=16, out_channels=32)
        self.layer3 = self._make_layer(in_channels=32, out_channels=32)

        # Compute the output size of the encoder
        self.conv_enc_size = self.get_enc_output(vis_obs_shape)

        self.apply_init_(self.modules())

    def forward(self, vis_obs):
        """Forward pass of the model

        Arguments:
            vis_obs {numpy.ndarray/torch.tensor} -- Visual observation
            
        Returns:
            {torch.tensor} -- Feature tensor
        """
        # Forward observation encoder
        # Propagate input through the visual encoder
        h = self.layer1(vis_obs)
        h = self.layer2(h)
        h = self.activ_fn(self.layer3(h))
        # Flatten the output of the convolutional layers
        h = h.reshape((-1, self.conv_enc_size))

        return h

    def get_enc_output(self, shape):
        """Computes the output size of the encoder by feeding a dummy tensor.

        Arguments:
            shape {tuple} -- Input shape of the data feeding the first encoder
            
        Returns:
            {int} -- Number of output features returned by the utilized encoder
        """
        o = self.layer1(torch.zeros(1, *shape))
        o = self.layer2(o)
        o = self.layer3(o)
        return int(np.prod(o.size()))

    
    def _make_layer(self, in_channels, out_channels, stride=1):
        """Creates a neural convolutional based layer like in the DAAC paper.

        Arguments:
            in_channels {int} -- Number of input channels
            out_channels {int} -- Number of output channels
            stride {int} -- Stride of the convolution. Default: 1

        Returns:
            {Module} -- The created layer
        """
        layers = []

        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1))
        layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        layers.append(BasicConvBlock(out_channels, activ_fn=self.activ_fn))
        layers.append(BasicConvBlock(out_channels, activ_fn=self.activ_fn))

        return nn.Sequential(*layers)

    def apply_init_(self, modules):
        """Initializes the weights of a layer like in the DAAC paper.

        Arguments:
            modules {nn.Module}: The torch module
        """
        for m in modules:
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

class SmallImpalaCNN(Module):
    """https://github.com/ml-jku/helm/blob/main/model.py#L42"""
    def __init__(self, vis_obs_space, config, channel_scale=1, hidden_dim=256):
        super(SmallImpalaCNN, self).__init__()
        vis_obs_shape = vis_obs_space.shape
        self.obs_size = vis_obs_shape
        in_channels = self.obs_size[0]
        kernel1 = 8 if self.obs_size[1] > 9 else 4
        kernel2 = 4 if self.obs_size[2] > 9 else 2
        stride1 = 4 if self.obs_size[1] > 9 else 2
        stride2 = 2 if self.obs_size[2] > 9 else 1
        self.block1 = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=16*channel_scale, kernel_size=kernel1, stride=stride1),
                                    nn.ReLU())
        self.block2 = nn.Sequential(nn.Conv2d(in_channels=16*channel_scale, out_channels=32*channel_scale, kernel_size=kernel2, stride=stride2),
                                    nn.ReLU())

        in_features = self._get_feature_size(self.obs_size)
        self.fc = nn.Linear(in_features=in_features, out_features=hidden_dim)

        self.hidden_dim = hidden_dim
        self.apply(xavier_uniform_init)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        x = nn.ReLU()(x)
        return x

    def _get_feature_size(self, shape):
        dummy_input = torch.zeros((shape[0], *shape[1:])).unsqueeze(0)
        x = self.block2(self.block1(dummy_input))
        return np.prod(x.shape[1:])

def xavier_uniform_init(module):
    if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(module.weight.data, gain)
        nn.init.constant_(module.bias.data, 0.)
    return module

class BasicConvBlock(Module):
    """
    Residual Network Block:
    Used by the DAAC Algorithm by Raileanu & Fergus, 2021, https://arxiv.org/abs/2102.10330
    """
    def __init__(self, n_channels, stride=1, activ_fn=None):
        """Initializes a two layer residual convolutional neural network like in the DAAC paper.

        Arguments:
            in_channels {int} -- Number of input channels
            out_channels {int} -- Number of output channels
            stride {int} -- Stride of the convolution. Default: 1
            activ_fn {activation} -- activation function

        Returns:
            {Module} -- The created layer
        """
        super(BasicConvBlock, self).__init__()
        self.activ_fn = activ_fn
        self.conv1 = nn.Conv2d(n_channels, n_channels, kernel_size=3, stride=1, padding=(1,1))
        self.conv2 = nn.Conv2d(n_channels, n_channels, kernel_size=3, stride=1, padding=(1,1))

    def forward(self, h):
        """Forward pass of the model

        Arguments:
            h {torch.tensor} -- Feature tensor
            
        Returns:
            {torch.tensor} -- Feature tensor
        """
        h_identity = h

        h = self.activ_fn(h)
        h = self.conv1(h)
        h = self.activ_fn(h)
        h = self.conv2(h)

        h = h + h_identity
        return h

class LinearEncoder(Module):
    """
    A multi-layer linear vector encoder using nn.Sequential.
    """
    def __init__(self, config, space):
        """Initializes a multi-layer linear neural network.

        Arguments:
            config {dict} -- Modality config
            space {int} -- Input size
        """
        super().__init__()

        activ_fn = get_activation(config["activation"])

        layers = []
        input_dim = space.shape[0]
        
        # Add linear layers followed by activation function based on config["dims"]
        for output_dim in config["dims"]:
            layers.append(layer_init(nn.Linear(input_dim, output_dim)))
            layers.append(activ_fn)
            input_dim = output_dim
        
        # Use nn.Sequential to create the network
        self.network = nn.Sequential(*layers)

    def forward(self, h):
        """Forward pass of the model

        Arguments:
            h {torch.Tensor} -- Feature tensor
            
        Returns:
            {torch.Tensor} -- Feature tensor
        """
        return self.network(h)
