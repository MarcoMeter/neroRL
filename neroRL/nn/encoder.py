import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from neroRL.nn.module import Module

class CNNEncoder(Module):
    """
    A simple three layer CNN which serves as a visual encoder.
    """
    def __init__(self, vis_obs_space, config, activ_fn):
        """Initializes a three layer convolutional neural network.

        Arguments:
            config {dict} -- Model config
            vis_obs_space {box} -- Dimensions of the visual observation space
            activ_fn {activation} -- activation function
        """
        super().__init__()

        # Set the activation function
        self.activ_fn = activ_fn

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

        # Compute the output size of the encoder
        self.conv_enc_size = self.get_enc_output(vis_obs_shape)

    def forward(self, vis_obs):
        """Forward pass of the model

        Arguments:
            vis_obs {numpy.ndarray/torch.tensor} -- Visual observation
            device {torch.device} -- Current device
            
        Returns:
            {torch.tensor} -- Feature tensor
        """
        # Forward observation encoder
        # Propagate input through the visual encoder
        h = self.activ_fn(self.conv1(vis_obs))
        h = self.activ_fn(self.conv2(h))
        h = self.activ_fn(self.conv3(h))
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
        o = self.conv1(torch.zeros(1, *shape))
        o = self.conv2(o)
        o = self.conv3(o)
        return int(np.prod(o.size()))

class ResCNN(Module):
    """
    A simple residual three layer CNN which serves as a visual encoder.
    """
    def __init__(self, vis_obs_space, config, activ_fn, hidden_size=256, channels=[16,32,32]):
        """Initializes a three layer convolutional neural network.

        Arguments:
            config {dict} -- Model config
            vis_obs_space {box} -- Dimensions of the visual observation space
            activ_fn {activation} -- activation function
        """
        super().__init__()

        # Set the activation function
        self.activ_fn = activ_fn

        vis_obs_shape = vis_obs_space.shape

        self.layer1 = self._make_layer(vis_obs_shape[0], channels[0])
        self.layer2 = self._make_layer(channels[0], channels[1])
        self.layer3 = self._make_layer(channels[1], channels[2])

        # Compute the output size of the encoder
        self.conv_enc_size = self.get_enc_output(vis_obs_shape)

        self.fc = nn.Linear(self.conv_enc_size, hidden_size)

        self.apply_init_(self.modules())

    def forward(self, vis_obs):
        """Forward pass of the model

        Arguments:
            vis_obs {numpy.ndarray/torch.tensor} -- Visual observation
            device {torch.device} -- Current device
            
        Returns:
            {torch.tensor} -- Feature tensor
        """
        # Forward observation encoder
        # Propagate input through the visual encoder
        h = self.layer1(vis_obs)
        h = self.layer2(h)
        h = self.layer3(h)
        # Flatten the output of the convolutional layers
        h = h.reshape((-1, self.conv_enc_size))

        h = self.activ_fn(self.fc(h))

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
        layers = []

        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1))
        layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        layers.append(BasicConvBlock(out_channels, activ_fn=self.activ_fn))
        layers.append(BasicConvBlock(out_channels, activ_fn=self.activ_fn))

        return nn.Sequential(*layers)

    def apply_init_(self, modules):
        for m in modules:
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

class BasicConvBlock(nn.Module):
    """
    Residual Network Block
    """
    def __init__(self, n_channels, stride=1, activ_fn=None):
        super(BasicConvBlock, self).__init__()

        self.conv1 = nn.Conv2d(n_channels, n_channels, kernel_size=3, stride=1, padding=(1,1))
        self.activ_fn = activ_fn
        self.conv2 = nn.Conv2d(n_channels, n_channels, kernel_size=3, stride=1, padding=(1,1))
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.activ_fn(x)
        out = self.conv1(out)
        out = self.activ_fn(out)
        out = self.conv2(out)

        out += identity
        return out