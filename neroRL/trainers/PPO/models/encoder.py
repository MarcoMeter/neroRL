import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

class CNNEncoder(nn.Module):
    def __init__(self, vis_obs_space, config, activ_fn):
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

    def forward(self, vis_obs, device):
        """Forward pass of the model

        Arguments:
            vis_obs {numpy.ndarray/torch.tensor} -- Visual observation

        Returns:
            {torch.tensor} -- Feature tensor
        """
        h: torch.Tensor

        # Forward observation encoder
        vis_obs = torch.tensor(vis_obs, dtype=torch.float32, device=device)      # Convert vis_obs to tensor
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