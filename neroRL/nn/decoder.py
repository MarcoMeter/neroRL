import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from neroRL.nn.module import Module

class CNNDecoder(Module):
    """A simple CNN decoder that is the inverse of the Atari CNN encoder."""
    def __init__(self, input_dim, output_shape, use_embedding=False):
        """
        Arguments:
            input_dim {int} -- Input dimension for the embedding layer if used
            output_shape {tuple} -- Shape of the output image (channels, height, width)

        Keyword Arguments:
            use_embedding {bool} -- Whether to use an embedding layer or not (default: {False})
        """
        super(CNNDecoder, self).__init__()
        self.use_embeddding = use_embedding
        # Don't use the embedding if the decoder shall be attached to the output of the cnn encoder
        # Otherwise the embedding is needed to transform the latent space (output of the memory mechanism) to the input space
        if use_embedding:
            self.linear_embedding = nn.Linear(input_dim, 3136)
            nn.init.orthogonal_(self.linear_embedding.weight, np.sqrt(2))
        self.deconv1 = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1)
        nn.init.orthogonal_(self.deconv1.weight, np.sqrt(2))
        self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2)
        nn.init.orthogonal_(self.deconv2.weight, np.sqrt(2))
        self.deconv3 = nn.ConvTranspose2d(32, output_shape[0], kernel_size=8, stride=4)
        nn.init.orthogonal_(self.deconv3.weight, np.sqrt(2))
        self.output_shape = output_shape

    def forward(self, x):
        if self.use_embeddding:
            x = F.relu(self.linear_embedding(x))
        x = x.view(x.size(0), 64, 7, 7)
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = torch.sigmoid(self.deconv3(x))  # alternative torch.tanh(self.deconv3(x))
        # x = (x + 1) / 2                   # scale to [0, 1]
        return x
