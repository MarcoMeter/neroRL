import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from transformers import TransfoXLModel, TransfoXLConfig

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

class LinVecEncoder(Module):
    """
    A simple one linear layer vector encoder.
    """
    def __init__(self, in_features, out_features, activ_fn):
        """Initializes a  one layer linear neural network.

        Arguments:
            in_features {int} -- Size of input
            out_features {int} -- Size of output
            activ_fn {activation} -- activation function
        """
        super().__init__()

        self.lin_layer = nn.Linear(in_features, out_features)
        nn.init.orthogonal_(self.lin_layer.weight, np.sqrt(2))
        self.activ_fn = activ_fn

    def forward(self, h):
        """Forward pass of the model

        Arguments:
            h {torch.tensor} -- Feature tensor
            
        Returns:
            {torch.tensor} -- Feature tensor
        """
        return self.activ_fn(self.lin_layer(h))

class FrozenHopfield(nn.Module):
    def __init__(self, hidden_dim, input_dim, embeddings, beta):
        super(FrozenHopfield, self).__init__()
        self.rand_obs_proj = torch.nn.Parameter(torch.normal(mean=0.0, std=1 / np.sqrt(hidden_dim), size=(hidden_dim, input_dim)), requires_grad=False)
        self.word_embs = embeddings
        self.beta = beta

    def forward(self, observations):
        observations = self._preprocess_obs(observations)
        observations = observations @ self.rand_obs_proj.T
        similarities = observations @ self.word_embs.T / (
                    observations.norm(dim=-1).unsqueeze(1) @ self.word_embs.norm(dim=-1).unsqueeze(0) + 1e-8)
        softm = torch.softmax(self.beta * similarities, dim=-1)
        state = softm @ self.word_embs
        return state

    def _preprocess_obs(self, obs):
        obs = obs.mean(1)
        obs = torch.stack([o.view(-1) for o in obs])
        return obs

class HELMEncoder(nn.Module):
    def __init__(self, input_dim, mem_len=511, beta=100, device='cuda'):
        super(HELMEncoder, self).__init__()
        config = TransfoXLConfig()
        config.mem_len = mem_len
        self.mem_len = config.mem_len

        self.transfo_xl_wt103 = TransfoXLModel.from_pretrained('transfo-xl-wt103', config=config)
        self.transfo_xl_wt103.to(device)
        n_tokens = self.transfo_xl_wt103.word_emb.n_token
        word_embs = self.transfo_xl_wt103.word_emb(torch.arange(n_tokens)).to(device)
        hidden_dim = self.transfo_xl_wt103.d_embed
        self.frozen_hopfield = FrozenHopfield(hidden_dim, input_dim, word_embs, beta=beta)

        for p in self.transfo_xl_wt103.parameters():
            p.requires_grad_(False)

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.out_dim = hidden_dim
        self.memory = None
        self.eval()

    def forward(self, observation):
        vocab_encoding = self.frozen_hopfield.forward(observation)
        out = self.transfo_xl_wt103(inputs_embeds=vocab_encoding.unsqueeze(1), output_hidden_states=True, mems=self.memory)
        self.memory = out.mems
        hidden = out.last_hidden_state[:, -1, :]
        return hidden.detach()