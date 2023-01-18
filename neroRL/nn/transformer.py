import numpy as np
import torch
from torch import nn
from einops import rearrange
from neroRL.nn.module import Module

class MultiHeadAttention(nn.Module):
    """Multi Head Attention without dropout inspired by https://github.com/aladdinpersson/Machine-Learning-Collection
    https://youtu.be/U0s0f995w14"""
    def __init__(self, embed_dim, num_heads):
        """
        Arguments:
            embed_dim {int} -- Size of the embedding dimension
            num_heads {int} -- Number of attention heads
        """
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_size = embed_dim // num_heads

        assert (
            self.head_size * num_heads == embed_dim
        ), "Embedding dimension needs to be divisible by the number of heads"

        self.values = nn.Linear(self.head_size, self.head_size, bias=False)
        self.keys = nn.Linear(self.head_size, self.head_size, bias=False)
        self.queries = nn.Linear(self.head_size, self.head_size, bias=False)
        self.fc_out = nn.Linear(self.num_heads * self.head_size, embed_dim)

    def forward(self, values, keys, query, mask):
        """
        Arguments:
            values {torch.tensor} -- Value in shape of (N, L, D)
            keys {torch.tensor} -- Keys in shape of (N, L, D)
            query {torch.tensor} -- Queries in shape of (N, L, D)
            mask {torch.tensor} -- Attention mask in shape of (N, L)

        Returns:
            torch.tensor -- Output
            torch.tensor -- Attention weights
        """
        # Get number of training examples and sequence lengths
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # Split the embedding into self.num_heads different pieces
        values = values.reshape(N, value_len, self.num_heads, self.head_size)
        keys = keys.reshape(N, key_len, self.num_heads, self.head_size)
        query = query.reshape(N, query_len, self.num_heads, self.head_size)

        values = self.values(values)  # (N, value_len, heads, head_dim)
        keys = self.keys(keys)  # (N, key_len, heads, head_dim)
        queries = self.queries(query)  # (N, query_len, heads, heads_dim)

        # Einsum does matrix mult. for query*keys for each training example
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        # queries shape: (N, query_len, heads, heads_dim),
        # keys shape: (N, key_len, heads, heads_dim)
        # energy: (N, heads, query_len, key_len)

        # Mask padded indices so their attention weights become 0
        if mask is not None:
            energy = energy.masked_fill(mask.unsqueeze(1).unsqueeze(1) == 0, float("-1e20")) # -inf causes NaN

        # Normalize energy values and apply softmax wo retreive the attention scores
        attention = torch.softmax(energy / (self.embed_dim ** (1 / 2)), dim=3)
        # attention shape: (N, heads, query_len, key_len)

        # Scale values by attention weights
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.num_heads * self.head_size
        )
        # attention shape: (N, heads, query_len, key_len)
        # values shape: (N, value_len, heads, heads_dim)
        # out after matrix multiply: (N, query_len, heads, head_dim), then
        # we reshape and flatten the last two dimensions.

        # Forward projection
        out = self.fc_out(out)
        # Linear layer doesn't modify the shape, final shape will be
        # (N, query_len, embed_dim)

        return out, attention
        
class TransformerBlock(Module):
    def __init__(self, embed_dim, num_heads, layer_norm):
        """Transformer Block made of LayerNorms, Multi Head Attention and one fully connected feed forward projection.

        Arguments:
            embed_dim {int} -- Size of the embeddding dimension
            num_heads {int} -- Number of attention headds
            attention_norm {str} -- Whether to apply LayerNorm "pre" or "post" attention
            projection_norm {str} -- Whether to apply LayerNorm "pre" or "post" the feed forward projection
        """
        super(TransformerBlock, self).__init__()

        # Attention
        self.attention = MultiHeadAttention(embed_dim, num_heads)

        # LayerNorms
        self.layer_norm = layer_norm
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        if layer_norm == "pre":
            self.norm_kv = nn.LayerNorm(embed_dim)
        self.gate1 = GRUGate(embed_dim) # or GRUGatingUnit
        self.gate2 = GRUGate(embed_dim) # or GRUGatingUnit


        # Feed forward projection
        self.fc = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.ReLU())

    def forward(self, value, key, query, mask):
        """
        Arguments:
            values {torch.tensor} -- Value in shape of (N, L, D)
            keys {torch.tensor} -- Keys in shape of (N, L, D)
            query {torch.tensor} -- Queries in shape of (N, L, D)
            mask {torch.tensor} -- Attention mask in shape of (N, L)

        Returns:
            torch.tensor -- Output
            torch.tensor -- Attention weights
        """
        # Apply pre-layer norm across the attention input
        if self.layer_norm == "pre":
            query_ = self.norm1(query)
            value = self.norm_kv(value)
            key = value
        else:
            query_ = query

        # Forward MultiHeadAttention
        attention, attention_weights = self.attention(value, key, query_, mask)

        # Add skip connection and run through normalization
        # h = attention + query
        h = self.gate1((attention, query))
        # Apply post-layer norm across the attention output (i.e. projection input)
        if self.layer_norm == "post":
            h = self.norm1(h)

        # Apply pre-layer norm across the projection input (i.e. attention output)
        if self.layer_norm == "pre":
            h_ = self.norm2(h)
        else:
            h_ = h

        # Forward projection
        forward = self.fc(h_)

        # Add skip connection and run through normalization
        # out = forward + h
        out = self.gate2((forward, h))
        # Apply post-layer norm across the projection output
        if self.layer_norm == "post":
            out = self.norm2(out)
        return out, attention_weights

class SinusoidalPosition(nn.Module):
    """Relative positional encoding"""
    def __init__(self, dim, min_timescale = 2., max_timescale = 1e4):
        super().__init__()
        freqs = torch.arange(0, dim, min_timescale)
        inv_freqs = max_timescale ** (-freqs / dim)
        self.register_buffer('inv_freqs', inv_freqs)

    def forward(self, seq_len):
        seq = torch.arange(seq_len - 1, -1, -1.)
        sinusoidal_inp = rearrange(seq, 'n -> n ()') * rearrange(self.inv_freqs, 'd -> () d')
        pos_emb = torch.cat((sinusoidal_inp.sin(), sinusoidal_inp.cos()), dim = -1)
        return pos_emb

class Transformer(nn.Module):
    """Transformer encoder architecture without dropout. Positional encoding can be either "relative", "learned" or "" (none)."""
    def __init__(self, config, input_dim, activation) -> None:
        """Sets up the input embedding, positional encoding and the transformer blocks.

        Arguments:
            config {dict} -- Transformer config
            input_dim {int} -- Dimension of the input
            activation {torch.nn.modules.activation} -- Activation function of the input embedding
        """
        super().__init__()
        self.config = config
        self.num_blocks = config["num_blocks"]
        self.embed_dim = config["embed_dim"]
        self.num_heads = config["num_heads"]
        self.max_episode_steps = config["max_episode_steps"]
        self.activation = activation

        # Input embedding layer
        self.linear_embedding = nn.Linear(input_dim, self.embed_dim)
        nn.init.orthogonal_(self.linear_embedding.weight, np.sqrt(2))

        # Determine positional encoding
        if config["positional_encoding"] == "relative":
            self.pos_embedding = SinusoidalPosition(dim = self.embed_dim)
        elif config["positional_encoding"] == "learned":
            self.pos_embedding = nn.Parameter(torch.randn(self.max_episode_steps, self.embed_dim)) # (batch size, max episoded steps, num layers, layer size)
        else:
            pass    # No positional encoding is used
        
        # Instantiate transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(self.embed_dim, self.num_heads, config["layer_norm"]) 
            for _ in range(self.num_blocks)])

    def forward(self, h, memories, mask, memory_indices):
        """
        Arguments:
            h {torch.tensor} -- Input (query)
            memories {torch.tesnor} -- Whole episoded memories of shape (N, L, num blocks, D)
            mask {torch.tensor} -- Attention mask (dtype: bool) of shape (N, L)
            memory_indices {torch.tensor} -- Memory window indices (dtype: long) of shape (N, L)

        Returns:
            torch.tensor -- Output of the entire transformer encoder
            torch.tensor -- Out memories (i.e. inputs to the transformer blocks)
        """
        # Feed embedding layer and activate
        h = self.activation(self.linear_embedding(h))

        # Add positional encoding to every transformer block input
        if self.config["positional_encoding"] == "relative":
            pos_embedding = self.pos_embedding(self.max_episode_steps)[memory_indices]
            memories = memories + pos_embedding.unsqueeze(2)
            # memories[:,:,0] = memories[:,:,0] + pos_embedding # add positional encoding only to first layer?
        elif self.config["positional_encoding"] == "learned":
            memories = memories + self.pos_embedding[memory_indices].unsqueeze(2)
            # memories[:,:,0] = memories[:,:,0] + self.pos_embedding[memory_indices] # add positional encoding only to first layer?

        # Forward transformer blocks
        out_memories = []
        for i, block in enumerate(self.transformer_blocks):
            out_memories.append(h.detach())
            h, attention_weights = block(memories[:, :, i], memories[:, :, i], h.unsqueeze(1), mask) # args: value, key, query, mask
            h = h.squeeze()
            if len(h.shape) == 1:
                h = h.unsqueeze(0)
        return h, torch.stack(out_memories, dim=1)

class GRUGatingUnit(torch.nn.Module):
    """
    Overview:
        GRU Gating Unit used in GTrXL.
        https://github.com/opendilab/DI-engine/blob/main/ding/torch_utils/network/gtrxl.py
    """

    def __init__(self, input_dim: int, bg: float = 2.):
        """
        Arguments:
            - input_dim: (:obj:`int`): dimension of input.
            - bg (:obj:`bg`): gate bias. By setting bg > 0 we can explicitly initialize the gating mechanism to
            be close to the identity map. This can greatly improve the learning speed and stability since it
            initializes the agent close to a Markovian policy (ignore attention at the beginning).
        """
        super(GRUGatingUnit, self).__init__()
        self.Wr = torch.nn.Linear(input_dim, input_dim, bias=False)
        self.Ur = torch.nn.Linear(input_dim, input_dim, bias=False)
        self.Wz = torch.nn.Linear(input_dim, input_dim, bias=False)
        self.Uz = torch.nn.Linear(input_dim, input_dim, bias=False)
        self.Wg = torch.nn.Linear(input_dim, input_dim, bias=False)
        self.Ug = torch.nn.Linear(input_dim, input_dim, bias=False)
        self.bg = nn.Parameter(torch.full([input_dim], bg))  # bias
        self.sigmoid = torch.nn.Sigmoid()
        self.tanh = torch.nn.Tanh()

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        """
        Overview:
            Compute output value with gating mechanism
        Arguments:
            - x: (:obj:`torch.Tensor`): first input.
            - y: (:obj:`torch.Tensor`): second input.
            x and y have same shape and last shape is input_dim.
        Returns:
            - g: (:obj:`torch.Tensor`): output of GRU. Same shape of x and y.
        """
        r = self.sigmoid(self.Wr(y) + self.Ur(x))
        z = self.sigmoid(self.Wz(y) + self.Uz(x) - self.bg)
        h = self.tanh(self.Wg(y) + self.Ug(torch.mul(r, x)))  # element wise multiplication
        g = torch.mul(1 - z, x) + torch.mul(z, h)
        return g  # x.shape == y.shape == g.shape

class GRUGate(nn.Module):
    """Implements a gated recurrent unit for use in AttentionNet
    https://github.com/ray-project/ray/blob/d909ac2b5fe5afe65feb192b49aca53abe962cee/rllib/models/torch/modules/gru_gate.py"""

    def __init__(self, dim: int, init_bias: int = 0.0, **kwargs):
        """
        input_shape (torch.Tensor): dimension of the input
        init_bias: Bias added to every input to stabilize training
        """
        super().__init__(**kwargs)
        # Xavier initialization of torch tensors
        self._w_r = nn.Parameter(torch.zeros(dim, dim))
        self._w_z = nn.Parameter(torch.zeros(dim, dim))
        self._w_h = nn.Parameter(torch.zeros(dim, dim))
        nn.init.xavier_uniform_(self._w_r)
        nn.init.xavier_uniform_(self._w_z)
        nn.init.xavier_uniform_(self._w_h)
        self.register_parameter("_w_r", self._w_r)
        self.register_parameter("_w_z", self._w_z)
        self.register_parameter("_w_h", self._w_h)

        self._u_r = nn.Parameter(torch.zeros(dim, dim))
        self._u_z = nn.Parameter(torch.zeros(dim, dim))
        self._u_h = nn.Parameter(torch.zeros(dim, dim))
        nn.init.xavier_uniform_(self._u_r)
        nn.init.xavier_uniform_(self._u_z)
        nn.init.xavier_uniform_(self._u_h)
        self.register_parameter("_u_r", self._u_r)
        self.register_parameter("_u_z", self._u_z)
        self.register_parameter("_u_h", self._u_h)

        self._bias_z = nn.Parameter(
            torch.zeros(
                dim,
            ).fill_(init_bias)
        )
        self.register_parameter("_bias_z", self._bias_z)

    def forward(self, inputs, **kwargs):
        # Pass in internal state first.
        h, X = inputs

        r = torch.tensordot(X, self._w_r, dims=1) + torch.tensordot(
            h, self._u_r, dims=1
        )
        r = torch.sigmoid(r)

        z = (
            torch.tensordot(X, self._w_z, dims=1)
            + torch.tensordot(h, self._u_z, dims=1)
            - self._bias_z
        )
        z = torch.sigmoid(z)

        h_next = torch.tensordot(X, self._w_h, dims=1) + torch.tensordot(
            (h * r), self._u_h, dims=1
        )
        h_next = torch.tanh(h_next)

        return (1 - z) * h + z * h_next