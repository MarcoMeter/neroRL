import numpy as np
import torch
import torch.nn.functional as F

from torch import nn
from einops import rearrange
from neroRL.nn.module import Module
from neroRL.utils.utils import batched_index_select

class MultiHeadAttention(nn.Module):
    """Multi Head Attention without dropout inspired by https://github.com/aladdinpersson/Machine-Learning-Collection
    https://youtu.be/U0s0f995w14"""
    def __init__(self, embed_dim, num_heads, share_heads):
        """
        Arguments:
            embed_dim {int} -- Size of the embedding dimension
            num_heads {int} -- Number of attention heads
            share_heads {bool} -- Whether to share the weights of the heads
        """
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.share_heads = share_heads
        self.head_size = embed_dim // num_heads

        assert (
            self.head_size * num_heads == embed_dim
        ), "Embedding dimension needs to be divisible by the number of heads"

        # Create the linear layers for the keys, values and queries
        if self.share_heads:
            # If we share the weights of the heads, we only need one set of linear layers
            self.values = nn.Linear(self.head_size, self.head_size, bias=False)
            self.keys = nn.Linear(self.head_size, self.head_size, bias=False)
            self.queries = nn.Linear(self.head_size, self.head_size, bias=False)
        else:
            # If we don't share the weights of the heads, we need as many sets of linear layers as there are heads
            self.values = nn.ModuleList([nn.Linear(self.head_size, self.head_size, bias=False) for _ in range(self.num_heads)])
            self.keys = nn.ModuleList([nn.Linear(self.head_size, self.head_size, bias=False) for _ in range(self.num_heads)])
            self.queries = nn.ModuleList([nn.Linear(self.head_size, self.head_size, bias=False) for _ in range(self.num_heads)])
            
        self.fc_out = nn.Linear(self.num_heads * self.head_size, embed_dim)

    def forward(self, values, keys, query, mask):
        """
        Arguments:
            values {torch.tensor} -- Values in shape of (N, L, D)
            keys {torch.tensor} -- Keys in shape of (N, L, D)
            query {torch.tensor} -- Queries in shape of (N, L, D)
            mask {torch.tensor} -- Attention mask in shape of (N, L)

        Returns:
            {torch.tensor} -- Output
            {torch.tensor} -- Attention weights
        """
        # Get number of training examples and sequence lengths
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # Split the embedding into self.num_heads different pieces
        values = values.reshape(N, value_len, self.num_heads, self.head_size)
        keys = keys.reshape(N, key_len, self.num_heads, self.head_size)
        query = query.reshape(N, query_len, self.num_heads, self.head_size)

        if self.share_heads:
            values = self.values(values)  # (N, value_len, heads, head_dim)
            keys = self.keys(keys)  # (N, key_len, heads, head_dim)
            queries = self.queries(query)  # (N, query_len, heads, heads_dim)
        else:
            # Apply the linear layer separately to each head
            values = [self.values[i](values[:, :, i].unsqueeze(2)) for i in range(self.num_heads)]
            keys = [self.keys[i](keys[:, :, i].unsqueeze(2)) for i in range(self.num_heads)]
            queries = [self.queries[i](query[:, :, i].unsqueeze(2)) for i in range(self.num_heads)]
            
            # Concatenate the different heads
            values = torch.cat(values, dim=2) # (N, value_len, heads, head_dim)
            keys = torch.cat(keys, dim=2) # (N, value_len, heads, head_dim)
            queries = torch.cat(queries, dim=2) # (N, value_len, heads, head_dim)

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
    def __init__(self, embed_dim, num_heads, share_heads, config):
        """Transformer Block made of LayerNorms, Multi Head Attention and one fully connected feed forward projection.

        Arguments:
            embed_dim {int} -- Size of the embeddding dimension
            num_heads {int} -- Number of attention headds
            share_heads {bool} -- Whether to share the weights of the heads
            config {dict} -- General config
        """
        super(TransformerBlock, self).__init__()

        # Attention
        self.attention = MultiHeadAttention(embed_dim, num_heads, share_heads)

        # Setup GTrXL if used
        self.use_gtrxl = config["gtrxl"]
        if self.use_gtrxl:
            self.gate1 = GRUGate(embed_dim, config["gtrxl_bias"], config["gtrxl_swap"])
            self.gate2 = GRUGate(embed_dim, config["gtrxl_bias"], config["gtrxl_swap"])

        # LayerNorms
        self.layer_norm = config["layer_norm"]
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        if self.layer_norm == "pre":
            self.norm_kv = nn.LayerNorm(embed_dim)

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

        # GRU Gate or skip connection
        if self.use_gtrxl:
            # Forward GRU gating
            h = self.gate1(query, attention)
        else:
            if self.layer_norm == "pre":
                attention = F.relu(attention)
            # Skip connection
            h = attention + query
        
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

        # GRU Gate or skip connection
        if self.use_gtrxl:
            # Forward GRU gating
            out = self.gate2(h, forward)
        else:
            if self.layer_norm == "pre":
                forward = F.relu(forward)
            # Skip connection
            out = forward + h
        
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
        self.share_heads = config["share_heads"]
        self.max_episode_steps = config["max_episode_steps"]
        self.activation = activation

        # Input embedding layer
        self.linear_embedding = nn.Linear(input_dim, self.embed_dim)
        nn.init.orthogonal_(self.linear_embedding.weight, np.sqrt(2))

        # Fusion layer
        self.fusion = nn.Linear(self.embed_dim * 2, self.embed_dim)
        nn.init.orthogonal_(self.fusion.weight, np.sqrt(2))

        # Determine positional encoding
        if config["positional_encoding"] == "relative":
            self.pos_embedding = SinusoidalPosition(dim = self.embed_dim)
        elif config["positional_encoding"] == "learned":
            self.pos_embedding = nn.Parameter(torch.randn(self.max_episode_steps, self.embed_dim)) # (batch size, max episoded steps, num layers, layer size)
        else:
            pass    # No positional encoding is used
        
        # Instantiate transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(self.embed_dim, self.num_heads, self.share_heads, config) 
            for _ in range(self.num_blocks)])

    def forward(self, h, memories, mask, memory_indices, hidden_state = None):
        """
        Arguments:
            h {torch.tensor} -- Input (query)
            memories {torch.tesnor} -- Whole episoded memories of shape (N, L, num blocks, D)
            mask {torch.tensor} -- Attention mask (dtype: bool) of shape (N, L)
            memory_indices {torch.tensor} -- Memory window indices (dtype: long) of shape (N, L)

        Returns:
            {torch.tensor} -- Output of the entire transformer encoder
            {torch.tensor} -- Out memories (i.e. inputs to the transformer blocks)
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

        # Select the last memory item of the last TrXL block
        # max_memory_indices = torch.max(memory_indices * mask, dim=1).values.long()
        # last_memory_indices = torch.clamp(max_memory_indices - 1, min=0, max=memories.size()[1] - 1)
        # last_memory_last_block = batched_index_select(memories[:, :, -1, :], 1, last_memory_indices.unsqueeze(0)).squeeze(0)

        h = self.fusion(torch.cat((h, hidden_state), dim=-1))
        h = self.activation(h)

        # # Add positional encoding to query
        # if self.config["positional_encoding"] == "relative":
        #     # apply mask to memory indices
        #     masked_memory_indices = memory_indices * mask
        #     # select max memory indices across dim 1
        #     masked_memory_indices = torch.max(memory_indices * mask, dim=1).values.long()
        #     pos_embedding = self.pos_embedding(self.max_episode_steps)[masked_memory_indices]
        #     h = h + pos_embedding

        # Forward transformer blocks
        out_memories = []
        for i, block in enumerate(self.transformer_blocks):
            out_memories.append(h.detach())
            h, attention_weights = block(memories[:, :, i], memories[:, :, i], h.unsqueeze(1), mask) # args: value, key, query, mask
            h = h.squeeze()
            if len(h.shape) == 1:
                h = h.unsqueeze(0)
        return h, torch.stack(out_memories, dim=1)
    
    def init_transformer_weights(self):
        if self.config["init_weights"] == "tfixup":
            for p in self.transformer_blocks.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

            temp_state_dic = {}
            for name, param in self.transformer_blocks.named_parameters():
                if any(s in name for s in ["linear1.weight", "linear2.weight", "fc_out.weight"]):
                    temp_state_dic[name] = (0.67 * (self.num_layers) ** (- 1. / 4.)) * param
                # elif "self_attn.in_proj_weight" in name:
                #     temp_state_dic[name] = (0.67 * (self.num_layers) ** (- 1. / 4.)) * (param * (2**0.5))

            for name in self.transformer_blocks.state_dict():
                if name not in temp_state_dic:
                    temp_state_dic[name] = self.transformer_blocks.state_dict()[name]
            self.transformer_blocks.load_state_dict(temp_state_dic)
        elif self.config["init_weights"] == "xavier":
            for p in self.transformer_blocks.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
        elif self.config["init_weights"] == "orthogonal":
            for p in self.transformer_blocks.parameters():
                if p.dim() > 1:
                    nn.init.orthogonal_(p, np.sqrt(2))
        elif self.config["init_weights"] == "kaiming":
            for p in self.transformer_blocks.parameters():
                if p.dim() > 1:
                    nn.init.kaiming_uniform_(p)

class GRUGate(torch.nn.Module):
    """
    Overview:
        GRU Gating Unit used in GTrXL.
        Inspired by https://github.com/dhruvramani/Transformers-RL/blob/master/layers.py
    """

    def __init__(self, input_dim: int, bg: float = 0.0, swap_inputs:bool = False):
        """
        Arguments:
            input_dim {int} -- Input dimension
            bg {float} -- Initial gate bias value. By setting bg > 0 we can explicitly initialize the gating mechanism to
            be close to the identity map. This can greatly improve the learning speed and stability since it
            initializes the agent close to a Markovian policy (ignore attention at the beginning). (default: {0.0})
            swap_inputs {bool} -- Swap GRU inputs (default: {False})
        """
        super(GRUGate, self).__init__()
        self.swap_inputs = swap_inputs
        self.Wr = torch.nn.Linear(input_dim, input_dim, bias=False)
        self.Ur = torch.nn.Linear(input_dim, input_dim, bias=False)
        self.Wz = torch.nn.Linear(input_dim, input_dim, bias=False)
        self.Uz = torch.nn.Linear(input_dim, input_dim, bias=False)
        self.Wg = torch.nn.Linear(input_dim, input_dim, bias=False)
        self.Ug = torch.nn.Linear(input_dim, input_dim, bias=False)
        self.bg = nn.Parameter(torch.full([input_dim], float(bg)))  # bias
        self.sigmoid = torch.nn.Sigmoid()
        self.tanh = torch.nn.Tanh()
        nn.init.xavier_uniform_(self.Wr.weight)
        nn.init.xavier_uniform_(self.Ur.weight)
        nn.init.xavier_uniform_(self.Wz.weight)
        nn.init.xavier_uniform_(self.Uz.weight)
        nn.init.xavier_uniform_(self.Wg.weight)
        nn.init.xavier_uniform_(self.Ug.weight)

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        """        
        Arguments:
            x {torch.Tensor} -- First input
            y {torch.Tensor} -- Second input

        Returns:
            {torch.tensor} -- Output
        """
        if not self.swap_inputs:
            r = self.sigmoid(self.Wr(y) + self.Ur(x))
            z = self.sigmoid(self.Wz(y) + self.Uz(x) - self.bg)
            h = self.tanh(self.Wg(y) + self.Ug(torch.mul(r, x)))
            return torch.mul(1 - z, x) + torch.mul(z, h)
        else:
            r = self.sigmoid(self.Wr(x) + self.Ur(y))
            z = self.sigmoid(self.Wz(x) + self.Uz(y) - self.bg)
            h = self.tanh(self.Wg(x) + self.Ug(torch.mul(r, y)))
            return torch.mul(1 - z, y) + torch.mul(z, h)
