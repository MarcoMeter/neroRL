import numpy as np
import torch
from torch import nn
from torch.distributions.categorical import Categorical

from neroRL.nn.module import Module

class MultiDiscreteActionPolicy(Module):
    """Multi-Discrete Action Space based on categorical distributions"""
    def __init__(self, in_features, action_space_shape):
        """
        Arguments:
            in_features {int} -- Number of to be fed features
            action_space_shape {tuple} -- Shape of the action space
        """
        super().__init__()
        # Linear layer before head
        self.linear = nn.Linear(in_features=in_features, out_features=512)
        nn.init.orthogonal_(self.linear.weight, np.sqrt(2))
        # Define policy/action dimensions
        self.policy_branches = nn.ModuleList()
        for num_actions in action_space_shape:
            actor_branch = nn.Linear(in_features=512, out_features=num_actions)
            nn.init.orthogonal_(actor_branch.weight, np.sqrt(0.01))
            self.policy_branches.append(actor_branch)

    def forward(self, h, activ_fn):
        """
        Arguments:
            h {torch.tensor} -- The fed input data
            activ_fn {function} -- The to be applied activation function to the linear layer before feeding the head

        Returns:
            {list} --  A list containing categorical distributions for each action dimension
        """
        h = activ_fn(self.linear(h))
        pi = []
        for i, branch in enumerate(self.policy_branches):
            pi.append(Categorical(logits=self.policy_branches[i](h)))
        return pi

class ValueEstimator(Module):
    """Estimation of the value function as part of the agnet's critic"""
    def __init__(self, in_features):
        """
        Arguments:
            in_features {int}: Number of to be fed features
        """
        super().__init__()
        # Linear layer before head
        self.linear = nn.Linear(in_features=in_features, out_features=512)
        nn.init.orthogonal_(self.linear.weight, np.sqrt(2))
        # Value head
        self.value = nn.Linear(in_features=512, out_features=1)
        nn.init.orthogonal_(self.value.weight, 1)

    def forward(self, h, activ_fn):
        """
        Arguments:
            h {toch.tensor} -- The fed input data
            activ_fn {function} -- The to be applied activation function to the linear layer before feeding the head

        Returns:
            {torch.tensor}: Estimated value
        """
        h = activ_fn(self.linear(h))
        return self.value(h).reshape(-1)

class AdvantageEstimator(Module):
    """Used by the DAAC Algorithm by Raileanu & Fergus, 2021, https://arxiv.org/abs/2102.10330"""
    def __init__(self, in_features, action_space_shape):
        super().__init__()
        self.advantage = nn.Linear(in_features=in_features + sum(action_space_shape), out_features=1)
        nn.init.orthogonal_(self.advantage.weight, 0.01)

    def forward(self, h, one_hot_actions):
        h = torch.cat(h, one_hot_actions)
        return self.advantage(h).reshape(-1)
