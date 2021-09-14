import numpy as np
import torch
from torch import nn
from torch.distributions.categorical import Categorical
import torch.nn.functional as F

from neroRL.nn.module import Module

class MultiDiscreteActionPolicy(Module):
    """Multi-Discrete Action Space based on categorical distributions"""
    def __init__(self, in_features, action_space_shape, activ_fn):
        """
        Arguments:
            in_features {int} -- Number of to be fed features
            action_space_shape {tuple} -- Shape of the action space
            activ_fn {function} -- The to be applied activation function to the linear layer before feeding the head
        """
        super().__init__()
        # Set the activation function
        self.activ_fn = activ_fn
        # Linear layer before head
        self.linear = nn.Linear(in_features=in_features, out_features=512)
        nn.init.orthogonal_(self.linear.weight, np.sqrt(2))
        # Define policy/action dimensions
        self.policy_branches = nn.ModuleList()
        for num_actions in action_space_shape:
            actor_branch = nn.Linear(in_features=512, out_features=num_actions)
            nn.init.orthogonal_(actor_branch.weight, np.sqrt(0.01))
            self.policy_branches.append(actor_branch)

    def forward(self, h):
        """
        Arguments:
            h {torch.tensor} -- The fed input data

        Returns:
            {list} --  A list containing categorical distributions for each action dimension
        """
        h = self.activ_fn(self.linear(h))
        return [Categorical(logits=branch(h)) for branch in self.policy_branches]

class ValueEstimator(Module):
    """Estimation of the value function as part of the agnet's critic"""
    def __init__(self, in_features, activ_fn):
        """
        Arguments:
            in_features {int} -- Number of to be fed features
        """
        super().__init__()
        # Set the activation function
        self.activ_fn = activ_fn
        # Linear layer before head
        self.linear = nn.Linear(in_features=in_features, out_features=512)
        nn.init.orthogonal_(self.linear.weight, np.sqrt(2))
        # Value head
        self.value = nn.Linear(in_features=512, out_features=1)
        nn.init.orthogonal_(self.value.weight, 1)

    def forward(self, h):
        """
        Arguments:
            h {toch.tensor} -- The fed input data

        Returns:
            {torch.tensor} -- Estimated value
        """
        h = self.activ_fn(self.linear(h))
        return self.value(h).reshape(-1)

class AdvantageEstimator(Module):
    """Used by the DAAC Algorithm by Raileanu & Fergus, 2021, https://arxiv.org/abs/2102.10330"""
    def __init__(self, in_features, action_space_shape):
        """
        Arguments:
            in_features {int} -- Number of to be fed features
            action_space_shape {tuple} -- Dimensions of the action space
        """
        super().__init__()
        # Set action space
        self.action_space_shape = action_space_shape
        # Calculate the total number of actions
        self.total_num_actions = sum(action_space_shape)
        # Advantage head
        self.advantage = nn.Linear(in_features=in_features + self.total_num_actions, out_features=1)
        nn.init.orthogonal_(self.advantage.weight, 0.01)

    def forward(self, h, actions, device):
        """
        Arguments:
            h {toch.tensor} -- The fed input data
            actions {toch.tensor} -- The actions of the agent
            device {torch.device} -- Current device

        Returns:
            {torch.tensor} -- Estimated advantage function
        """
        if actions is None:
            one_hot_actions = torch.zeros(h.shape[0], self.total_num_actions).to(device)
            h = torch.cat((h, one_hot_actions), dim=1)
        else:
            for i in range(len(self.action_space_shape)):
                action, num_actions = actions[:, i], self.action_space_shape[i]
                one_hot_actions = F.one_hot(action.long().squeeze(-1), num_actions).float()
                h = torch.cat((h, one_hot_actions), dim=1)
        
        return self.advantage(h).reshape(-1)
