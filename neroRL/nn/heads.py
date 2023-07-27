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
            activ_fn {function} -- The to be applied activation function to the linear layer
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
 
class GroundTruthEstimator(Module):
    """Estimation of the environment's ground truth information"""
    def __init__(self, in_features, out_features, activ_fn):
        """
        Arguments:
            in_features {int} -- Number of to be fed features
            out_features {int} -- Number of to be estimated features
            activ_fn {function} -- The to be applied activation function to the linear layer
        """
        super().__init__()
        # Set the activation function
        self.activ_fn = activ_fn
        # Linear layer before head
        self.linear = nn.Linear(in_features=in_features, out_features=512)
        nn.init.orthogonal_(self.linear.weight, np.sqrt(2))
        # Estimaton head
        self.ground_truth_estimation = nn.Linear(in_features=512, out_features=out_features)
        nn.init.orthogonal_(self.ground_truth_estimation.weight, np.sqrt(2))

    def forward(self, h):
        """
        Arguments:
            h {toch.tensor} -- The fed input data

        Returns:
            {torch.tensor} -- Estimated ground truth information
        """
        h = self.activ_fn(self.linear(h))
        return self.ground_truth_estimation(h)
