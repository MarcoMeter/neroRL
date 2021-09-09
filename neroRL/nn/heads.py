import numpy as np
import torch
from torch import nn
from torch.distributions.categorical import Categorical

from neroRL.nn.module import Module

class MultiDiscreteActionPolicy(Module):
    def __init__(self, in_features, action_space_shape):
        super().__init__()

        # Define Policy branches
        self.policy_branches = nn.ModuleList()
        for num_actions in action_space_shape:
            actor_branch = nn.Linear(in_features=in_features, out_features=num_actions)
            nn.init.orthogonal_(actor_branch.weight, np.sqrt(0.01))
            self.policy_branches.append(actor_branch)

    def forward(self, h):
        pi = []
        for i, branch in enumerate(self.policy_branches):
            pi.append(Categorical(logits=self.policy_branches[i](h)))
        return pi

class AdvantageEstimator(Module):
    def __init__(self, in_features, action_space_shape):
        super().__init__()
        self.advantage = nn.Linear(in_features=in_features + sum(action_space_shape), out_features=1)
        nn.init.orthogonal_(self.advantage.weight, 0.01)

    def forward(self, h, one_hot_actions):
        h = torch.cat(h, one_hot_actions)
        return self.advantage(h).reshape(-1)

class ValueEstimator(Module):
    def __init__(self, in_features):
        super().__init__()
        self.value = nn.Linear(in_features=in_features, out_features=1)
        nn.init.orthogonal_(self.value.weight, 1)

    def forward(self, h):
        return self.value(h).reshape(-1)