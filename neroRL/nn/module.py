import torch
from torch import nn

class Module(nn.Module):
    def __init__(self):
        super().__init__()

    def grad_norm(self):
        grads = []
        for name, parameter in self.named_parameters():
            grads.append(parameter.grad.view(-1))
        return torch.linalg.norm(torch.cat(grads)).item()

    def grad_mean(self):
        grads = []
        for name, parameter in self.named_parameters():
            grads.append(parameter.grad.view(-1))
        return torch.mean(torch.cat(grads)).item()