import torch
from torch import nn

class Module(nn.Module):
    """nn.Module is extended by functions to compute the norm and the mean of this module's parameters."""
    def __init__(self):
        super().__init__()

    def grad_norm(self):
        """Concatenates the gradient of this module's parameters and then computes the norm.

        Returns:
            {float}: Returns the norm of the gradients of this model's parameters. Returns None if no parameters are available.
        """
        grads = []
        for name, parameter in self.named_parameters():
            grads.append(parameter.grad.view(-1))
        return torch.linalg.norm(torch.cat(grads)).item() if len(grads) > 0 else None

    def grad_mean(self):
        """Concatenates the gradient of this module's parameters and then computes the mean.

        Returns:
            {float}: Returns the mean of the gradients of this module's parameters. Returns None if no parameters are available.
        """
        grads = []
        for name, parameter in self.named_parameters():
            grads.append(parameter.grad.view(-1))
        return torch.mean(torch.cat(grads)).item() if len(grads) > 0 else None

class Sequential(nn.Sequential):
    """nn.Sequential is extended by functions to compute the norm and the mean of this module's parameters."""
    def grad_norm(self):
        """Concatenates the gradient of this module's parameters and then computes the norm.

        Returns:
            {float}: Returns the norm of the gradients of this module's parameters. Returns None if no parameters are available.
        """
        grads = []
        for name, parameter in self.named_parameters():
            grads.append(parameter.grad.view(-1))
        return torch.linalg.norm(torch.cat(grads)).item() if len(grads) > 0 else None

    def grad_mean(self):
        """Concatenates the gradient of this module's parameters and then computes the mean.

        Returns:
            {float}: Returns the mean of the gradients of this module's parameters. Returns None if no parameters are available.
        """
        grads = []
        for name, parameter in self.named_parameters():
            grads.append(parameter.grad.view(-1))
        return torch.mean(torch.cat(grads)).item() if len(grads) > 0 else None