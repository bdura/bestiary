import numpy as np
import torch
from torch import autograd
from torch import nn


class Critic(nn.Module):
    """
    PyTorch module that encapsulates critics for unsupervised learning.
    """

    def __init__(self, module: torch.nn.Module, **kwargs):
        """
        Parameters
        ----------
        module: The module class that will perform the forward pass.
            Its output should be one-dimensional logits:
            depending on the critic, a sigmoid might be applied.
        kwargs: Additional arguments to pass to the module class for initialisation.
        """
        super(Critic, self).__init__()

        self.module_ = module(**kwargs)

    def forward(self, x):
        """Performs the forward pass"""
        return self.module_(x)

    def loss(self, f, g):
        """Computes the critic loss."""
        pass

    def distance(self, f, g):
        """Computes the distance"""
        pass


class JensenShannon(Critic):
    """
    The Jensen-Shannon divergence is the probability distance used in Ian Goodfellow's paper on GANs.
    """

    def __init__(self, module, **kwargs):
        super(JensenShannon, self).__init__(module, **kwargs)

    def forward(self, x):
        x = self.module_(x)
        x = torch.sigmoid(x)
        return x

    def loss(self, f: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        """
        Computes the critic loss.

        Parameters
        ----------
        f: (B, *) tensor, containing B samples from the f distribution.
        g: (B, *) tensor (same as f), containing B samples from the g distribution.

        Returns
        -------
        loss: The critic loss.
        """
        objective_f = self(f).log().mean()
        objective_g = (1 - self(g)).log().mean()

        objective = objective_f / 2. + objective_g / 2.

        # Optimisers minimise a loss
        loss = - objective

        return loss

    def distance(self, f: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        """
        Computes the distance between the two
        Parameters
        ----------
        f
        g

        Returns
        -------

        """
        return np.log(2) - self.loss(f, g)


class Wasserstein(Critic):

    def __init__(self, module, kappa=10, **kwargs):
        super(Wasserstein, self).__init__(module, **kwargs)
        self.kappa = kappa

    def loss(self, f: torch.Tensor, g: torch.Tensor, penalised: bool = True) -> torch.Tensor:
        """
        Computes the critic loss for the Wasserstein distance.

        Parameters
        ----------
        f: (B, *) tensor, containing B samples from the f distribution.
        g: (B, *) tensor (same as f), containing B samples from the g distribution.
        penalised: Whether to use the penalised form of the critic.
            (Useful for training the critic).

        Returns
        -------
        loss: The critic loss.
        """
        device = f.device

        objective_f = self(f).mean()
        objective_g = self(g).mean()

        objective = objective_f - objective_g

        if penalised:
            # Uniform distribution U[0, 1]
            a = torch.randn(f.shape[0], ).to(device).unsqueeze(1)

            if len(f.shape) > 2:
                a = a.unsqueeze(2).unsqueeze(2)

            z = a * f + (1 - a) * g
            z.requires_grad_(True)

            gradient = autograd.grad(self(z).sum(), z, create_graph=True)[0]

            norm_gradient = torch.norm(gradient, dim=1)

            penalty = (norm_gradient - 1).pow(2).mean()

            objective = objective - self.kappa * penalty

        loss = - objective

        return loss

    def distance(self, f, g):
        return - self.loss(f, g, penalised=False)
