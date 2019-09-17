import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Dirichlet
from torch.distributions.kl import kl_divergence


class PriorDirichlet(nn.Module):
    def __init__(self, input_dim, output_dim, a0):
        super().__init__()
        self.output_dim = output_dim
        h = 50
        self.fc1 = nn.Linear(input_dim, h)
        self.fc2 = nn.Linear(h, output_dim)

        self.eps = 0.01
        self.a0 = a0
        self.eps_var = 1e-6

    def forward(self, x):
        x = F.relu(self.fc1(x))
        logits = self.fc2(x)
        alpha = torch.exp(logits)
        precision = torch.sum(alpha, 1)
        mean = F.softmax(logits, 1)

        return mean, alpha, precision

    def _kl_loss(self, y_precision, y_alpha, precision, alpha):

        loss = torch.lgamma(y_precision + self.eps_var) - torch.sum(torch.lgamma(y_alpha + self.eps_var), 1) \
            - torch.lgamma(precision + self.eps_var) \
            + torch.sum(torch.lgamma(alpha + self.eps_var), 1)

        l2 = torch.sum((y_alpha-alpha)
                       * (torch.digamma(y_alpha + self.eps_var) - torch.digamma(alpha + self.eps_var)), 1)

        return loss + l2

    def kl_loss(self, x_in, x_out, target):
        mu_in, alpha_in, precision_in = self.forward(x_in)
        mu_out, alpha_out, precision_out = self.forward(x_out)

        precision_in = precision_in.unsqueeze(1)
        precision_out = precision_out.unsqueeze(1)

        # KL OUT
        target_out_alpha = torch.ones((x_in.shape[0], self.output_dim)).float()
        target_out_precision = self.output_dim * torch.ones(x_in.shape[0], 1)

        kl_out = self._kl_loss(target_out_precision,
                               target_out_alpha, precision_out, alpha_out)

        # KL IN
        y_onehot = torch.FloatTensor(x_in.shape[0], self.output_dim)
        y_onehot.zero_()

        y_onehot.scatter_(1, target, 1)

        mu_c = y_onehot * (1 - self.eps*(self.output_dim-1))
        mu_c += self.eps * torch.ones_like(y_onehot)

        target_in_alpha = mu_c * self.a0
        target_in_precision = self.a0 * torch.ones((x_in.shape[0], 1))

        kl_in = self._kl_loss(target_in_precision,
                              target_in_alpha, precision_in, alpha_in)

        return torch.mean(kl_in + kl_out)


class PriorCNN(PriorDirichlet):
    def __init__(self, input_dim, output_dim, a0):
        super().__init__(input_dim, output_dim, a0)
        c = input_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.a0 = a0
        self.layers = nn.Sequential(nn.Conv2d(c, 16, 5, 1),
                                    nn.ReLU(),
                                    nn.Conv2d(16, 32, 5, 1))

        self.max_pool = nn.MaxPool2d(2)

        self.fc1 = nn.Linear(3200, 100)
        self.fc2 = nn.Linear(100, output_dim)

    def forward(self, x):
        x = self.layers(x)
        x = self.max_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        logits = self.fc2(x)

        alpha = torch.exp(logits)
        precision = torch.sum(alpha, 1)
        mean = F.softmax(logits, 1)

        return mean, alpha, precision


class PriorEnsemble(nn.Module):
    def __init__(self, nets):
        super().__init__()

        if isinstance(nets, dict):
            modules = list(nets.values())
        if isinstance(nets, list):
            modules = nets

        self.ensemble = nn.ModuleList(modules)

    def forward(self, x):
        probs, alphas, precs = [], [], []
        for model in self.ensemble:
            mean, alpha, precision = model(x)
            probs.append(mean)
            alphas.append(alpha)
            precs.append(precision)

        return torch.stack(probs), torch.stack(alphas), torch.stack(precs)
