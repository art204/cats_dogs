import torch
from torch import nn


class Flatten(nn.Module):
    def forward(self, x):
        return torch.flatten(x, start_dim=1)
