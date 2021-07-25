import torch
import torch.nn.functional as F

from torch import nn


class ClassificationHead(nn.Module):
    """A standard 3-layer fully-connected classification head."""
    def __init__(self, in_channels, representation_size=1024):
        super().__init__()
        self.fc1 = nn.Linear(in_channels, representation_size)
        self.fc2 = nn.Linear(representation_size, representation_size)
        self.fc3 = nn.Linear(representation_size, 2)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x