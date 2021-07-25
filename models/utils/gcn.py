import torch
import copy

from torch import nn
from torch.nn import functional as F
from torch.nn.modules.container import ModuleList


class NeighborAttention(nn.Module):
    """
    A graph-based attention replacement.
    """
    def __init__(self, n, d_model):
        super().__init__()
        self.n = n
        self.linear = nn.Linear((n+1)*d_model, d_model) # [(n+1)*L -> L]
    
    def forward(self, features, rois):
        # compute the center of each roi
        roi_centers = rois.mean(1)
        
        # for each roi, find the n nearest rois (+ self)
        dx = roi_centers[None, :, 0] - roi_centers[:, 0, None] # [L, L]
        dy = roi_centers[None, :, 1] - roi_centers[:, 1, None] # [L, L]
        d = torch.sqrt(dx**2 + dy**2) # [L, L]
        idx = torch.argsort(d, 1)[:, :(self.n+1)] # [L, n+1]

        # for each roi, 'attend' to self and the n nearest rois
        features = features[idx, :] # [L, n+1, C]
        features = features.permute(0, 2, 1) # [L, C, n+1]
        features = features.flatten(1) # [L, C*(n+1)]
        features = self.linear(features) # [L, c]
        features = F.relu(features) # [L, c]
        
        return features


class NeighborEncoderLayer(nn.Module):
    """
    Copy-paste of torch.nn.TransformerEncoderLayer but
    uses 'NeighborAttention' instead of the regural
    torch.nn.MultiheadAttention.
    """
    def __init__(self, n=2, d_model=256, dim_feedforward=2048, dropout=1e-3):
        super().__init__()
        self.attn = NeighborAttention(n, d_model)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, features, rois):
        features2 = self.attn(features, rois)
        features = features + self.dropout1(features2)
        features = self.norm1(features)
        features2 = self.linear2(self.dropout(F.relu(self.linear1(features))))
        features = features + self.dropout2(features2)
        features = self.norm2(features)
        return features


class SimpleNeighborEncoderLayer(nn.Module):
    """
    Copy-paste of torch.nn.TransformerEncoderLayer but
    uses 'NeighborAttention' instead of the regural
    torch.nn.MultiheadAttention.
    """
    def __init__(self, n=2, d_model=256, dim_feedforward=2048):
        super().__init__()
        self.attn = NeighborAttention(n, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, features, rois):
        features = features + self.attn(features, rois)
        features = F.relu(self.norm(features))
        return features


class GCN(nn.Module):
    """
    NeighborEncoder is a stack of NeighborEncoderLayers.
    """
    def __init__(self, num_layers=2, *args, **kwargs):
        super().__init__()
        self.layers = ModuleList([NeighborEncoderLayer(*args, **kwargs) for i in range(num_layers)])
    
    def forward(self, features, rois):
        for layer in self.layers:
            features = layer(features, rois)

        return features
