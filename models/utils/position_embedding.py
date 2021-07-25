import torch

from torch import nn, Tensor


class PositionEmbeddingMatrix(nn.Module):
    """
    This is a standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    Implementation based on the Detection Transformer:
    https://github.com/facebookresearch/detr/blob/master/models/position_encoding.py
    """
    def __init__(self, num_pos_feats=64, scale=None, temperature=1e5):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.scale = scale
        self.temperature = temperature        

    def forward(self, tensor: Tensor):
        device = tensor.device
        C, H, W = tensor.shape
        
        # create a grid of x and y cooridnates
        y_embed, x_embed = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device))
        
        # scale both x and y from [0, W], [0, H] to [0, scale]
        if self.scale is not None:
            eps = 1e-6
            y_embed = y_embed / (y_embed[-1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, -1:] + eps) * self.scale

        # divide x and y in the last axis by values in (a nonlinear) range [1, temperature]
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)
        pos_x = x_embed[:, :, None] / dim_t[None, None, :]
        pos_y = y_embed[:, :, None] / dim_t[None, None, :]
        
        # combine sin(x), cos(x), sin(y), cos(y) 
        pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
        pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
        pos = torch.cat((pos_y, pos_x), dim=2).permute(2, 0, 1)
        
        return pos


class PositionEmbeddingPoints(nn.Module):
    """
    This is a standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    Implementation based on the Detection Transformer:
    https://github.com/facebookresearch/detr/blob/master/models/position_encoding.py
    """
    def __init__(self, num_pos_feats=64, scale=100, temperature=1e5):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.scale = scale
        self.temperature = temperature        

    def forward(self, tensor: Tensor):
        device = tensor.device
        x_embed = tensor[:, 0]
        y_embed = tensor[:, 1]
        
        # scale both x and y from [0, W], [0, H] to [0, scale]
        if self.scale is not None:
            y_embed *= self.scale
            x_embed *= self.scale

        # divide x and y in the last axis by values in (a nonlinear) range [1, temperature]
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)
        pos_x = x_embed[:, None] / dim_t[None, :]
        pos_y = y_embed[:, None] / dim_t[None, :]
        
        # combine sin(x), cos(x), sin(y), cos(y) 
        pos_x = torch.stack((pos_x[:, 0::2].sin(), pos_x[:, 1::2].cos()), dim=2).flatten(1)
        pos_y = torch.stack((pos_y[:, 0::2].sin(), pos_y[:, 1::2].cos()), dim=2).flatten(1)
        pos = torch.cat((pos_y, pos_x), dim=1)
        
        return pos
