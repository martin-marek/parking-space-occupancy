import torch

from torch import nn, Tensor
from typing import Optional, List
from torch.nn.functional import grid_sample


@torch.jit.script
def linspace(start: Tensor, stop: Tensor, num: int):
    """
    Creates a tensor of shape [num, *start.shape] whose values are evenly spaced from start to end, inclusive.
    Replicates the multi-dimensional bahaviour of numpy.linspace in PyTorch.
    """
    # create a tensor of 'num' steps from 0 to 1
    steps = torch.arange(num, dtype=torch.float32, device=start.device) / (num - 1)
    
    # reshape the 'steps' tensor to [-1, *([1]*start.ndim)] to allow for broadcastings
    # - using 'steps.reshape([-1, *([1]*start.ndim)])' would be nice here but torchscript
    #   "cannot statically infer the expected size of a list in this contex", hence the code below
    for i in range(start.ndim):
        steps = steps.unsqueeze(-1)
    
    # the output starts at 'start' and increments until 'stop' in each dimension
    out = start[None] + steps*(stop - start)[None]
    
    return out


@torch.jit.script
def roi_grid(rois: Tensor, size: int = 3):
    """
    Using rois of shape [N, 4, 2 (x, y)] as input, returns a
    sampled grid of [size (y), size (x)] points within each roi.
    Output has shape [N, size (y), size (x), 2 (x, y)].
    """
    # interpolate two opposite edges of each roi
    idx_edge_1 = linspace(start=rois[:, 1], stop=rois[:, 2], num=size)
    idx_edge_2 = linspace(start=rois[:, 0], stop=rois[:, 3], num=size)
    
    # interpolate between the edges of each roi
    rois_interpolated = linspace(start=idx_edge_1, stop=idx_edge_2, num=size)
    
    # reshape the output from [size_x, size_y, N, 2] to [N, size_y, size_x, 2]
    rois_interpolated = rois_interpolated.permute([2, 1, 0, 3])
            
    return rois_interpolated


@torch.jit.script
def index_by(tensor: Tensor, idx: Tensor):
    """
    Returns values of 'tensor' indexed by the relative cooridnates of 'idx'.
    'idx' must be in range [0, 1].
    """
    assert idx.min() >= 0
    assert idx.max() <= 1
    
    # convert indices from relative to absolute
    _, h, w = tensor.shape
    idx_abs = idx.clone()
    idx_abs[:, 0] *= w-1
    idx_abs[:, 1] *= h-1
    idx_abs = idx_abs.round().to(torch.long)

    # get tensor values indexed by `idx_abs`, results in shape [c, *idx.shape]
    tensor = tensor[:, idx_abs[..., 1], idx_abs[..., 0]]

    # permute the tensor as [idx.shape[0], c, *idx.shape[1:]]
    tensor = tensor.transpose(1, 0)

    return tensor


@torch.jit.script
def roi_pool_square(tensor: Tensor, rois: Tensor, size: int = 3):
    """
    Pool a square arond each ROI from 'tensor'.
    For an ROI of size [w, h], a square of size
    [s, s] is pooled, where s = max(w, h).
    'rois' must be in range [0, 1], have shape [N, 4, 2 (x, y)] and
    be ordered along the second axis (clockwise or anticlockwise).
    """
    # compute coordinates of the squares
    w = torch.amax(rois[:, :, 0], 1) - torch.amin(rois[:, :, 0], 1)
    h = torch.amax(rois[:, :, 1], 1) - torch.amin(rois[:, :, 1], 1)
    c = torch.mean(rois, 1, keepdim=True)
    c = c.repeat(1, 4, 1)
    c[:, 0, 0] += w/2
    c[:, 0, 1] += h/2
    c[:, 1, 0] -= w/2
    c[:, 1, 1] += h/2
    c[:, 2, 0] -= w/2
    c[:, 2, 1] -= h/2
    c[:, 3, 0] += w/2
    c[:, 3, 1] -= h/2
    
    # compute a grid of indices for each roi
    rois_interpolated = roi_grid(c, size)
    
    # translate the grid from [0, 1] to [-1, 1] coordinates
    rois_interpolated = (rois_interpolated * 2) - 1
    
    # gather the values indexed by the ROI grids
    warps = torch.stack([grid_sample(tensor[None], r[None], align_corners=True)[0] for r in rois_interpolated])
    
    return warps


@torch.jit.script
def roi_pool_qdrl(tensor: Tensor, rois: Tensor, size: int = 3):
    """
    Pool quadrilateral ROIs from 'tensor'.
    'rois' must be in range [0, 1], have shape [N, 4, 2 (x, y)] and
    be ordered along the second axis (clockwise or anticlockwise).
    """    
    # compute a grid of indices for each roi
    rois_interpolated = roi_grid(rois, size)
    
    # translate the grid from [0, 1] to [-1, 1] coordinates
    rois_interpolated = (rois_interpolated * 2) - 1
    
    # gather the values indexed by the ROI grids
    warps = torch.stack([grid_sample(tensor[None], r[None], align_corners=True)[0] for r in rois_interpolated])
    
    return warps


@torch.jit.script
def roi_pool(tensor: Tensor, rois: Tensor, size: int = 3, pooling_type: str = 'square'):
    """
    Returns either a square or a quadrilateral pooler.
    """
    if pooling_type == 'square':
        return roi_pool_square(tensor, rois, size)
    elif pooling_type == 'qdrl':
        return roi_pool_qdrl(tensor, rois, size)
    else:
        raise Exception(f'unknown pooling method "{pooling_type}"')


@torch.jit.script
def get_level_idx(features: List[Tensor], rois: Tensor, size: int):
    """
    The original pooling heuristic proposed in "Feature Pyramid Networks for Object Detection". 
    """
    # get the approximate res of input image
    scale_factor = 4 # resnet value
    image_h = scale_factor * features[0].shape[2]
    image_w = scale_factor * features[0].shape[3]
    
    # get the res of each roi in relative units
    w = rois[:, :, 0].amax(1) - rois[:, :, 0].amin(1)
    h = rois[:, :, 1].amax(1) - rois[:, :, 1].amin(1)
    
    # get the res of each roi in terms of input image
    w = w * image_w
    h = h * image_h
    
    # calculate the target level idx for each roi using the heuristic proposed in the fpn paper
    k_min = 2 # resnet value
    k_max = 5 # resnet value
    lvl_0 = 4 # this is a potential hyper-param
    k = lvl_0 + torch.log2(torch.sqrt(w*h) / 224.)
    k = k.floor().int()
    k = k.clamp(k_min, k_max)
    
    # return the target layer as indexed in a list
    roi_pooling_level = k - k_min
    return roi_pooling_level


@torch.jit.script
def pool_FPN_features(features: List[Tensor], rois: Tensor, size: int, pooling_type: str = 'square'):
    """
    Pool quadrilateral ROIs from a feature pyramid.
    'rois' must be in range [0, 1], have shape [N, 4, 2 (x, y)] and
    be ordered along the second axis (clockwise or anticlockwise).
    """
    device = features[0].device
    
    # calculate the target level idx for each roi
    roi_pooling_level = get_level_idx(features, rois, size)
    
    # pool the selected levels
    c = features[0].shape[1]
    pooled_rois = torch.zeros((len(rois), c, size, size), device=device)
    for level, level_features in enumerate(features):
        # select those rois with a scale corresponding to current level
        idx_in_level = torch.nonzero(level == roi_pooling_level).squeeze(1)
        if len(idx_in_level) > 0:
            rois_per_level = rois[idx_in_level]
            pooled_rois[idx_in_level] = roi_pool(level_features[0], rois_per_level, size, pooling_type)
    
    return pooled_rois
