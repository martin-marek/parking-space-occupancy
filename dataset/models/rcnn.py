from torch import nn
from torchvision.models import resnet50
from torchvision.ops.misc import FrozenBatchNorm2d

from .utils import pooling


class RCNN(nn.Module):
    """
    An R-CNN inspired parking lot classifier.
    Pools ROIs directly from image and separately
    passes each pooled ROI through a CNN.
    """
    def __init__(self, roi_res=100, pooling_type='square'):
        super().__init__()
        # load backbone
        self.backbone = resnet50(pretrained=True, norm_layer=FrozenBatchNorm2d)
        self.backbone.fc = nn.Linear(in_features=2048, out_features=2)
        
        # freeze bottom layers
        layers_to_train = ['layer4', 'layer3', 'layer2']
        for name, parameter in self.backbone.named_parameters():
            if all([not name.startswith(layer) for layer in layers_to_train]):
                parameter.requires_grad_(False)
        
        # ROI pooling
        self.roi_res = roi_res
        self.pooling_type = pooling_type
        
    def forward(self, image, rois):
        # pool ROIs from image
        warps = pooling.roi_pool(image, rois, self.roi_res, self.pooling_type)
        
        # pass warped images through classifier
        class_logits = self.backbone(warps)
        
        return class_logits
