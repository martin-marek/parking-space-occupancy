from torch import nn
from torchvision.models import resnet50
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops.misc import FrozenBatchNorm2d

from .utils import pooling
from .utils.gcn import GCN


class RCNN_GCN(nn.Module):
    """
    An R-CNN inspired parking lot classifier
    augmented with a GCN before the classification head.
    Pools ROIs directly from image and separately
    passes each pooled ROI through a CNN. The GCN then
    attends to each parking space and its n nearest neighbors.
    """
    def __init__(self, roi_res=100, gcn_layers=2, pooling_type='square'):
        super().__init__()
        
        # load pretrained backbone
        self.roi_res = roi_res
        backbone = resnet50(pretrained=True, norm_layer=FrozenBatchNorm2d)
        del backbone.fc
        self.backbone = IntermediateLayerGetter(backbone, return_layers={'avgpool': 'avgpool'})
        
        # freeze bottom layers
        layers_to_train = ['layer4', 'layer3', 'layer2']
        for name, parameter in self.backbone.named_parameters():
            if all([not name.startswith(layer) for layer in layers_to_train]):
                parameter.requires_grad_(False)

        # ROI pooling
        self.roi_res = roi_res
        self.pooling_type = pooling_type
        
        # create attention module
        hidden_dim = 2048
        self.gcn = GCN(num_layers=gcn_layers, d_model=hidden_dim)
        
        # classification head
        self.class_head = nn.Linear(in_features=hidden_dim, out_features=2)
        
    def forward(self, image, rois):
        # pool ROIs from image
        warps = pooling.roi_pool(image, rois, self.roi_res, self.pooling_type)
        
        # pass warped images through backbone
        features = self.backbone(warps)['avgpool'] # [L, C, 1, 1]
        features = features.flatten(1) # [L, C]
        
        # pass backbone features through the attention module
        features = self.gcn(features, rois)
        
        # create class logits
        class_logits = self.class_head(features)
        
        return class_logits
