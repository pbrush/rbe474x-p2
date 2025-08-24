# STANDARD
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_align
from torchvision.models import resnet50, ResNet50_Weights

# CUSTOM
from resnet import *
from network_utils import *

class Network(nn.Module):
    def __init__(self, params: MaskRCNN_params):
        super(Network, self).__init__()
        self.Backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

        self.RPN = RPN(params.rpn.in_channels, 
                       params.rpn.out_channels,
                       params.rpn.n_anchors)
        self.ROIBox_Head = ROIBoxHead(params.roibox.in_channels,
                                     params.roibox.n_classes)
        self.Mask_Head = MaskHead(params.roibox.in_channels,
                                 params.roibox.n_classes)
        
        

    def forward(self, x):

        # Backbone
        

        # RPN
        rpn_logits, rpn_deltas = self.RPN.forward(x)
        rpn_activations = F.relu(rpn_logits)
        
        # ROI Pooling
        rois_scores, rois_deltas = self.ROIBox_Head.forward(x)

        # Mask
        masks = self.Mask_Head.forward(x)
        return rpn_logits, rpn_deltas, rois_scores, rois_deltas, masks
    

