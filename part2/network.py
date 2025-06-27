# STANDARD
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_align

# CUSTOM
from resnet import *
from network_utils import *

class Network(nn.Module):
    def __init__(self, params: MaskRCNN_params):
        super(Network, self).__init__()
        self.RPN = RPN(params.rpn.in_channels, 
                       params.rpn.out_channels,
                       params.rpn.n_anchors)
        self.ROIBoxHead = ROIBoxHead(params.roibox.in_channels,
                                     params.roibox.n_classes)
        self.MaskHead = MaskHead(params.roibox.in_channels,
                                 params.roibox.n_classes)
        
        
    # define your network here!

    def forward(self, x):

        
        
        return x
    

