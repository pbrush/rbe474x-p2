# STANDARD
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

# CUSTOM

# Will give deltas and boxes
class RPN(nn.Module):
    def __init__(self, in_channels, out_channels, n_anchors):
        super(RPN, self).__init__()
        self.shared_conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.logits = nn.Conv2d(out_channels, n_anchors, kernel_size=1)
        self.bbox_deltas = nn.Conv2d(out_channels, n_anchors * 4, kernel_size=1)
    
    def forward(self, x):
        x = F.relu(self.shared_conv(x)) # Don't forget ReLU
        logits = self.logits(x)
        deltas = self.bbox_deltas(x)
        return logits, deltas

# Object detection and Classification
class ROIBoxHead(nn.Module):
    def __init__(self, in_channels, n_classes):
        super(ROIBoxHead, self).__init__()
        self.fc1 = nn.Linear(in_channels * 7 * 7, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.class_scores = nn.Linear(1024, n_classes)
        self.bbox_delta_scores = nn.Linear(1024, n_classes * 4)

    def forward(self, x):
        x = x.flatten()
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        class_scores = self.class_scores(x)
        bbox_delta_scores = self.bbox_delta_scores(x)
        return class_scores, bbox_delta_scores

# Mask segmentation
class MaskHead(nn.Module):
    def __init__(self, in_channels, n_classes):
        super(MaskHead, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 256, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.deconv = nn.ConvTranspose2d(256, 256, 2, padding=2)
        self.mask_predictor = nn.Conv2d(256, n_classes, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.deconv(x))
        masks = self.mask_predictor(x)
        return masks
    
@dataclass
class RPN_params:
    in_channels: int
    out_channels: int
    n_anchors: int

@dataclass
class ROIBox_head_params:
    in_channels: int
    n_classes: int

@dataclass
class Mask_Head_params:
    in_channels: int
    n_classes: int

@dataclass
class MaskRCNN_params:
    rpn: RPN_params
    roibox: ROIBox_head_params
    mask: Mask_Head_params