import torch
from torch import nn

class ResNet_2D_Encoder:
    
    def __init__(self):
        self.branch1 = nn.Sequential(
            nn.Conv2d(),
            nn.BatchNorm2d(),
            nn.ReLU(),
            nn.Conv2d(),
            nn.Dropout()
        )
        self.branch2 = nn.Conv2d()
        self.final_ReLU = nn.ReLU()

    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x_sum = torch.sum(x1, x2)
        y = self.final_ReLU(x_sum)
        return y
    
class ResNet_2D_Decoder:
    
    def __init__(self):
        self.branch1 = nn.Sequential(
            nn.ConvTranspose2d(),
            nn.BatchNorm2d(),
            nn.ReLU(),
            nn.ConvTranspose2d(),
            nn.Dropout()
        )
        self.branch2 = nn.ConvTranspose2d()
        self.final_ReLU = nn.ReLU()
    
    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x_sum = torch.sum(x1, x2)
        y = self.final_ReLU(x_sum)
        return y