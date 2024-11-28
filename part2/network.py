
import torch
import torch.nn as nn
import torch.nn.functional as F



class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.flatten = nn.Flatten()
        self.Relu_Stack = nn.Sequential(
            
        )
        
    # define your network here!

    def forward(self, x):

        x = self.flatten(x)
        y = x
        
        return y
    

