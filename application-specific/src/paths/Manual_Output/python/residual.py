import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        device =torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    def forward(self, x):
        identity = x
        
        x += identity

        return x