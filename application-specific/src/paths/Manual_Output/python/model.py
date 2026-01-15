# import the necessary packages
import torch
from torch import nn
from python.residual import ResidualBlock

class CNN(nn.Module):
    def __init__(self):     
        device =torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
 
        super(CNN,self).__init__()
        self.conv2d_1 = nn.Conv2d( 
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=1,
            bias=True,
            padding_mode='zeros',
            device=device,
            dtype=torch.float32,
            in_channels=3,
        )
        
        
        self.maxpool2d_1 = nn.MaxPool2d( 
            kernel_size=2,
            stride=2,
            padding=0,
            return_indices=False,
            ceil_mode=False,
        )
        
        

    def forward(self, x):
        x = self.conv2d_1(x)
        x = self.maxpool2d_1(x)
        
        return x
