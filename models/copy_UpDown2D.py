__author__ = "Jumperkables"

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..tools.activations import 256_sigmoid

class FCUp_Down(nn.Module):
    """
    __author__ = Jumperkables
    Fully convolutional Up down U net designed by Tom Winterbottom
    """

    def __init__(self, depth, start_channels, end_channels, args, scale_factor=16):
        super().__init__()
        self.UDChain = Up_Down_Chain(depth, start_channels, end_channels, args)
        self.krnl_0 = args.krnl_0
        self.krnl_1 = args.krnl_1


    def forward(self, x):
        x = self.UDChain(x)
        return(x)



class Up_Down_Chain(nn.Module):
    """
    __author__ = Jumperkables
    Chain of down sampling layers and upsampling layers
    """
    def __init__(self, depth, start_channels, end_channels, scale_factor=16):
        """
        start_channels  = number of input channels for initial conv layer
        end_channels    = number of output channels
        scale_factor    = multiple of channels that doubles, e.g.: in_channels, 16, 32, 64, ...., 64, 32, out_channels
        depth = depth of the UNet: 
            0: gives only in/out layer  (2 total layers)
            1: (4 total layers)
            2: (6 total layers)
        """
        super().__init__()
        # Get the first and last elements of the up and down chains
        down_chain = []
        up_chain = []

        down_chain.append( Down(start_channels, scale_factor) )
        up_chain.append( OutConv(scale_factor, end_channels) )
        
        # Fill in the rest given the depth
        for x in range(depth):
            down_chain.append( Down(scale_factor*(x+1), scale_factor*(x+2)) )
            up_chain.insert(0, Up(2*scale_factor*(x+2), scale_factor*(x+1)) ) # Prepend

        layers = down_chain + up_chain
        self.layers = nn.ModuleList(layers)
        self.half = int(len(layers)/2)  # Len(layers) is always even


    def forward(self, x):
        """
        Residues will are saved to move forward into appropriate places on up chain
        """
        residues = []
        # Downward Pass
        x = self.layers[0](x)
        for layer in self.layers[1:self.half]:
            x = layer(x)
            residues.insert(0, x)
        
        # Upward Pass
        for idx, layer in enumerate(self.layers[self.half:(len(self.layers)-1)]):
            x = layer(x, residues[idx])
        x = self.layers[-1](x)

        return(x)

            
"""
Following classes are adapted from: https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_parts.py
"""
class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=self.krnl_0, stride=2)

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)



class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        
    def forward(self, x):
        if self.img_type == "binary":
            return F.sigmoid(self.conv(x))
        if self.img_type == "greyscale":
            return 256_sigmoid(self.conv(x)) 
        else:
            raise(Exception("Not yet implemented error"))

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=self.krnl_1, padding=self.padding1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=self.krnl_1, padding=self.padding1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)
