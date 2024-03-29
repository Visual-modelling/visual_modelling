__author__ = "Jumperkables"
import copy
import os, sys

import torch
import torch.nn as nn
import torch.nn.functional as F

def sigmoid_256(inputs):
    """
    Scale the sigmoid function from 0-255 for greyscale image outputs
    """
    return (255*torch.sigmoid(inputs))


class FCUpDown2D(nn.Module):
    """
    __author__ = Jumperkables
    Fully convolutional Up down U net designed by Tom Winterbottom
    """
    #(args.UD_depth, args.in_no, args.out_no, args.UD_channel_factor
    def __init__(self, args):
        super().__init__()
        self.UDChain = UpDownChain(args)


    def forward(self, x):
        x, probe_ret = self.UDChain(x)
        return x, probe_ret



class UpDownChain(nn.Module):

    """
    __author__ = Jumperkables
    Chain of down sampling layers and upsampling layers
    """
    def __init__(self, args):
        """
        args.in_no  = number of input channels for initial conv layer
        args.out_no = number of output channels
        args.channel_factor = multiple of channels that doubles, e.g.: in_channels, 16, 32, 64, ...., 64, 32, out_channels
        args.depth = depth of the UNet: 
            0: gives only in/out layer  (2 total layers)
            1: (4 total layers)
            2: (6 total layers)
        """
        super().__init__()
        # Get the first and last elements of the up and down chains
        down_chain = []
        up_chain = []
        # TODO Deprecated with removal of model_in_no???
        #if args.model_in_no != False:
        #    args.in_no = args.model_in_no
        down_chain.append( Down(args.in_no, args.channel_factor, args) )
        up_chain.append( OutConv(args.channel_factor, args.out_no, args) )

        # Fill in the rest given the depth
        for z in range(args.depth):
            down_chain.append( Down((2**(z))*args.channel_factor, (2**(z+1))*args.channel_factor, args) )
            up_chain.insert(0, Up(2*(2**(z+1))*args.channel_factor, (2**(z))*args.channel_factor, args) ) # Prepend

        layers = down_chain + up_chain
        self.layers = nn.ModuleList(layers)
        self.half = int(len(layers)/2)  # Len(layers) is always even


    def forward(self, x):
        """
        Residues will are saved to move forward into appropriate places on up chain
        """
        residues = []
        probe_ret = []
        # Downward Pass
        x = self.layers[0](x)
        probe_ret.append(x)
        for layer in self.layers[1:self.half]:
            x = layer(x)
            probe_ret.append(x)
            residues.insert(0, x)

        # Upward Pass
        for idx, layer in enumerate(self.layers[self.half:(len(self.layers)-1)]):
            x = layer(x, residues[idx])
            probe_ret.append(x)
        if self.half != len(self.layers):
            x = self.layers[-1](x)
            probe_ret.append(x)

        return(x, probe_ret)


"""
Following classes are adapted from: https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_parts.py
"""
class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, args):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, args)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, args, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels, args)

    def forward(self, x1, x2):
        
        x1 = self.up(x1)
        x2 = self.up(x2)
        # input is CHW
        #diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        #diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        #x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
        #                diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        #x=x2+x1
        return self.conv(x)
        #return(self.conv(x1))



class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels, args, bilinear=True):
        super(OutConv, self).__init__()
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        if args.img_type == "binary":
            self.img_activation = torch.sigmoid
        if args.img_type == "greyscale":
            self.img_activation = torch.sigmoid #sigmoid_256
        else:
            raise(Exception("Not yet implemented this image activation"))

    def forward(self, x):
        x = self.up(x)
        return self.img_activation(self.conv(x))


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, args):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=args.krnl_size, padding=args.padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=args.krnl_size, padding=args.padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.in_no = 5
    args.out_no = 1
    args.img_type = "greyscale"
    args.krnl_size = 3
    args.padding = 1
    args.depth = 5
    args.channel_factor = 64
    model = FCUpDown2D(args)
    import ipdb; ipdb.set_trace()
    inputs = torch.ones(16,5,64,64)
    outputs = model(inputs)
    print("done")
