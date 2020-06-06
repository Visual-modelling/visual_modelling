__author__ = "Jumperkables"

import torch
import torch.nn as nn
import torch.nn.functional as F

from tools.activations import sigmoid_256

class FCUp_Down3D(nn.Module):
    """
    __author__ = Jumperkables
    Fully convolutional Up down U net designed by Tom Winterbottom
    """
    #(args.UD_depth, args.in_no, args.out_no, args.UD_channel_factor
    def __init__(self, args):
        super().__init__()
        self.UDChain = Up_Down_Chain(args)


    def forward(self, x):
        x = self.UDChain(x)
        return(x)



class Up_Down_Chain(nn.Module):

    """
    __author__ = Jumperkables
    Chain of down sampling layers and upsampling layers
    """
    def __init__(self, args):
        """
        """
        super().__init__()
        # Get the first and last elements of the up and down chains
        if args.img_type == "RGB":
            start_channels = 3
        else:
            start_channels = 1
        down_chain = []
        up_chain = []
        down_chain.append( Down(start_channels, args.channel_factor, args) )
        up_chain.append( OutConv(args.channel_factor, start_channels, args) )

        # Fill in the rest given the depth
        for z in range(args.depth):
            down_chain.append( Down((2**(z))*args.channel_factor, (2**(z+1))*args.channel_factor, args) )
            up_chain.insert(0, Up((2**(z+1))*args.channel_factor, (2**(z))*args.channel_factor, args, z) ) # z is used to determine if output_padding is needed if the deconvolution would happen on an odd number


        layers = down_chain + up_chain
        self.layers = nn.ModuleList(layers)
        self.half = int(len(layers)/2)  # Len(layers) is always even


    def forward(self, x):
        """
        Residues will are saved to move forward into appropriate places on up chain
        """
        residues = []
        # Downward Pass
        x = self.layers[0](x.unsqueeze(1))
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

    def __init__(self, in_channels, out_channels, args):
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels, args)

    def forward(self, x):
        x = self.conv(x)
        return(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, args, depth, bilinear=False):
        super().__init__()
        if depth==0:
            if args.in_no%2 == 0:
                out_pad = 0
            else:
                out_pad = 1
        else:
            if((args.in_no/2**depth)%2 == 0):
                out_pad = 0
            else:
                out_pad = 1
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=(args.krnl_size_t, args.krnl_size, args.krnl_size), stride=2, padding=(args.padding_t, args.padding, args.padding), output_padding=(out_pad,1,1))


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
        #x = torch.cat([x2, x1], dim=1)
        x=x2+x1
        return(x)
        #return(self.conv(x1))



class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels, args, bilinear=False):
        super(OutConv, self).__init__()
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=(args.krnl_size_t, args.krnl_size, args.krnl_size), stride=2, padding=(args.padding_t, args.padding, args.padding), output_padding=((args.in_no%2)==0 ,1,1))

        self.conv = nn.Conv2d(args.in_no, args.out_no, kernel_size=1) # 2D convolution 1x1 convolution to change channels to args.out_no
        if args.img_type == "binary":
            self.img_activation = F.sigmoid
        if args.img_type == "greyscale":
            self.img_activation = sigmoid_256
        else:
            raise(Exception("Not yet implemented this image activation"))

    def forward(self, x):
        x = self.up(x)
        return self.img_activation(self.conv(x.squeeze(1)))


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, args):
        super().__init__()
        self.conv   =    nn.Conv3d(in_channels, out_channels, kernel_size=(args.krnl_size_t, args.krnl_size, args.krnl_size), stride=2, padding=(args.padding_t, args.padding, args.padding))
        self.conv2  =    nn.Conv3d(out_channels, out_channels, kernel_size=(args.krnl_size_t, args.krnl_size, args.krnl_size), stride=1, padding=(args.padding_t, args.padding, args.padding))
        #self.bn     =    nn.BatchNorm3d(out_channels)
        self.relu   =    nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.conv2(x)
        #x = self.bn(x)
        x = self.relu(x)
        return(x)
