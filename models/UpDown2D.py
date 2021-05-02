__author__ = "Jumperkables"
import copy
import os, sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from tools.activations import sigmoid_256


class FCUp_Down2D_2_FCPred(nn.Module):
    """
    __author__ = Jumperkables
    Class that adapts the FCUp_Down2D model to the gravity prediction task. Damn i wish pytorch lightning was around when i started coding this
    """
    def __init__(self, args, load_path="", load_mode="append_classification", mode="grav_pred"):
        super().__init__()
        """
        load_mode:
            'append_classification' : Add an extra classification layer at the end
        mode:
            'grav_pred' or 'bounces_pred', decides if the FC output at the end has 3 or 1 output dim respectively
        """
        self.load_mode = load_mode
        self.duplicate = False
        self.args = args
        assert load_mode in ["append_classification"], f"Invalid load_mode:{load_mode}"
        assert args.out_no == 1, f"Gravity prediction requires a single output image to turn into classification"
        cargs = copy.deepcopy(args)
        if args.model_in_no != False:
            cargs.in_no = args.model_in_no
            self.duplicate = cargs.in_no
        if load_path is not "":
            # Pretrained model might have more than 1 input or output. We will have to replace the input and output layers with those of appropriate shape
            model_weights = torch.load(os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), load_path))
            #in_no = model_weights['UDChain.layers.0.maxpool_conv.1.double_conv.0.weight'].shape[1]
            #cargs = copy.deepcopy(args)
            #cargs.in_no = in_no
            #self.in_no = in_no

            full = FCUp_Down2D(cargs)
            full.load_state_dict(torch.load(os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), load_path)), strict=False)
        else:
            full = FCUp_Down2D(cargs)
        # Replace the first layer
        if mode == "grav_pred":
            pred_out_dim = 3    # gx, gy, gz
        elif mode == "bounces_pred":
            pred_out_dim = 1    # n bounces
        if load_mode == "append_classification":
            self.classifier_fc = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(64*64, 500),
                nn.BatchNorm1d(500),
                nn.GELU(),
                nn.Dropout(0.3),
                nn.Linear(500, pred_out_dim)   # For predicting x,y,z gravity
            )
        self.full = full 

    def forward(self, x):
        bsz = x.shape[0]
        if self.load_mode == "pad":
            if self.duplicate != False:
                x = x.repeat(1, self.duplicate, 1, 1)
            pass
        x = self.full(x)
        #import ipdb; ipdb.set_trace()#TODO Finish handling this grav_pred
        x = x.view(bsz, -1)
        x = self.classifier_fc(x)
        if self.args.grav_pred:
            x = x.tanh()
        if self.args.bounces_pred:
            x = F.relu(x)
        return x



class FCUp_Down2D_2_Segmentation(nn.Module):
    """
    __author__ = Jumperkables
    Class that adapts the FCUp_Down2D model to an appropriate segmentation task
    """
    def __init__(self, args, load_path="", load_mode="pad"):
        super().__init__()
        """
        load_mode:
            'replace'   : Replace the initial convolution layer with 'in_no' input channels with another a single channel
            'pad'       : Keep original 'in_no' input channels and pad the inputs by copying the input 'in_no' times
        """
        self.load_mode = load_mode
        self.duplicate = False
        assert load_mode in ["replace", "pad"], f"Invalid load_mode:{load_mode}"
        assert args.out_no == 1, f"Segmentation adaption not implemented on pretrained models that output more than 1 image. Arguements imply the model outputs {args.out_no} images"
        cargs = copy.deepcopy(args)
        if args.model_in_no != False:
            cargs.in_no = args.model_in_no
            self.duplicate = cargs.in_no
        if load_path is not "":
            # Pretrained model might have more than 1 input or output. We will have to replace the input and output layers with those of appropriate shape
            model_weights = torch.load(os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), load_path))
            #in_no = model_weights['UDChain.layers.0.maxpool_conv.1.double_conv.0.weight'].shape[1]
            #cargs = copy.deepcopy(args)
            #cargs.in_no = in_no
            #self.in_no = in_no

            full = FCUp_Down2D(cargs)
            full.load_state_dict(torch.load(os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), load_path)), strict=False)
        else:
            full = FCUp_Down2D(cargs)
        # Replace the first layer
        if load_mode == "replace":
            full.UDChain.layers[0].maxpool_conv[1].double_conv[0]= nn.Conv2d(1,args.channel_factor, kernel_size=args.krnl_size, padding=args.padding)
        elif load_mode == "pad":
            pass
        self.full = full 

    def forward(self, x):
        if self.load_mode == "pad":
            if self.duplicate != False:
                x = x.repeat(1, self.duplicate, 1, 1)
            pass
        x = self.full(x)
        return x


class FCUp_Down2D_2_MNIST(nn.Module):
    """
    __author__ = Jumperkables
    Class that adapts the FCUp_Down2D model to an appropriate MNIST adaption
    """
    def __init__(self, args, load_path="", load_mode="pad"):
        super().__init__()
        """
        load_mode:
            'replace'   : Replace the initial convolution layer with 'in_no' input channels with another a single channel. Furthermore, model in half and add a classification MLP
            'pad'       : Keep original 'in_no' input channels and pad the inputs by copying the input 'in_no' times
        """
        self.load_mode = load_mode
        self.duplicate = False
        assert load_mode in ["replace", "pad"], f"Invalid load_mode:{load_mode}"
        #model_weights = torch.load(os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), load_path))
        #in_no = model_weights['UDChain.layers.0.maxpool_conv.1.double_conv.0.weight'].shape[1]
        cargs = copy.deepcopy(args)
        if args.model_in_no != False:
            cargs.in_no = args.model_in_no
            self.duplicate = cargs.in_no

        full = FCUp_Down2D(cargs)
        if load_path is not "":
            full.load_state_dict(torch.load(os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), load_path)), strict=False)
        # Replace the first layer
        if load_mode == "replace":
            full.UDChain.layers[0].maxpool_conv[1].double_conv[0]= nn.Conv2d(1,args.channel_factor, kernel_size=args.krnl_size, padding=args.padding)
            # Remove the up layers
            full.UDChain.layers = full.UDChain.layers[:int(len(full.UDChain.layers)/2)]

            # The front half of the network intact
            self.front = full

            # Create the linear layers to classification
            fc_0_len = full(torch.ones(1,1,64,64)).view(-1).shape[0]
            self.fc_0 = nn.Sequential(
                nn.Linear(fc_0_len,500),
                nn.ReLU(inplace=True)
            )
            self.fc_1 = nn.Sequential(
                nn.Linear(500,10),
                nn.ReLU(inplace=True)
            )
        elif load_mode == "pad":
            self.full = full

    def forward(self, x):
        #import ipdb; ipdb.set_trace()
        if self.load_mode == "replace":
            x = self.front(x)
            bsz = x.shape[0]
            x = x.view(bsz, -1)
            x = self.fc_0(x)
            x = self.fc_1(x)
        elif self.load_mode == "pad":
            if self.duplicate != False:
                x = x.repeat(1, self.duplicate, 1, 1)
            x = self.full(x)
        return x



class FCUp_Down2D(nn.Module):
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
        #import ipdb; ipdb.set_trace()
        # Downward Pass
        x = self.layers[0](x)
        for layer in self.layers[1:self.half]:
            x = layer(x)
            residues.insert(0, x)

        # Upward Pass
        for idx, layer in enumerate(self.layers[self.half:(len(self.layers)-1)]):
            x = layer(x, residues[idx])
        if self.half != len(self.layers):
            x = self.layers[-1](x)

        return(x)


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
            self.img_activation = F.sigmoid
        if args.img_type == "greyscale":
            self.img_activation = sigmoid_256
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
