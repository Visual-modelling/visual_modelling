__author__ = "Daniel Kluvanec"

import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleTransformer(nn.Module):
    """
    Pixel-wise image transformer that takes and outputs images of the shape (batch, channels, height, width).
    Takes a sequence of n images and returns m (<=n) images as predictions.
    Built as a simple transformer encoder
    """
    def __init__(self, args):
        """
        args.in_channels: int
        args.out_channels: int
        args.n_layers: int
        args.nhead: int = number of heads
        args.dim_feedforward: int = the dimension of the feedforward network model
        args.dropout: float
        """
        super().__init__()
        self.args = args
        transformer_encoder_layer = nn.TransformerEncoderLayer(args.in_channels, args.nhead, args.dim_feedforward, args.dropout)
        self.transformer_encoder = nn.TransformerEncoder(transformer_encoder_layer, args.n_layers, nn.BatchNorm1d)
    
    def forward(self, x:torch.Tensor):
        shape = x.shape  # should be (batch, sequence, height, width)
        batchsize = shape[0]
        sequence_length = shape[1]
        height = shape[2]
        width = shape[3]
        imsize = height * width

        x = x.view(batchsize, sequence_length, imsize)  # (batch, sequence, imsize)
        x = torch.transpose(x, 0, 1)  # (sequence, batch, imsize)

        x = self.transformer_encoder(x)

        x = torch.view(sequence_length, batchsize, height, width)
        x = torch.transpose(x, 0, 1)  # (batch, sequence, height, width)
        x = x[:, :-self.args.out_channels, :, :]  # (batch, out_channels, height, width)

        return x

