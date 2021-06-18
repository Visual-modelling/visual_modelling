__author__ = "Daniel Kluvanec"

import torch
from torch._C import device
import torch.nn as nn
import torch.nn.functional as F

from models.UpDown2D import sigmoid_256

class PixelTransformer(nn.Module):
    """
    Pixel-wise image transformer that takes and outputs images of the shape (batch, channels, height, width).
    Takes a sequence of n images and returns m (<=n) images as predictions.
    Built as a simple transformer encoder
    """
    def __init__(self, args):
        """
        args.in_no: int = numer of input channels
        args.out_no: int = number of output channels
        args.d_model: int = number of features in the input (flattened image dimensions)
        args.n_layers: int
        args.nhead: int = number of heads
        args.dim_feedforward: int = the dimension of the feedforward network model
        args.dropout: float
        args.pixel_regression_layer: bool = whether to add a pixel regression linear layer pair at the end
        args.norm_layer: str = which normalisation layer to use, one of ("layer_norm", "batch_norm")
        """
        super().__init__()
        self.args = args
        
        # norm layer
        if args.norm_layer == 'layer_norm':
            self.norm_layer_class = nn.LayerNorm
        elif args.norm_layer == 'batch_norm':
            self.norm_layer_class = nn.BatchNorm1d
        else:
            raise ValueError(f"Unknown norm_layer: {args.norm_layer}")

        # positional encodings
        positions = torch.arange(args.in_no)[:, None]
        positional_i = torch.arange(args.d_model)[None, :]
        positional_angle_rates = 1 / torch.pow(10000, (2* (positional_i//2)) / args.d_model)
        positional_angle_rads = positions * positional_angle_rates
        positional_angle_rads[:, 0::2] = torch.sin(positional_angle_rads[:, 0::2])
        positional_angle_rads[:, 1::2] = torch.cos(positional_angle_rads[:, 1::2])
        self.register_buffer('pos_encoding', positional_angle_rads[:, None, :])
        self.pos_encoding.requires_grad = False

        # transformer
        transformer_encoder_layer = nn.TransformerEncoderLayer(args.d_model, args.nhead, args.dim_feedforward, args.dropout)
        self.transformer_encoder = nn.TransformerEncoder(transformer_encoder_layer, args.n_layers, self.norm_layer_class(args.d_model))

        # pixel regression layer
        if args.pixel_regression_layer:
            self.pixel_regression_layer = nn.Sequential(
                nn.Linear(args.d_model, args.dim_feedforward),
                nn.ReLU(),
                nn.Dropout(args.dropout),
                nn.Linear(args.dim_feedforward, args.d_model),
            )
        else:
            self.pixel_regression_layer = nn.Identity()

    def forward(self, x:torch.Tensor):
        shape = x.shape  # should be (batch, sequence, height, width)
        batchsize = shape[0]
        sequence_length = shape[1]
        height = shape[2]
        width = shape[3]
        imsize = height * width

        # reshape
        x = x.view(batchsize, sequence_length, imsize)  # (batch, sequence, imsize)
        x = torch.transpose(x, 0, 1)  # (sequence, batch, imsize)

        # layers
        x = x + self.pos_encoding
        x = self.transformer_encoder(x)
        x = self.pixel_regression_layer(x)  # identity if not args.pixel_regression_layer
        x = sigmoid_256(x)

        # reshape
        x = torch.transpose(x, 0, 1)  # (batch, sequence, imsize)
        x = x.view(batchsize, sequence_length, height, width)
        
        # extract output
        x = x[:, -self.args.out_no:, :, :]  # (batch, out_no, height, width)

        return x
