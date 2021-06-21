__author__ = "Daniel Kluvanec"

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.UpDown2D import sigmoid_256

def hardsigmoid_256(inputs):
    """
    Scale the hardsigmoid function from 0-255 for greyscale image outputs
    """
    return 255 * F.hardsigmoid(inputs)

def identity(inputs):
    return inputs


class PositionalEncoder(nn.Module):
    """ Class that saves and returns the positional encodings"""
    def __init__(self, d_model, max_sequence_len):
        """
        :param d_model: The number of features in each embedding
        :param max_sequence_len: How many embeddings to store
        """
        super().__init__()
        # positional encodings
        positions = torch.arange(max_sequence_len)[:, None]
        positional_i = torch.arange(d_model)[None, :]
        positional_angle_rates = 1 / torch.pow(10000, (2 * (positional_i // 2)) / d_model)
        positional_angle_rads = positions * positional_angle_rates
        positional_angle_rads[:, 0::2] = torch.sin(positional_angle_rads[:, 0::2])
        positional_angle_rads[:, 1::2] = torch.cos(positional_angle_rads[:, 1::2])

        self.register_buffer('pos_encoding', positional_angle_rads[:, None, :])
        self.pos_encoding.requires_grad = False

    def forward(self):
        return self.pos_encoding


class TransformerEncoder(nn.Module):
    """
    Transformer that returns hidden activations in forward function as well
    """
    def __init__(self, n_layers, d_model, nhead, dim_feedforward, dropout):
        super().__init__()
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout) for _ in range(n_layers)
        ])

    def forward(self, x):
        hidden_xs = []
        for transformer_layer in self.transformer_layers:
            x = transformer_layer(x)
            hidden_xs.append(x)
        # hidden_x includes output
        return x, hidden_xs


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
        args.pixel_regression_layers: int = number of layers at the end of transformer
        args.norm_layer: str = which normalisation layer to use, one of ("layer_norm", "batch_norm")
        args.output_activation: str = what activation function to use at the end of the network (linear, hardsigmoid-256, sigmoid-256)
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

        # output_activation
        if args.output_activation == 'linear':
            self.output_activation_function = identity
        elif args.output_activation == 'hardsigmoid-256':
            self.output_activation_function = hardsigmoid_256
        elif args.output_activation == 'sigmoid-256':
            self.output_activation_function = sigmoid_256
        else:
            raise ValueError(f"Unknown output_activation: {args.output_activation}")

        # positional encoder
        self.pos_encoder = PositionalEncoder(args.d_model, args.in_no)

        # transformer
        self.transformer = TransformerEncoder(args.n_layers, args.d_model, args.nhead, args.dim_feedforward, args.dropout)

        # pixel regression layers
        pixel_regression_layers = []
        for i in range(args.pixel_regression_layers):
            pixel_regression_layers.append(nn.ReLU())
            pixel_regression_layers.append(nn.Dropout(args.dropout))
            pixel_regression_layers.append(nn.Linear(args.d_model, args.d_model))

        self.pixel_regression_layer = nn.Sequential(*pixel_regression_layers)

    def forward(self, x):
        shape = x.shape  # should be (batch, sequence, height, width)
        batchsize = shape[0]
        sequence_length = shape[1]
        height = shape[2]
        width = shape[3]
        imsize = height * width

        # reshape
        x = torch.transpose(x, 0, 1)  # (sequence, batch, height, width)
        x = x.view(sequence_length, batchsize, imsize)  # (batch, sequence, imsize)

        # layers
        x = x + self.pos_encoder()
        x, hidden_xs = self.transformer_encoder(x)
        x = x[-self.args.out_no:, :, :, :]  # (out_no, batch, height, width)
        x = self.pixel_regression_layer(x)
        x = self.output_activation_function(x)

        # reshape
        x = x.view(self.args.out_no, batchsize, height, width)
        x = torch.transpose(x, 0, 1)  # (out_no, sequence, imsize)
        return x, hidden_xs
