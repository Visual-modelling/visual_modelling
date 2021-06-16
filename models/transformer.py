__author__ = "DeanSlack, Daniel Kluvanec"

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self, k, heads=1, masked=False):
        super().__init__()
        self.k, self.heads = k, heads
        # set masked self-attention true/false
        self.masked = masked

        # initialise query, key, and value layer for all heads (as single concatenated tensor)
        self.tokeys = nn.Linear(k, k * heads, bias=False)
        self.toqueries = nn.Linear(k, k * heads, bias=False)
        self.tovalues = nn.Linear(k, k * heads, bias=False)

        # unify outputs of attention heads into k dim tensor
        self.unifyheads = nn.Sequential(nn.Linear(heads * k, k))

    def forward(self, x):
        b, t, k = x.size()

        h = self.heads
        queries = self.toqueries(x).view(b, t, h, k)
        keys = self.tokeys(x).view(b, t, h, k)
        values = self.tovalues(x).view(b, t, h, k)

        # fold heads into the batch dimension
        keys = keys.transpose(1, 2).contiguous().view(b * h, t, k)
        queries = queries.transpose(1, 2).contiguous().view(b * h, t, k)
        values = values.transpose(1, 2).contiguous().view(b * h, t, k)

        queries = queries / (k ** (1/4))
        keys = keys / (k ** (1/4))

        # get dot product of queries and keys, and scale
        dot = torch.bmm(queries, keys.transpose(1, 2))
        # dot has size (b*h, t, t) containing raw weights

        if self.masked:
            indices = torch.triu_indices(t, t, offset=1)
            dot[:, indices[0], indices[1]] = float('-inf')

        dot = F.softmax(dot, dim=2)
        # dot now contains row-wise normalized weights

        # apply the self attention to the values
        out = torch.bmm(dot, values).view(b, h, t, k)

        # swap h, t back, unify heads
        out = out.transpose(1, 2).contiguous().view(b, t, h * k)

        return self.unifyheads(out)


class TransformerBlock(nn.Module):
  def __init__(self, in_dim, out_dim, heads, ff=True, residual=False, norm=False, masked=False):
    super().__init__()

    self.in_dim = in_dim
    self.out_dim = out_dim
    self.ff, self.residual, self.norm = ff, residual, norm

    # initialise attenion layer
    self.attention = SelfAttention(self.in_dim, heads=heads, masked=masked)

    if self.norm == True:
        self.norm = nn.LayerNorm(self.in_dim)
        self.norm2 = nn.LayerNorm(self.out_dim)

    if self.ff == True:
        self.ff1 = nn.Sequential(
        nn.Linear(self.in_dim, self.out_dim),
        nn.ReLU(),
        # nn.Dropout(0.1),
        nn.Linear(self.in_dim, self.out_dim),
        #   nn.Dropout(0.1)
        )

  def forward(self, x):

    if self.residual == True:
        x = x + self.attention(x)
    else:
        x = self.attention(x)

    if self.norm == True:
        x = self.norm1(x)

    if self.ff == True:
        if self.residual == True and self.in_dim == self.out_dim:
            x = x + self.ff1(x)
        else:
            x = self.ff1(x)

        if self.norm == True:
            x = self.norm2(x)

    return x


class VMDecoder(nn.Module):
    """
    Transformer decoder model for next frame prediction. Positional encoding is inferred from masked self-attention, rather than an explicit positional embedding.
    """
    def __init__(self, **kwargs):
        super().__init__()
        
        self.in_dim = 4096
        self.layers = 1
        self.heads = 1
        self.ff = True
        self.residual = True
        self.norm = True
        self.masked = True

        # gather allowed attributes and update from kwargs
        attrs = list(self.__dict__.keys())
        self.__dict__.update((k, v) for k, v in kwargs.items() if k in attrs and v is not None)

        # initialise decoder blocks
        self.tblocks = [TransformerBlock(in_dim=self.in_dim, out_dim=self.in_dim, heads=self.heads, ff=self.ff,
                                         residual=self.residual, norm=self.norm, masked=True)
                        for i in range(self.layers)]

        self.tblocks = nn.Sequential(*self.tblocks)

        # pixel regression layer
        self.to_pixels = nn.Sequential(
            nn.Linear(self.in_dim, self.in_dim * 4),
            nn.Linear(self.in_dim * 4, self.in_dim),
            nn.Sigmoid()
        )


    def forward(self, x):
        # get input batch dimensions
        b, t, k = x.size()

        # send input to transformer block
        x = self.tblocks(x)

        # select last in sequence hidden state
        x = x[:, -1, :]

        # x = x.view(b, x.size(1) * x.size(2))
        x = self.to_pixels(x)

        # fold onto batch dimension
        x = x.view(b, k)

        # send each hidden state to image regression layer
        x = self.to_pixels(x)

        return x