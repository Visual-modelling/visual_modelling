
import torch
import torch.nn as nn
import torch.nn.functional as F
from tools.activations import sigmoid_256

def load_model(model_name, config):
    """ Model Loader for all the models contained within this file.
        'model_name': str
        'config': list
    """
    model = None
    config = [eval(x) for x in config]

    if model_name == 'VMTransformer':
        model = VMTransformer(*config)
    if model_name == 'VMTransformer2':
        model = VMTransformer2(*config)

    return model


class SelfAttention(nn.Module):
    def __init__(self, k, heads=1):
        super().__init__()
        self.k, self.heads = k, heads
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

        dot = F.softmax(dot, dim=2)
        # dot now contains row-wise normalized weights

        # apply the self attention to the values
        out = torch.bmm(dot, values).view(b, h, t, k)

        # swap h, t back, unify heads
        out = out.transpose(1, 2).contiguous().view(b, t, h * k)

        return self.unifyheads(out)


class TransformerBlock(nn.Module):
  def __init__(self, k, heads, ff=False, residual=False, norm=True):
    super().__init__()

    self.ff, self.residual, self.norm = ff, residual, norm

    # initialise attenion layer
    self.attention = SelfAttention(k, heads=heads)

    self.norm1 = nn.LayerNorm(k)
    self.norm2 = nn.LayerNorm(k)

    self.ff1 = nn.Sequential(
      nn.Linear(k, k * 4),
      nn.ReLU(),
      nn.Linear(k * 4, k)
    )

  def forward(self, x):

    if self.residual:
        x = x + self.attention(x)
    else:
        x = self.attention(x)

    if self.norm:
        x = self.norm1(x)

    if self.ff is True:
        if self.residual:
            x = x + self.ff1(x)
        else:
            x = self.ff1(x)

    if self.norm:
        x = self.norm2(x)

    return x


class VMTransformer(nn.Module):
    """
    Simplest transformer encoder model for next image prediction. No positional encoding - position
    of each image in time is implicitly inferred using the last image representation query as input
    to the classification layer.
    """
    def __init__(self, k=4096, heads=1, depth=1, ff=True, residual=True, norm=True):
        super().__init__()

        # self attention layers
        self.tblocks = []
        for _ in range(depth):
            self.tblocks.append(TransformerBlock(k=k, heads=heads, ff=ff,
                                                 residual=residual, norm=norm))
        self.tblocks = nn.Sequential(*self.tblocks)

        # classification / pixel regression block
        self.to_pixels = nn.Sequential(
            nn.Linear(k, k),
            nn.Linear(k, k),
            nn.Linear(k, k)#,
            # rectify output to between 0 and 1
            #nn.Hardtanh(0, 1)
        )

    def forward(self, x):
        # get input batch dimensions
        b, t, k = x.size()

        # send input to transformer block
        x = self.tblocks(x)
        # use image representation from last image query
        x = x[:, -1, :]

        # send last image representation to pixel regression block
        x = self.to_pixels(x)

        #######
        # To work with grayscale images
        x = sigmoid_256(x) 
        #x = 255*x
        #######

        return x











class VMTransformer2(nn.Module):
    """
    Includes encoding of image to higher dimensional representation.

    """
    def __init__(self, k=4096, seq_len=5, heads=1, depth=1, ff=True, residual=True, norm=True):
        super().__init__()

        self.seq_len = seq_len
        self.emb_dim = 8192

        self.img_emb = nn.Sequential(
            # trial with non-linearity
            nn.Linear(k, self.emb_dim)
        )

        # self attention layers
        self.tblocks = []
        for _ in range(depth):
            self.tblocks.append(TransformerBlock(k=self.emb_dim, heads=heads, ff=ff,
                                                 residual=residual, norm=norm))
        self.tblocks = nn.Sequential(*self.tblocks)

        # classification / pixel regression layer
        self.to_pixels = nn.Sequential(
            nn.Linear(self.emb_dim, k),
            nn.Linear(k, k),
            nn.Linear(k, k),
            # rectify output to between 0 and 1
            nn.Hardtanh(0, 1)
        )

    def forward(self, x):
        # get input batch dimensions
        b, t, k = x.size()

        # project image tensor to parameterised image embedding space
        x = self.img_emb(x)

        # send input to transformer block
        x = self.tblocks(x)
        # use image representation from last image query
        x = x[:, -1, :]

        # send last image representation to pixel regression block
        x = self.to_pixels(x)

        return x
