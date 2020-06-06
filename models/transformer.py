import random
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from transformers import BertModel

#####
# Dean's transformer model adapted by Winterbottom 
#####

def load_model(model_name, config):
    """ Model Loader for all the models contained within this file.
        'model_name': str
        'config': list
    """
    model = None
    config = [eval(x) for x in config]

    if model_name == 'VMTransformer':
        model = VMTransformer(*config)
    if model_name == 'VMTransformerOld':
        model = VMTransformerOld(*config)

    return model


class Maxout(nn.Module):
    def __init__(self, pool_size):
        super().__init__()
        self._pool_size = pool_size

    def forward(self, x):
        assert x.shape[1] % self._pool_size == 0, \
            'Wrong input last dim size ({}) for Maxout({})'.format(x.shape[1], self._pool_size)
        m, i = x.view(*x.shape[:1], x.shape[1] // self._pool_size, self._pool_size, *x.shape[2:]).max(2)
        return m


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=10):
        super().__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class SelfAttention(nn.Module):
    def __init__(self, k, heads=1):
        super().__init__()
        self.k, self.heads = k, heads

        self.tokeys = nn.Linear(k, k * heads, bias=False)
        self.toqueries = nn.Linear(k, k * heads, bias=False)
        self.tovalues = nn.Linear(k, k * heads, bias=False)

        self.unifyheads = nn.Sequential(nn.Linear(heads * k, k))

    def forward(self, x):
        b, t, k = x.size()
        h = self.heads
        queries = self.toqueries(x).view(b, t, h, k)
        keys = self.tokeys(x)   .view(b, t, h, k)
        values = self.tovalues(x) .view(b, t, h, k)

        # - fold heads into the batch dimension
        keys = keys.transpose(1, 2).contiguous().view(b * h, t, k)
        queries = queries.transpose(1, 2).contiguous().view(b * h, t, k)
        values = values.transpose(1, 2).contiguous().view(b * h, t, k)

        queries = queries / (k ** (1/4))
        keys = keys / (k ** (1/4))

        dot = torch.bmm(queries, keys.transpose(1, 2))

        dot = F.softmax(dot, dim=2)

        out = torch.bmm(dot, values).view(b, h, t, k)

        out = out.transpose(1, 2).contiguous().view(b, t, h * k)
        return self.unifyheads(out)


class TransformerBlock(nn.Module):
  def __init__(self, k, heads, ff=False, residual=False, norm=True):
    super().__init__()

    self.ff = ff
    self.residual = residual
    self.norm = norm

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
    def __init__(self, k, heads=1, depth=1, ff=True, residual=False, norm=True, seq_len=5):
        super().__init__()

        self.seq_len = seq_len
        # self.label_dim = label_dim

        # initialize pixel and temporal embedding layers
        self.pix_emb = nn.Embedding(k, k)
        self.pos_emb = PositionalEmbedding(k, max_len=self.seq_len)

        # self attention layers
        tblocks = []
        for i in range(depth):
            tblocks.append(TransformerBlock(k=k, heads=heads, ff=ff, residual=residual, norm=norm))
        self.tblocks = nn.Sequential(*tblocks)

        # classification/regression layer
        self.toprobs = nn.Sequential(
            nn.Linear(k, k),
            nn.Linear(k, k),
            nn.Sigmoid())

    def forward(self, x):

        b, t, h, w = x.size()
        k = h*w
        x=x.view(b, t, -1)
        # generate pixel positional embeddings
        coordinates = torch.arange(t).cuda()
        coordinates = self.pix_emb(coordinates)[None, :, :].expand(b, t, k)
        # generate temporal embeddings
        positions = self.pos_emb(x)

        # add pixel coordinate embeddings
        x *= coordinates
        # add pixel temporal embeddings
        x += positions

        # cls_token = torch.ones(b, 1, k).cuda()
        # x = torch.cat((cls_token, x), dim=1)

        x = self.tblocks(x.cuda())

        # Average-pool over the t dimension and project to class
        # probabilities
        # x = self.toprobs(x.mean(dim=1))

        x = self.toprobs(x[:, 0, :])

        return x.view(b,h,w).unsqueeze(1)
