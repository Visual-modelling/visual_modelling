import torch
import torch.nn as nn
import torch.nn.functional as F

def l1_loss(img, gtimg):
    return F.l1_loss(img, gtimg)