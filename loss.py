import torch
import torch.nn as nn
import torch.nn.functional as F

def l2loss(img, gtimg):
    return -F.sum(F.squared_difference(img, gtimg))