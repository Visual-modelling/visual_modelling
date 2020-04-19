import torch.nn as nn
import torch.nn.functional as F


def sigmoid_256(inputs):
    """
    Scale the sigmoid function from 0-255 for greyscale image outputs
    """
    return (255*F.sigmoid(inputs))
