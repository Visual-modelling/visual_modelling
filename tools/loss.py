import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import pytorch_msssim # import ssim, ms_ssim, SSIM, MS_SSIM

def l1_loss(img1, img2):
    """
    Torch tensor inputs
    """
    return F.l1_loss(img1, img2)

def PSNR(img1, img2, pixel_max):
    """
    Torch/numpy tensor inputs
    """
    mse = torch.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = pixel_max #255.0 previously
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def ssim(img1, img2, data_range): # data_range = 255
    """
    ??
    """
    return pytorch_msssim.ssim( img1, img2, data_range=data_range, size_average=False)

def ms_ssim(img1, img2, data_range): # data_range = 255
    """
    ??
    """
    return pytorch_msssim.ms_ssim( img1, img2, data_range=data_range, size_average=False)