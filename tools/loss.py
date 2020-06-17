import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import pytorch_msssim # import ssim, ms_ssim, SSIM, MS_SSIM

def PSNR(img1, img2, pixel_max):
    """
    Torch/numpy tensor inputs
    """
    mse = torch.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = pixel_max #255.0 previously
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def ssim(data_range=255, size_average=True, channel=1): # data_range = 255
    """
    ??
    """
    return pytorch_msssim.SSIM(data_range=data_range, size_average=size_average, channel=channel, nonnegative_ssim=True)

def ms_ssim(img1, img2, data_range): # data_range = 255
    """
    ??
    """
    return pytorch_msssim.ms_ssim( img1, img2, data_range=data_range, size_average=False)

# These losses supplied by Zheming Zhou
class FocalLoss(nn.Module):
    'Focal Loss - https://arxiv.org/abs/1708.02002'
    def __init__(self, alpha=0.25, gamma=2, reduce=False, reduction=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduce = reduce
        self.reduction = reduction

    def forward(self, pred_logits, target):
        pred = pred_logits.sigmoid()
        ce = F.binary_cross_entropy_with_logits(pred_logits, target, reduction='none')
        alpha = target * self.alpha + (1. - target) * (1. - self.alpha)
        pt = torch.where(target == 1,  pred, 1 - pred)
        ret = alpha * (1. - pt) ** self.gamma * ce
        if not self.reduce:
            return ret
        elif self.reduction == "sum":
            return torch.sum(ret)
        elif self.reduction == "mean":
            return torch.mean(ret)
        else:
            raise Exception("%s Reduction not implemented" % (self.reduction))
