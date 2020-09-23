from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import pytorch_msssim # import ssim, ms_ssim, SSIM, MS_SSIM
import piq 
import os, sys
from .pytorch_i3d import InceptionI3d
import numpy as np
import tensorflow.compat.v1 as tf
from . import frechet_video_distance as fvd

tf.compat.v1.enable_eager_execution() 
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


######################################
######################################
# Loss classes functions
######################################
######################################
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

class FVD():
    def __init__(self):
        """
        Implementation based from: https://github.com/google-research/google-research/blob/master/frechet_video_distance/example.py
        """
        #tf.app.run(main)
        #print("FVD loss object initialised")
        pass
    #def __call__(self, vid, gt):
    #    return tf.app.run(self.callin(vid, gt))

    def __call__(self, vid, gt):
        vid, gt = vid.unsqueeze(0).permute(0,1,3,4,2), gt.unsqueeze(0).permute(0,1,3,4,2)
        vid, gt = vid.repeat(16,1,1,1,3), gt.repeat(16,1,1,1,3)   # Turn greyscale images into 'RGB' by cloning in each channel
        assert vid.shape[0] == gt.shape[0], "Vid and ground truth batch sizes should be the same" 
        NUMBER_OF_VIDEOS = vid.shape[0]
        VIDEO_LENGTH = vid.shape[1]
        assert NUMBER_OF_VIDEOS % 16 == 0, "For some reason, the batch size (number of videos) processed through this implementation must be divisible by 16"
        with HiddenPrints():
            with tf.Graph().as_default():
                vid, gt = tf.convert_to_tensor(vid.numpy()), tf.convert_to_tensor(gt.numpy())
                result = fvd.calculate_fvd(fvd.create_id3_embedding(fvd.preprocess(vid, (224, 224))), fvd.create_id3_embedding(fvd.preprocess(gt,(224, 224))))
                with tf.Session() as sess:
                    sess.run(tf.global_variables_initializer())
                    sess.run(tf.tables_initializer())
                    return float(sess.run(result))


######################################
######################################
# Utility functions
######################################
######################################
def torch_to_tf(tensor):
    return tf.convert_to_tensor(tensor.numpy())


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout



######################################
######################################
# Main 
######################################
######################################
if __name__ == "__main__":
    #tf.app.run(main)
    #    fvd_loss = FVD()
    #    vid = tf.zeros([16, 10, 64, 64, 3])
    #    gt = tf.ones([16, 10, 64, 64, 3]) * 255
    #    import ipdb; ipdb.set_trace()
    #    fvd_metric = fvd_loss(vid, gt)
    #    print(f"It worked, thank fuck. FVD is {fvd_metric}")
    vid = torch.zeros([3, 3, 64, 64])
    gt = torch.ones([3, 3, 64, 64]) * 255
    FID_Metric = piq.FID()
    lpips_Metric = piq.LPIPS()
    psnr_val = piq.psnr(vid, gt)
    fid = FID_Metric(torch.rand(1000,24), torch.rand(1000,24))
    lpips_val = lpips_Metric(vid, gt)
    print(f"FID:{fid}, PSNR:{psnr_val}, LPIPS:{lpips_val}")
