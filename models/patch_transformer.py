import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from mmseg.models.backbones.mix_transformer import mit_b1, MixVisionTransformer
from mmseg.models.decode_heads.segformer_head import SegFormerHead
from mmseg.ops import resize

# Local imports
from .UpDown2D import sigmoid_256

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '0'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()


"""
Adaptation of the SegFormerHead to pass the latent layers in forward too for our use in linear probes
"""
class SegFormerHeadWithProbes(SegFormerHead):
    def __init__(self, feature_strides, **kwargs):
        super().__init__(feature_strides, **kwargs)

    def forward(self, inputs):
        x = self._transform_inputs(inputs)  # len=4, 1/4,1/8,1/16,1/32
        c1, c2, c3, c4 = x
        probe_ret = []

        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4.shape
        _c4 = self.linear_c4(c4).permute(0,2,1).reshape(n, -1, c4.shape[2], c4.shape[3])
        _c4 = resize(_c4, size=c1.size()[2:],mode='bilinear',align_corners=False)
        probe_ret.append(_c4)

        _c3 = self.linear_c3(c3).permute(0,2,1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c3 = resize(_c3, size=c1.size()[2:],mode='bilinear',align_corners=False)
        probe_ret.append(_c3)

        _c2 = self.linear_c2(c2).permute(0,2,1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c2 = resize(_c2, size=c1.size()[2:],mode='bilinear',align_corners=False)
        probe_ret.append(_c2)

        _c1 = self.linear_c1(c1).permute(0,2,1).reshape(n, -1, c1.shape[2], c1.shape[3])
        probe_ret.append(_c1)

        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))

        x = self.dropout(_c)
        x = self.linear_pred(x)
        probe_ret.append(x)

        return x, probe_ret


"""
VM Adaptation of the patch transformer, by Jumperkables
"""
class VM_MixSeg(nn.Module):
    def __init__(self, in_chans, out_chans, img_size=64):
        super().__init__()
        self.out_chans = out_chans
        setup(0,1)  # Demanded due to the torch distributed requirements of this library
        #cleanup()
        self.encoder = MixVisionTransformer(in_chans=in_chans, img_size=img_size)
        self.decode_head = SegFormerHeadWithProbes(feature_strides=[4,8,16,32], in_channels=[64,128,256,512], channels=128, num_classes=16*out_chans, in_index=[0, 1, 2, 3], decoder_params={"embed_dim":256}, dropout_ratio=0.1, align_corners=False)

    def forward(self, x):
        probe_ret = self.encoder(x) # TODO THIS IS NOT OPTIMALLY DONE FOR probe_ret HERE
        out, probe_out = self.decode_head(probe_ret)   # (B, 16*out_chans, 16, 16)
        probe_ret += probe_out
        out = out.view(x.shape[0], 16*self.out_chans, -1)  # (B, out_chans, 256)
        out = F.fold(out, output_size=(64,64), kernel_size=(4,4), stride=(4,4)) # (B, out_chans, 64, 64) NOTE spatial dimensions of patches are preserved
        probe_ret.append(out)
        return sigmoid_256(out), probe_ret

if __name__ == "__main__":
    imgs = torch.ones(32,5,64,64)
    imgs = imgs.to(0)
    vm_mixseg = VM_MixSeg(5, 2)
    vm_mixseg.to(0)
    out, probe_ret = vm_mixseg(imgs)
    breakpoint()
    print("Finishing")


