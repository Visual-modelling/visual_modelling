import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from mmseg.models.backbones.mix_transformer import mit_b1, MixVisionTransformer
from mmseg.models.decode_heads.segformer_head import SegFormerHead


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank)#, world_size=world_size)

def cleanup():
    dist.destroy_process_group()


"""
VM Adaptation of the patch transformer, by Jumperkables
"""
class VM_MixSeg(nn.Module):
    def __init__(self, in_chans, out_chans, img_size=64):
        super().__init__()
        self.out_chans = out_chans
        setup(0,1)  # Demanded due to the torch distributed requirements of this library
        self.encoder = MixVisionTransformer(in_chans=5, img_size=64)
        self.decode_head = SegFormerHead(feature_strides=[4,8,16,32], in_channels=[64,128,256,512], channels=128, num_classes=16*out_chans, in_index=[0, 1, 2, 3], decoder_params={"embed_dim":256}, dropout_ratio=0.1, align_corners=False)

    def forward(self, x):
        probe_ret = self.encoder(x) # TODO THIS IS NOT OPTIMALLY DONE FOR probe_ret HERE
        out = self.decode_head(probe_ret)   # (B, 16*out_chans, 16, 16)
        out = out.view(x.shape[0], 16*self.out_chans, -1)  # (B, out_chans, 256)
        out = F.fold(out, output_size=(64,64), kernel_size=(4,4), stride=(4,4)) # (B, out_chans, 64, 64) NOTE spatial dimensions of patches are preserved
        return out, probe_ret

if __name__ == "__main__":
    imgs = torch.ones(32,5,64,64)
    imgs = imgs.to(0)
    vm_mixseg = VM_MixSeg(5, 2)
    vm_mixseg.to(0)
    out, probe_ret = vm_mixseg(imgs)
    print("Finishing")


