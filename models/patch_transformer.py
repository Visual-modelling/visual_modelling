import os
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from mmseg.models.backbones.mix_transformer import mit_b1, MixVisionTransformer
from mmseg.models.decode_heads.segformer_head import SegFormerHead


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()


"""
VM Adaptation of the patch transformer, by Jumperkables
"""
class VM_MixSeg(nn.Module):
    def __init__(self, in_chans, out_chans, img_size=64):
        super().__init__()
        setup(0,1)  # Demanded due to the torch distributed requirements of this library
        self.encoder = MixVisionTransformer(in_chans=5, img_size=64)
        self.decode_head = SegFormerHead(feature_strides=[4,8,16,32], in_channels=[64,128,256,512], channels=128, num_classes=256, in_index=[0, 1, 2, 3], decoder_params={"embed_dim":256}, dropout_ratio=0.1, align_corners=False)

    def forward(self, x):
        import ipdb; ipdb.set_trace()
        probe_ret = self.encoder(x) # TODO THIS IS NOT OPTIMALLY DONE FOR probe_ret HERE
        out = self.decode_head(probe_ret)
        return out, probe_ret

if __name__ == "__main__":
    imgs = torch.ones(32,5,64,64)
    imgs = imgs.to(0)
    vm_mixseg = VM_MixSeg(5, 1)
    vm_mixseg.to(0)
    out, probe_ret = VM_MixSeg(imgs)
    print("Finishing")


