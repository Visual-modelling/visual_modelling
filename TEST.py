from models.patch_transformer import MixVisionTransformer
import torch
import torch.nn as nn

from mmseg.models.backbones.unet import DeconvModule

deconv0 = nn.Sequential(
    DeconvModule(64,4),
    DeconvModule(4,1)
)
deconv1 = nn.Sequential(
    DeconvModule(128,64),
    DeconvModule(64,4),
    DeconvModule(4,1)
)
deconv2 = nn.Sequential(
    DeconvModule(256,128),
    DeconvModule(128,64),
    DeconvModule(64,4),
    DeconvModule(4,1)
)
deconv3 = nn.Sequential(
    DeconvModule(512,256),
    DeconvModule(256,128),
    DeconvModule(128,64),
    DeconvModule(64,4),
    DeconvModule(4,1)
)

ins = torch.ones(32,5,64,64)
label = torch.ones(32,1,64,64)
test = MixVisionTransformer(img_size=64, in_chans=5)
out = test(ins)
out0 = deconv0(out[0])
out1 = deconv1(out[1])
out2 = deconv2(out[2])
out3 = deconv3(out[3])
print(out0.shape, out1.shape, out2.shape, out3.shape)
import ipdb; ipdb.set_trace()
print("Beeg sorted")
