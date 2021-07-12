from models.patch_transformer import MixVisionTransformer
import torch

ins = torch.ones(32,5,64,64)
label = torch.ones(32,1,64,64)
test = MixVisionTransformer(img_size=64)
import ipdb; ipdb.set_trace()
out = test(ins)
print("Beeg sorted")
