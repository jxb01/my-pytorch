import torch
import torch.nn as nn
import torchvision.models as models

backon=models.resnet34(pretrained="default")
main = nn.Sequential(*list(backon.children())[:-2],nn.LazyConv2d(5,1,1,0))#x y w h conf
x = torch.randn(1, 3, 342, 256)  # 模拟输入
out = main(x)
print(out.shape)