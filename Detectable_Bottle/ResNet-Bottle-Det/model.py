import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.modules import conv

class ratio2(nn.Module):
    def __init__(self):
        super().__init__()
        self.main=nn.Sequential(nn.LazyConv2d(32,7,stride=2),nn.ReLU(),nn.ReLU(),nn.Conv2d(32,7,3,stride=2),
                                nn.Flatten(),nn.LazyLinear(16),nn.ReLU(),nn.Linear(16,2))
    def forward(self, img):
        return self.main(img)
class ratio(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone= models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.backbone.conv1=nn.Conv2d(3, 64, kernel_size=(7,7), stride=(2,2), padding=(3,3), bias=False)
        self.num_ftrs = self.backbone.fc.in_features
        self.backbone.avgpool = nn.Sequential(
            nn.Conv2d(self.num_ftrs, 7, 3, 1),
        )
        self.backbone.fc = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(2),
        )
    def forward(self, img):
        return self.backbone(img)
class mlp(nn.Module):
    def __init__(self):
        super().__init__()
        self.main=nn.Sequential(nn.LazyLinear(64),nn.ReLU(),nn.Linear(64,1),nn.Sigmoid())
    def forward(self,x):
        return self.main(x)*0.5

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.backbone = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        self.num_ftrs = self.backbone.fc.in_features
        self.backbone.avgpool=nn.Sequential(
            nn.Conv2d(self.num_ftrs, 7, 3, 1),
        )
        self.backbone.fc=nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(64),
            nn.ReLU(inplace=True),
            nn.LazyLinear(4)
        )
        self.ratio=ratio2()
    def forward(self, x):
        out = self.backbone(x)
        center=out[:,0:2]
        wh=out[:,2:4]
        offest=self.ratio(x)
        result=torch.cat([center,wh+0.5*offest],1)
        print(offest.mean().item(), offest.std().item())
        # result=self.ratio(x,out)+out
        return result
