import torch.nn as nn
class Res(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, padding=1):
        super().__init__()
        self.use = (in_channels != out_channels) or (stride != 1)
        self.conv_path = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, padding, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.conv_path2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 5, stride, 2, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.relu = nn.ReLU(inplace=True)
        self.gate1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=stride, padding=2, bias=False)
                                   ,nn.BatchNorm2d(out_channels),nn.Sigmoid())
        if self.use:
            self.conv_cut = nn.Conv2d(in_channels, out_channels, stride=stride, kernel_size=1, bias=False)

    def forward(self, x):

        x1=self.conv_path(x)
        x2=self.conv_path2(x)
        if self.use:
             result=self.relu(self.conv_cut(x)+x1+x2*self.gate1(x))
        else:
             result=self.relu(x+x1+x2*self.gate1(x))
        return result



def genres(in_channels, out_channels, num_res,stride=1, padding=1):
    temp=[]
    for i in range(num_res):
        if i==0:
            temp.append(Res(in_channels, out_channels, stride=stride, padding=padding))
        else:
            temp.append(Res(out_channels, out_channels, stride=1, padding=padding))
    return temp



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.head = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.body1 = nn.Sequential(*genres(32, 64, 2, stride=2, padding=1))
        self.body2 = nn.Sequential(*genres(64, 128, 2, stride=2, padding=1))
        self.end= nn.Sequential(nn.Conv2d(128,4,kernel_size=1),nn.AdaptiveAvgPool2d(1),nn.Flatten(),nn.Sigmoid())
    def forward(self, x):
        out = self.head(x)
        out = self.body1(out)
        out = self.body2(out)
        return self.end(out)

