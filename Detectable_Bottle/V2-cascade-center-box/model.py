import torch.nn as nn
import torch
class fix_wh(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            *genres(7,64,1,stride=2,padding=1),
            *genres(64,32,1),
            nn.Conv2d(32,2,1,1,0),
            nn.AdaptiveAvgPool2d(1)
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flat = nn.Flatten()
        self.tah = nn.Tanh()
    def forward(self,img,position):
        pos_map = position.unsqueeze(-1).unsqueeze(-1)  # [batch, 4, 1, 1]
        pos_map = pos_map.expand(-1, -1, img.shape[2], img.shape[3])
        pos_map = torch.cat([img,pos_map],1)
        out=self.conv(pos_map)
        out=self.flat(out)
        out = self.tah(out)*0.3
        return out

class ratio(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            *genres(3, 64, 1, stride=2, padding=1),
            *genres(64, 32, 1),
            nn.Conv2d(32, 1, 1, 1, 0),
            nn.AdaptiveAvgPool2d(1)
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flat = nn.Flatten()
        self.sigmoid = nn.Sigmoid()

    def forward(self, img):
        out = self.conv(img)
        out = self.flat(out)
        out = self.sigmoid(out) * 1+0.5
        return out

class fix_xy(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            *genres(5,64,2,stride=2,padding=1),
            *genres(64,32,1),
            nn.Conv2d(32,2,1,1,0),
            nn.AdaptiveAvgPool2d(1)
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flat = nn.Flatten()
        self.tah = nn.Tanh()
    def forward(self,img,position):
        pos_map = position.unsqueeze(-1).unsqueeze(-1)  # [batch, 4, 1, 1]
        pos_map = pos_map.expand(-1, -1, img.shape[2], img.shape[3])
        pos_map = torch.cat([img,pos_map],1)
        out=self.conv(pos_map)
        out=self.flat(out)
        out = self.tah(out)*0.3
        return out


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
            nn.Conv2d(in_channels, out_channels, 5, stride, padding=2, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.relu = nn.ReLU(inplace=True)

        if self.use:
            self.conv_cut = nn.Conv2d(in_channels, out_channels, stride=stride, kernel_size=1, bias=False)

    def forward(self, x):

        x1=self.conv_path(x)
        x2=self.conv_path2(x)
        if self.use:
             result=self.relu(self.conv_cut(x)+x1+x2)
        else:
             result=self.relu(x+x1+x2)
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
        self.fix_wh=fix_wh()
        self.fixposition=fix_xy()
        self.ratio=ratio()
    def forward(self, x):
        out = self.head(x)
        temp = x
        out = self.body1(out)
        out = self.body2(out)
        position=self.end(out)
        out_xy=self.fixposition(temp,position[:,0:2])
        # 取出前两个数 (x, y)
        center = position[:, 0:2]
        offset = out_xy
        # 取出后两个数 (w, h)
        wh = position[:, 2:4]  # 形状是 [B, 2]
        # 让后前个数 (xy) 与 offset 相加
        # 注意：这里没有用到循环，PyTorch 会自动按位置对应相加
        xy_refined = center + offset  # 形状依然是 [B, 2]
        # 最后，把 cx, cy 和 修正后的 w, h 重新拼起来
        position = torch.cat([xy_refined,wh], dim=1)
        out=self.fix_wh(img=temp,position=position)
        center = position[:, 0:2]  # 形状是 [B, 2]
        offset = out
        # 取出后两个数 (w, h)
        wh = position[:, 2:4]  # 形状是 [B, 2]
        # 让后两个数 (wh) 与 offset 相加
        # 注意：这里没有用到循环，PyTorch 会自动按位置对应相加
        wh_refined = wh + offset  # 形状依然是 [B, 2]
        # 最后，把 cx, cy 和 修正后的 w, h 重新拼起来
        rat = self.ratio(temp)
        final_box = torch.cat([center, wh_refined*rat], dim=1)

        return  final_box

