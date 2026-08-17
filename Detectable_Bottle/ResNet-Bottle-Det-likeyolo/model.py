import torch
import torch.nn as nn
import torchvision.models as models


class Net(nn.Module):
    """
    仿 YOLO 检测头（单类, 每格 1 框）:
    输出 (B, grid_w, grid_h, 5) = [sx, sy, w, h, conf]
      - sx, sy: 目标中心在格子内的偏移, sigmoid 约束到 (0,1)
      - w, h  : 归一化宽高（整图 0~1）, sigmoid 约束到 (0,1), 保证恒为正
      - conf  : 置信度, sigmoid 约束到 (0,1)
    坐标约定: dim1 = 列（宽方向, grid_w）, dim2 = 行（高方向, grid_h）
    """

    def __init__(self):
        super(Net, self).__init__()
        backon = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(backon.children())[:-2])  # 下采样 32 倍
        self.head = nn.Conv2d(512, 5, kernel_size=1)                  # 每格输出 5 个通道

    def forward(self, x):
        out = self.head(self.backbone(x))               # (B, 5, H, W)
        out = torch.sigmoid(out)                        # 5 个通道全部过 sigmoid
        out = torch.permute(out, (0, 3, 2, 1)).contiguous()  # (B, W, H, 5) = (B, grid_w, grid_h, 5)
        return out
