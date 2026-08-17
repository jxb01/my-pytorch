import torch
import torch.nn as nn
from torchvision import datasets, transforms
from matplotlib import pyplot as plt
from torchvision.tv_tensors import BoundingBoxes

from model import Net
import os
from PIL import Image

import cv2 as cv
from torch.amp import autocast, GradScaler
from torchvision.transforms import v2


def giou_loss(pred, target):
    """
    计算 GIoU Loss
    pred, target: (batch, 4) 格式为 [cx, cy, w, h] 且归一化
    """
    # 将 [cx, cy, w, h] 转换为 [x1, y1, x2, y2]
    pred_x1 = pred[:, 0] - pred[:, 2] / 2
    pred_y1 = pred[:, 1] - pred[:, 3] / 2
    pred_x2 = pred[:, 0] + pred[:, 2] / 2
    pred_y2 = pred[:, 1] + pred[:, 3] / 2

    target_x1 = target[:, 0] - target[:, 2] / 2
    target_y1 = target[:, 1] - target[:, 3] / 2
    target_x2 = target[:, 0] + target[:, 2] / 2
    target_y2 = target[:, 1] + target[:, 3] / 2

    # 计算交集区域
    inter_x1 = torch.max(pred_x1, target_x1)
    inter_y1 = torch.max(pred_y1, target_y1)
    inter_x2 = torch.min(pred_x2, target_x2)
    inter_y2 = torch.min(pred_y2, target_y2)
    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)

    # 计算并集区域
    pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
    target_area = (target_x2 - target_x1) * (target_y2 - target_y1)
    union_area = pred_area + target_area - inter_area

    # 计算 IoU
    iou = inter_area / (union_area + 1e-7)

    # 计算最小外接框 (C) 的区域
    c_x1 = torch.min(pred_x1, target_x1)
    c_y1 = torch.min(pred_y1, target_y1)
    c_x2 = torch.max(pred_x2, target_x2)
    c_y2 = torch.max(pred_y2, target_y2)
    c_area = (c_x2 - c_x1) * (c_y2 - c_y1)

    # GIoU = IoU - (C \ U) / C
    giou = iou - (c_area - union_area) / (c_area + 1e-7)

    # GIoU Loss = 1 - GIoU
    return (1 - giou).mean()

class Bottledataset(torch.utils.data.Dataset):
    def __init__(self, img_root, labels_root, transform=None):
        super(Bottledataset, self).__init__()
        self.img_root = img_root
        self.labels_root = labels_root
        self.transform = transform
        self.img_filenames = [f for f in os.listdir(img_root) if f.endswith(('.jpg', '.png'))]
        self.labels_filenames = os.listdir(labels_root)

    def __len__(self):
        return len(self.labels_filenames)

    def __getitem__(self, index):
        # 读取标签
        label_path = os.path.join(self.labels_root, self.labels_filenames[index])
        with open(label_path, "r") as f:
            parts = f.readline().strip().split()

        cx = float(parts[1])
        cy = float(parts[2])
        width = float(parts[3])
        height = float(parts[4])

        # ✅ 读取图像（用 PIL，v2 完美支持）
        img_name = os.path.splitext(self.labels_filenames[index])[0] + ".jpg"
        img_path = os.path.join(self.img_root, img_name)
        img = Image.open(img_path).convert('RGB')
        labels =torch.tensor([cx, cy, width, height], dtype=torch.float32)
        # ✅ 应用变换
        if self.transform:
            img = self.transform(img)
        return img, labels
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform1 = transforms.Compose([
    transforms.Resize((342, 256)),
    transforms.ToTensor(),  # 归一化到 [0,1]
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
    datasets=Bottledataset(transform=transform1,img_root="dataset\\images\\train",labels_root="dataset\\labels\\train")
    print("device is",device)
    dataloader = torch.utils.data.DataLoader(dataset=datasets,batch_size=54,shuffle=True,num_workers=4,pin_memory=True,persistent_workers=True)
    print(len(dataloader))
    model = Net().to(device)
    model.train()
    if os.path.isfile("model.pth"):
        model.load_state_dict(torch.load("model.pth"))
    opt=torch.optim.SGD(model.parameters(),lr=0.0001,momentum=0.9)
    scaler = GradScaler(device)
    # model=torch.compile(model)
    epochs = 10
    losses = []
    best_loss=float("inf")
    for epoch in range(epochs):
        loss1=0
        for img,labels in dataloader:
            img = img.to(device)
            labels = labels.to(device)
            opt.zero_grad()
            # 前向传播自动混合精度
            with autocast(device_type="cuda", dtype=torch.float16):
                output = model(img)
                loss = giou_loss(output, labels)
            # 梯度缩放与反向传播
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            scaler.step(opt)
            scaler.update()
            loss1+=loss.item()
        avg_loss = loss1 / len(dataloader)

        # 保存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), "model.pth")
            print(f"Saved best model with loss: {best_loss:.4f}")
        losses.append(loss1 / len(dataloader))
        print("epoch is ",epoch+1,"loss is ",loss1/ len(dataloader))
    torch.save(model.state_dict(),"model.pth")
    plt.plot(losses)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.show()

