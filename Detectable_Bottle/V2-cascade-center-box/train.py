import torch
import torch.nn as nn
from torchvision import datasets, transforms
from matplotlib import pyplot as plt
from torchvision.tv_tensors import BoundingBoxes

from model import Net
import os
from PIL import Image
import my_tool
import cv2 as cv
from torch.amp import autocast, GradScaler
from torchvision.transforms import v2


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
        # 读取标签（YOLO格式）
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

        # ✅ 包装成 BoundingBoxes（v2 才能自动同步变换）
        # 注意：格式用 CXCYWH，和你的网络输出一致
        labels =torch.tensor([cx, cy, width, height], dtype=torch.float32)


        # ✅ 应用变换
        if self.transform:
            img = self.transform(img)

        return img, labels
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform1 = transforms.Compose([
      # 将 PIL/NumPy 转为 Tensor
    transforms.Resize((224, 224)),
    # v2.RandomHorizontalFlip(p=0.5),
    # v2.RandomRotation(degrees=10),
    #v2.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # 随机平移
    transforms.ToTensor(),  # 归一化到 [0,1]
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
    datasets=Bottledataset(transform=transform1,img_root="data\\train\\images",labels_root="data\\train\\labels")
    print("device is",device)
    dataloader = torch.utils.data.DataLoader(dataset=datasets,batch_size=54,shuffle=True,num_workers=4,pin_memory=True,persistent_workers=True)
    print(len(dataloader))
    model = Net().to(device)
    if os.path.isfile("model.pth"):
        model.load_state_dict(torch.load("model.pth"))
    opt=torch.optim.Adam(model.parameters(),lr=0.001)
    scaler = GradScaler(device)
    # model=torch.compile(model)
    epochs = 20
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
                loss = my_tool.combined_loss(output, labels, 0.5)
            # 梯度缩放与反向传播
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            loss1+=loss.item()
        avg_loss = loss1 / len(dataloader)

        # 保存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), "best_model.pth")
            print(f"Saved best model with loss: {best_loss:.4f}")
        losses.append(loss1 / len(dataloader))
        print("epoch is ",epoch+1,"loss is ",loss1/ len(dataloader))
    torch.save(model.state_dict(),"model.pth")
    plt.plot(losses)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.show()

