import torch
import torch.nn as nn
from torchvision import datasets, transforms
from matplotlib import pyplot as plt
from model import Net
import os
from PIL import Image
import cv2 as cv
class Bottledataset(torch.utils.data.Dataset):
    def __init__(self, img_root, labels_root,transform):
        super(Bottledataset, self).__init__()
        self.img_root = img_root
        self.labels_root = labels_root
        self.transform = transform
        self.img_filenames = os.listdir(self.img_root)
        self.labels_filenames = os.listdir(self.labels_root)
    def __len__(self):
        return len(self.labels_filenames)
    def __getitem__(self, index):
        with open(
                self.labels_root+"\\"+ self.labels_filenames[index],
                "r") as f:
            lines = f.readlines()
        lines = [line.rstrip() for line in lines]
        lines = lines[0].split()
        a,_=os.path.splitext(self.labels_filenames[index])
        cx = float(lines[1])
        cy = float(lines[2])
        width = float(lines[3])
        height = float(lines[4])
        labels = torch.tensor([
            cx - 0.5 * width,  # x1
            cy - 0.5 * height,  # y1
            cx + 0.5 * width,  # x2
            cy + 0.5 * height  # y2
        ], dtype=torch.float32)
        img = cv.imread(self.img_root+"\\" +a+".jpg")
        img=self.transform(img)
        return img ,labels

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform1 = transforms.Compose([transforms.ToPILImage(),transforms.Resize((224,224)),transforms.ToTensor()])
    datasets=Bottledataset(transform=transform1,img_root="data\\train\\images",labels_root="data\\train\\labels")
    print("device is",device)
    dataloader = torch.utils.data.DataLoader(dataset=datasets,batch_size=54,shuffle=True,num_workers=4,pin_memory=True,persistent_workers=True)
    print(len(dataloader))
    model = Net().to(device)
    if os.path.isfile("model.pth"):
        model.load_state_dict(torch.load("model.pth"))
    opt=torch.optim.Adam(model.parameters(),lr=0.0001)
    cri=nn.SmoothL1Loss()
    epochs = 50
    losses = []
    for epoch in range(epochs):
        loss1=0
        for img,labels in dataloader:
            img = img.to(device)
            labels = labels.to(device)
            output = model(img)
            loss = cri(output,labels)
            opt.zero_grad()
            loss.backward()
            opt.step()
            loss1+=loss.item()
        losses.append(loss1 / len(dataloader))
        print("epoch is ",epoch+1)
    torch.save(model.state_dict(),"model.pth")
    plt.plot(losses)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.show()

