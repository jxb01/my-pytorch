import os
import torch
from torch.utils.data import Dataset, DataLoader
from dataset import YOLODataset
from model import Net
import torch.optim as optim
from tool import *
from torch.amp import autocast, GradScaler
import matplotlib.pyplot as plt
if __name__ == "__main__":
    model = Net()
    if os.path.isfile('model.pth'):
        model.load_state_dict(torch.load('model.pth'))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    dataset = YOLODataset(
        image_dir="./dataset/images/train",
        label_dir="./dataset/labels/train",
        img_width=256,
        img_height=342,
        grid_height=11,
        grid_width=8,
        mosaic_prob=0.5,
    )
    dataloader = DataLoader(dataset,batch_size=80,shuffle=True,num_workers=4,pin_memory=True,persistent_workers=True)
    model.train()
    #scaler = GradScaler()
    epoch=10
    opt=optim.SGD(model.parameters(),lr=0.001,momentum=0.9,nesterov=True)
    losses=[]
    best_loss = float("inf")
    for epoch in range(epoch):
        loss1=0
        for img,target in dataloader:
            img = img.to(device)
            target = target.to(device)
            opt.zero_grad()
            # with autocast(device_type="cuda",dtype=torch.float16):
            output = model(img)
            # loss=compute_loss(output,target,len(target))
            #print("output.size is",output.size(),"target.size is",target.size())
            loss = compute_loss(output, target, grid_w=8, grid_h=11)
            loss.backward()
            opt.step()
            # scaler.scale(loss).backward()
            # scaler.unscale_(opt)
            # scaler.step(opt)
            # scaler.update()
            loss1+=loss.item()
        avg_loss = loss1 / len(dataloader)
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



