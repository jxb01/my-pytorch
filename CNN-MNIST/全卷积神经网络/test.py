import matplotlib.pyplot as plt
import torchvision
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn  as nn
import os
import sys
import PIL.Image as Image
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3,padding=1,stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3,padding=1,stride=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3,padding=1,stride=1)
        self.classifier = nn.Conv2d(128, 10, 1)
        self.pool=nn.AdaptiveAvgPool2d(1)
        self.relu=nn.ReLU()
    def forward(self,x):
        x=self.conv1(x)
        x=self.relu(x)
        x=self.conv2(x)
        x=self.relu(x)
        x=self.conv3(x)
        x=self.relu(x)
        x = self.classifier(x)
        x=self.pool(x)
        x=x.view(x.size(0),-1)
        return x
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    if not os.path.exists("model.pth"):
        print("model.pth not exist")
        sys.exit(0)
    transforms = transforms.Compose([transforms.ToTensor()])
    model = Net().to(device)
    model.load_state_dict(torch.load("model.pth"))
    model.eval()
    dataset_mnist = torchvision.datasets.MNIST(root='./data', train=False, transform=transforms)
    data = DataLoader(dataset_mnist, batch_size=128)
    losses = []
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in data:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print("最终准确率为", 100 * correct / total)
    img=Image.open("./data\\test.png")
    img = img.resize((28, 28))
    img = img.convert("RGB").convert("L")
    img_tensor=transforms(img)
    img_tensor=img_tensor.unsqueeze(0)
    img_tensor=img_tensor.to(device)
    plt.imshow(img_tensor.cpu().numpy().squeeze(), cmap="gray")
    plt.show()
    output = model(img_tensor)
    _, predicted = torch.max(output, 1)
    print("识别结果为",predicted.item())