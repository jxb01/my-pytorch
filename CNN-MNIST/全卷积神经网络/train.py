import torchvision
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn  as nn
import matplotlib.pyplot as plt


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
        x=self.classifier(x)
        x=self.pool(x)
        x=x.view(x.size(0),-1)
        return x

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    transforms = transforms.Compose([transforms.ToTensor()])
    dataset_mnist = torchvision.datasets.MNIST(root='./data', train=True, transform=transforms)
    data = DataLoader(dataset_mnist, batch_size=256, shuffle=True, num_workers=4, pin_memory=True,
                      persistent_workers=True)
    model = Net().to(device)
    model.load_state_dict(torch.load("model.pth"))
    cri = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    epochs = 10
    losses = []
    for epoch in range(epochs):
        running_loss = 0
        for images, labels in data:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = cri(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        losses.append(running_loss / len(data))
        print(f"第{epoch + 1}轮")
    print("all down")
    torch.save(model.state_dict(), "model.pth")
    plt.plot(losses)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.show()
