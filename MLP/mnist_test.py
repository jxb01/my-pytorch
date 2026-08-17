import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn

# ========== 1. 设备设置 ==========
if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
print(device)

# ========== 2. 加载测试数据 ==========
transform = transforms.Compose([transforms.ToTensor()])
test_set = torchvision.datasets.MNIST(root='./data', train=False, transform=transform)  # 改名为 test_set
test_loader = DataLoader(test_set, batch_size=64, shuffle=False)  # shuffle=False 更好


# ========== 3. 定义模型类 ==========
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# ========== 4. 加载模型 ==========
model = Model()
model.load_state_dict(torch.load('model.pth', map_location=torch.device('cpu'), weights_only=True))
model = model.to(device)
model.eval()

# ========== 5. 计算准确率 ==========
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        # 统计正确预测数
        correct += (predicted == labels).sum().item()
        total += labels.size(0)  # 统计总样本数

# 输出准确率
print(f"测试集正确预测数: {correct} / {total}")
print(f"测试集准确率: {correct / total * 100:.2f}%")