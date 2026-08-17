"""MNIST 手写数字识别 —— MLP 实现（优化版）"""
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


# ========== 配置 ==========
BATCH_SIZE = 64
EPOCHS = 5
LR = 0.001
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========== 数据加载 ==========
transform = T.Compose([T.ToTensor(), T.Normalize((0.1307,), (0.3081,))])  # MNIST 标准归一化

train_set = torchvision.datasets.MNIST(root="./data", train=True, transform=transform, download=True)
test_set  = torchvision.datasets.MNIST(root="./data", train=False, transform=transform, download=True)

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_set,  batch_size=BATCH_SIZE, shuffle=False)


# ========== 模型（精简为3层，约12万参数） ==========
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)  # 防止过拟合

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


# ========== 训练 & 评估工具 ==========
def evaluate(model, loader):
    """在给定数据集上评估模型，返回 loss 和准确率"""
    model.eval()
    total_loss, correct, total = 0, 0, 0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            total_loss += criterion(outputs, labels).item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += images.size(0)
    return total_loss / total, correct / total


def train():
    model = Net().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)

    train_losses, test_accs = [], []

    for epoch in range(1, EPOCHS + 1):
        # ---- 训练 ----
        model.train()
        running_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        scheduler.step()
        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # ---- 每个 epoch 后评估 ----
        test_loss, test_acc = evaluate(model, test_loader)
        test_accs.append(test_acc)

        print(f"Epoch {epoch:2d}/{EPOCHS}  |  "
              f"Train Loss: {avg_train_loss:.4f}  |  "
              f"Test Loss: {test_loss:.4f}  |  "
              f"Test Acc: {test_acc:.2%}")

    # ---- 保存模型 ----
    torch.save(model.state_dict(), "./mnist_mlp.pth")
    print(f"\n模型已保存 → mnist_mlp.pth")
    print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"最佳测试准确率: {max(test_accs):.2%}")

    # ---- 可视化 ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.plot(train_losses, marker="o")
    ax1.set_title("训练损失")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.grid(True, alpha=0.3)

    ax2.plot(test_accs, marker="o", color="green")
    ax2.set_title("测试准确率")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("training_curve.png", dpi=120)
    plt.show()


if __name__ == "__main__":
    train()
