"""
PyTorch 入门教程 —— 从零开始理解核心概念
==============================================
本教程涵盖：
  1. Tensor 基础操作
  2. 自动求导 (autograd)
  3. 构建神经网络 (nn.Module)
  4. 损失函数与优化器
  5. 完整训练流程（用合成数据做二分类）
"""

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# ============================================================
# 第1部分：Tensor 基础
# Tensor 是 PyTorch 的核心数据结构，类似于 NumPy 的 ndarray，
# 但可以在 GPU 上运算，并支持自动求导。
# ============================================================
print("=" * 50)
print("1. Tensor 基础操作")
print("=" * 50)

# 创建 Tensor 的几种方式
a = torch.tensor([1, 2, 3])              # 从列表创建
b = torch.zeros(2, 3)                     # 全 0
c = torch.ones(2, 3)                      # 全 1
d = torch.rand(2, 3)                      # 随机均匀分布 [0, 1)
e = torch.randn(2, 3)                     # 随机标准正态分布
f = torch.arange(0, 10, 2)               # 等差数列

print(f"从列表创建: {a}")
print(f"全0矩阵 shape={b.shape}:\n{b}")
print(f"随机正态:\n{e}")

# 基本运算
x = torch.tensor([1.0, 2.0, 3.0])
y = torch.tensor([4.0, 5.0, 6.0])
print(f"\nx + y = {x + y}")               # 加法
print(f"x * y = {x * y}")                 # 逐元素乘法
print(f"x @ y = {x @ y}")                 # 点积（矩阵乘法）
print(f"x 的均值: {x.mean():.2f}")
print(f"x 的和: {x.sum():.2f}")

# 形状变换
t = torch.arange(12).reshape(3, 4)        # 重塑为 3x4
print(f"\nreshape(3,4):\n{t}")
print(f"转置:\n{t.T}")                     # 转置
print(f"view 也是重塑:\n{t.view(4, 3)}")

# GPU 支持（如果可用）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n使用设备: {device}")

# ============================================================
# 第2部分：自动求导 (autograd)
# PyTorch 会自动追踪计算图，反向传播时自动计算梯度。
# ============================================================
print("\n" + "=" * 50)
print("2. 自动求导 (autograd)")
print("=" * 50)

# requires_grad=True 表示需要追踪梯度
w = torch.tensor([2.0], requires_grad=True)
b = torch.tensor([1.0], requires_grad=True)

# 前向计算: y = w * x + b
x_val = torch.tensor([3.0])
y_pred = w * x_val + b

# 定义损失: loss = (y_pred - y_true)^2
y_true = torch.tensor([10.0])
loss = (y_pred - y_true) ** 2

print(f"预测值 y_pred = {y_pred.item():.2f}, 真实值 y_true = {y_true.item():.2f}")
print(f"损失 loss = {loss.item():.2f}")

# 反向传播 —— 自动计算梯度
loss.backward()
print(f"w 的梯度 dl/dw = {w.grad.item():.2f}")  # 应为 2*(wx+b-y)*x = 2*(7-10)*3 = -18
print(f"b 的梯度 dl/db = {b.grad.item():.2f}")  # 应为 2*(wx+b-y)*1 = 2*(7-10) = -6

# ============================================================
# 第3部分：构建神经网络 (nn.Module)
# nn.Module 是构建所有神经网络的基类。
# ============================================================
print("\n" + "=" * 50)
print("3. 构建一个简单的分类网络")
print("=" * 50)


class SimpleClassifier(nn.Module):
    """一个简单的3层全连接网络，用于二分类"""

    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        # 定义网络层
        self.fc1 = nn.Linear(input_dim, hidden_dim)    # 输入 -> 隐藏层
        self.relu = nn.ReLU()                           # 激活函数
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)    # 隐藏层 -> 隐藏层
        self.fc3 = nn.Linear(hidden_dim, output_dim)    # 隐藏层 -> 输出
        self.sigmoid = nn.Sigmoid()                     # 输出概率

    def forward(self, x):
        """定义前向传播过程"""
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x


# 实例化模型
model = SimpleClassifier(input_dim=2, hidden_dim=16, output_dim=1).to(device)
print(model)
print(f"\n模型参数数量: {sum(p.numel() for p in model.parameters()):,}")

# ============================================================
# 第4部分：损失函数与优化器
# ============================================================
print("\n" + "=" * 50)
print("4. 损失函数 & 优化器")
print("=" * 50)

criterion = nn.BCELoss()                # 二分类交叉熵损失
optimizer = optim.Adam(model.parameters(), lr=0.01)  # Adam 优化器

print(f"损失函数: {criterion}")
print(f"优化器: {optimizer}")

# ============================================================
# 第5部分：生成合成数据并训练
# 生成两类点：(0,0)附近 和 (1,1)附近，让模型学会区分。
# ============================================================
print("\n" + "=" * 50)
print("5. 训练模型（二分类任务）")
print("=" * 50)

# 生成数据：类别0 集中在 (0,0) 附近，类别1 集中在 (1,1) 附近
torch.manual_seed(42)
n_samples = 500

# 类别0
X0 = torch.randn(n_samples, 2) * 0.5 + torch.tensor([0.0, 0.0])
y0 = torch.zeros(n_samples, 1)

# 类别1
X1 = torch.randn(n_samples, 2) * 0.5 + torch.tensor([2.0, 2.0])
y1 = torch.ones(n_samples, 1)

# 合并并打乱
X = torch.cat([X0, X1], dim=0)
y = torch.cat([y0, y1], dim=0)
shuffle_idx = torch.randperm(len(X))
X, y = X[shuffle_idx], y[shuffle_idx]

X, y = X.to(device), y.to(device)

print(f"数据量: {len(X)}, 特征维度: {X.shape[1]}")
print(f"正样本比例: {y.mean().item():.1%}")

# ============================================================
# 训练循环
# ============================================================
epochs = 200
losses = []

model.train()  # 设置为训练模式
for epoch in range(epochs):
    # 前向传播
    y_pred = model(X)
    loss = criterion(y_pred, y)

    # 反向传播
    optimizer.zero_grad()   # 清空梯度（重要！）
    loss.backward()          # 计算梯度
    optimizer.step()         # 更新参数

    losses.append(loss.item())

    if (epoch + 1) % 50 == 0:
        # 计算准确率
        y_pred_class = (y_pred > 0.5).float()
        acc = (y_pred_class == y).float().mean()
        print(f"Epoch [{epoch+1:3d}/{epochs}]  Loss: {loss.item():.4f}  Acc: {acc.item():.2%}")

# ============================================================
# 第6部分：评估模型
# ============================================================
print("\n" + "=" * 50)
print("6. 最终评估")
print("=" * 50)

model.eval()  # 设置为评估模式
with torch.no_grad():  # 不需要计算梯度
    y_pred = model(X)
    y_pred_class = (y_pred > 0.5).float()
    accuracy = (y_pred_class == y).float().mean()
    print(f"训练集准确率: {accuracy.item():.2%}")

    # 测试几个样例
    test_points = torch.tensor([
        [0.0, 0.0],   # 应该预测为 0
        [2.0, 2.0],   # 应该预测为 1
        [1.0, 1.0],   # 边界点
        [0.5, 0.5],   # 更接近类别0
    ]).to(device)
    predictions = model(test_points)
    print("\n测试预测:")
    for pt, prob in zip(test_points.cpu(), predictions.cpu()):
        cls = 1 if prob > 0.5 else 0
        print(f"  点 ({pt[0]:.1f}, {pt[1]:.1f}) -> 概率={prob.item():.4f}, 预测类别={cls}")

# ============================================================
# 第7部分：可视化
# ============================================================
print("\n" + "=" * 50)
print("7. 可视化结果")
print("=" * 50)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 左图：损失曲线
axes[0].plot(losses)
axes[0].set_title("训练损失曲线")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].grid(True, alpha=0.3)

# 右图：决策边界
axes[1].set_title("数据分布 & 预测")
X_np = X.cpu().numpy()
y_np = y.cpu().numpy().flatten()

# 画出真实数据点
axes[1].scatter(X_np[y_np == 0, 0], X_np[y_np == 0, 1],
                c="blue", label="类别 0", alpha=0.5, s=30)
axes[1].scatter(X_np[y_np == 1, 0], X_np[y_np == 1, 1],
                c="red", label="类别 1", alpha=0.5, s=30)

# 画出决策区域
with torch.no_grad():
    xx, yy = torch.meshgrid(torch.linspace(-2, 4, 100), torch.linspace(-2, 4, 100), indexing="ij")
    grid = torch.stack([xx.flatten(), yy.flatten()], dim=1).to(device)
    zz = model(grid).reshape(100, 100).cpu().numpy()
    axes[1].contourf(xx.cpu(), yy.cpu(), zz, levels=20, cmap="RdBu", alpha=0.3)
    axes[1].contour(xx.cpu(), yy.cpu(), zz, levels=[0.5], colors="black", linewidths=2, linestyles="--")

axes[1].legend()
axes[1].set_xlabel("特征 1")
axes[1].set_ylabel("特征 2")

plt.tight_layout()
plt.savefig("training_result.png", dpi=150)
print("图片已保存为 training_result.png！")
plt.show()

print("\n" + "=" * 50)
print("恭喜！你已经完成了 PyTorch 入门教程！")
print("=" * 50)
print("""
  回顾学到的内容：
  - Tensor: 数据的基本载体，类似 NumPy 数组
  - autograd: 自动求导，反向传播的核心
  - nn.Module: 构建神经网络的基类
  - Loss & Optimizer: 损失函数衡量误差，优化器更新参数
  - 训练循环: zero_grad -> forward -> loss -> backward -> step
""")
