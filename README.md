# my-pytorch 🧠

PyTorch 深度学习**学习与实践项目集** —— 从 MNIST 图像分类到塑料瓶目标检测。

A collection of PyTorch learning & practice projects: MNIST classification (MLP / CNN / fully-convolutional) and plastic-bottle object detection (ResNet baseline / cascade center-box / YOLO-like).

## 环境 (Environment)

实测于 conda 环境 `pytorch`：

| 组件 | 版本 |
|---|---|
| Python | 3.12 |
| PyTorch | 2.10.0（CUDA 可用） |
| torchvision | 0.25.0 |
| numpy / matplotlib | 2.5.2 / 3.11.0 |
| opencv-python / Pillow | 4.13.0 / 11.3.0 |
| scikit-learn | 1.8.0 |

> ⚠️ **兼容性提示**：`Detectable_Bottle/V2-cascade-center-box` 使用了 `torchvision.tv_tensors.BoundingBoxes`，
> 该模块在 **torchvision ≥ 0.19 已被移除**。如需运行该子项目，请使用 torchvision ≤ 0.18（例如 torch 2.3 + torchvision 0.18）。

## 目录结构

```
my-pytorch/
├── MLP/                          # PyTorch 入门 + MNIST 多层感知机 + 回归
│   ├── main.py                   # 入门教程：Tensor / autograd / nn.Module / 训练流程
│   ├── mniist-myself.py          # MNIST 手写 MLP（自实现）
│   ├── nmist_tr.py / mnist_test.py
│   └── predict_position.py       # 位置预测回归（sklearn 标准化）
├── CNN-MNIST/
│   ├── 卷积神经网络/              # MNIST 卷积神经网络 (CNN)
│   └── 全卷积神经网络/            # MNIST 全卷积神经网络 (FCN)
├── Detectable_Bottle/            # 塑料瓶目标检测（四套方案）
│   ├── V1-baseline/              # 检测基线
│   ├── V2-cascade-center-box/    # 级联中心框（YOLO 格式标签 + 混合精度）
│   ├── ResNet-Bottle-Det/        # ResNet 特征 + 检测头
│   └── ResNet-Bottle-Det-likeyolo/  # YOLO-like（自定义 Dataset / 损失函数）
├── Dataset_Templates/            # 数据集模板：MNIST / CIFAR / 塑料瓶
└── tool/                         # 图像/视频工具（视频抽帧等）
```

## 快速开始

```bash
conda activate pytorch          # 或使用你自己的虚拟环境
pip install -r requirements.txt

# MNIST MLP
python MLP/nmist_tr.py          # 训练
python MLP/mnist_test.py        # 测试

# MNIST CNN（卷积神经网络）
python "CNN-MNIST/卷积神经网络/train.py"
python "CNN-MNIST/卷积神经网络/test.py"

# 塑料瓶检测（YOLO-like 变体冒烟测试）
cd Detectable_Bottle/ResNet-Bottle-Det-likeyolo
python _smoke_test.py

# 塑料瓶检测训练 / 测试（各子项目目录内）
python train.py
python test.py
```

## 数据集说明

- **MNIST / CIFAR**：torchvision 会自动下载；本地已有副本放在 `Dataset_Templates/MNIST/`、`Dataset_Templates/CIFAR/`。
- **塑料瓶数据集**：原始 `dataset.zip`（约 748 MB）体积过大**不入库**，请自行放入对应子项目的 `dataset/` 目录后再训练/测试。

## 依赖安装

```bash
conda activate pytorch
pip install -r requirements.txt
```

完整依赖见 [`requirements.txt`](requirements.txt)。

## Credits / 致谢

- Author: jxb01 · 机器学习代码由作者本人编写；README、依赖清单与仓库工程化（.gitignore / 仓库整理）由 DeepSeek（DeepSeek Harness）辅助完成 — assistant by DeepSeek
- 作者：jxb01 · 机器学习代码为作者原创；README、requirements.txt、.gitignore 与上传整理由 DeepSeek（DeepSeek Harness）辅助完成 — assistant by DeepSeek
