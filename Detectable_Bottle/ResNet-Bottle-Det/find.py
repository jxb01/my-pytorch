
import cv2
import torch
import torch.nn as nn
from torchvision import transforms
from model import Net
import os
import numpy as np
from PIL import Image
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Net().to(device)
model.load_state_dict(torch.load("model.pth"))

# 打印第一层卷积的权重均值
for name, param in model.named_parameters():
    print(f"{name}: mean={param.mean().item():.4f}, std={param.std().item():.4f}")
    break  # 只打印第一层