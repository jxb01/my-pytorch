import os
import cv2
import torch
import random
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
from matplotlib import pyplot as plt
import matplotlib.patches as patches

# ImageNet 归一化（预训练 ResNet 要求: RGB + mean/std）
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def normalize(img_tensor):
    """0~1 的 RGB tensor → ImageNet 归一化"""
    return (img_tensor - IMAGENET_MEAN) / IMAGENET_STD


def denormalize(img_tensor):
    """ImageNet 归一化 → 0~1 的 RGB tensor（仅用于可视化）"""
    return img_tensor * IMAGENET_STD + IMAGENET_MEAN


class YOLODataset(Dataset):
    """
    YOLO 数据集（精简版）
    - 图像以 RGB + ImageNet 归一化输入（预训练 ResNet 要求）
    - 每个网格只预测 1 个框
    - 没有类别
    - 输出格式: (8, 11, 5)  [宽度, 高度, 通道]
    - 标注格式：每行 `type x y w h`
    """

    def __init__(
            self,
            image_dir,
            label_dir,
            img_width=256,
            img_height=342,
            grid_height=11,
            grid_width=8,
            mosaic_prob=0.5,
    ):
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)
        self.img_width = img_width
        self.img_height = img_height
        self.grid_height = grid_height
        self.grid_width = grid_width
        self.mosaic_prob = mosaic_prob
        self.output_channels = 5

        # 获取所有图片文件
        self.image_files = []
        for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
            self.image_files.extend(list(self.image_dir.glob(f'*{ext}')))
        self.image_files = [f for f in self.image_files if
                            f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']]

        print(f"找到 {len(self.image_files)} 张图片")
        print(f"输出格式: ({grid_width}, {grid_height}, 5)")
        print(f"Mosaic概率: {mosaic_prob}")
        print(f"图像格式: RGB + ImageNet 归一化")

    def __len__(self):
        return len(self.image_files)

    def load_image(self, img_path):
        """读取图片，返回 BGR 格式"""
        img = cv2.imread(str(img_path))
        return img

    def load_labels(self, img_path):
        """
        加载标注，格式：type x y w h（带类别）
        只提取 x, y, w, h
        """
        label_path = self.label_dir / (img_path.stem + '.txt')
        boxes = []
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) == 5:
                        cls = int(float(parts[0]))
                        x, y, w, h = map(float, parts[1:5])
                        boxes.append([x, y, w, h])
                    elif len(parts) == 4:
                        x, y, w, h = map(float, parts)
                        boxes.append([x, y, w, h])
        return boxes

    def boxes_to_target(self, boxes):
        target = torch.zeros(self.grid_width, self.grid_height, self.output_channels)

        for box in boxes:
            x, y, w, h = box

            grid_x = int(x * self.grid_width)
            grid_y = int(y * self.grid_height)

            if grid_x >= self.grid_width or grid_y >= self.grid_height:
                continue

            offset_x = x * self.grid_width - grid_x
            offset_y = y * self.grid_height - grid_y

            target[grid_x, grid_y, 0] = offset_x
            target[grid_x, grid_y, 1] = offset_y
            target[grid_x, grid_y, 2] = w
            target[grid_x, grid_y, 3] = h
            target[grid_x, grid_y, 4] = 1.0

        return target

    # ==================== Mosaic 核心逻辑 ====================

    def adjust_boxes_for_mosaic(self, boxes, orig_w, orig_h, target_w, target_h, offset_x, offset_y):
        adjusted = []

        for box in boxes:
            x, y, w, h = box

            px = x * orig_w
            py = y * orig_h
            pw = w * orig_w
            ph = h * orig_h

            scale_x = target_w / orig_w
            scale_y = target_h / orig_h

            new_px = px * scale_x + offset_x
            new_py = py * scale_y + offset_y
            new_pw = pw * scale_x
            new_ph = ph * scale_y

            new_x = new_px / self.img_width
            new_y = new_py / self.img_height
            new_w = new_pw / self.img_width
            new_h = new_ph / self.img_height

            x1 = max(0, new_x - new_w / 2)
            y1 = max(0, new_y - new_h / 2)
            x2 = min(1, new_x + new_w / 2)
            y2 = min(1, new_y + new_h / 2)

            new_w = x2 - x1
            new_h = y2 - y1
            new_x = (x1 + x2) / 2
            new_y = (y1 + y2) / 2

            if new_w < 0.01 or new_h < 0.01:
                continue

            adjusted.append([new_x, new_y, new_w, new_h])

        return adjusted

    def generate_mosaic(self, indices):
        """从4张图片生成1张Mosaic（全程 BGR）"""
        H, W = self.img_height, self.img_width

        cx = random.randint(W // 4, 3 * W // 4)
        cy = random.randint(H // 4, 3 * H // 4)

        mosaic_img = np.zeros((H, W, 3), dtype=np.uint8)
        all_boxes = []

        quadrants = [
            {'x_start': 0, 'y_start': 0, 'x_end': cx, 'y_end': cy},
            {'x_start': cx, 'y_start': 0, 'x_end': W, 'y_end': cy},
            {'x_start': 0, 'y_start': cy, 'x_end': cx, 'y_end': H},
            {'x_start': cx, 'y_start': cy, 'x_end': W, 'y_end': H},
        ]

        for i, idx in enumerate(indices):
            img_path = self.image_files[idx]
            img = self.load_image(img_path)  # BGR

            if img is None:
                continue

            h, w = img.shape[:2]
            boxes = self.load_labels(img_path)

            quad = quadrants[i]
            target_w = quad['x_end'] - quad['x_start']
            target_h = quad['y_end'] - quad['y_start']

            if target_w < 10 or target_h < 10:
                continue

            scale = min(target_w / w, target_h / h)
            new_w = int(w * scale)
            new_h = int(h * scale)

            # BGR 格式，不转换
            resized = cv2.resize(img, (new_w, new_h))

            x_offset = (target_w - new_w) // 2
            y_offset = (target_h - new_h) // 2

            actual_x = quad['x_start'] + x_offset
            actual_y = quad['y_start'] + y_offset

            mosaic_img[
                actual_y: actual_y + new_h,
                actual_x: actual_x + new_w
            ] = resized

            if len(boxes) > 0:
                adjusted = self.adjust_boxes_for_mosaic(
                    boxes,
                    orig_w=w,
                    orig_h=h,
                    target_w=new_w,
                    target_h=new_h,
                    offset_x=actual_x,
                    offset_y=actual_y
                )
                all_boxes.extend(adjusted)

        return mosaic_img, all_boxes

    # ==================== 核心：__getitem__ ====================

    def __getitem__(self, idx):
        if random.random() < self.mosaic_prob:
            other_indices = random.sample(range(len(self.image_files)), 3)
            indices = [idx] + other_indices

            mosaic_img, boxes = self.generate_mosaic(indices)

            if len(boxes) == 0:
                return self._load_normal(idx)

            # BGR → RGB, 归一化到 0~1, 再过 ImageNet 归一化
            img = cv2.cvtColor(mosaic_img, cv2.COLOR_BGR2RGB)
            img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            img = normalize(img)
            target = self.boxes_to_target(boxes)
            return img, target

        else:
            return self._load_normal(idx)

    def _load_normal(self, idx):
        """加载原始图片（BGR 格式）"""
        img_path = self.image_files[idx]
        img = self.load_image(img_path)  # BGR

        if img is None:
            img = np.zeros((self.img_height, self.img_width, 3), dtype=np.uint8)
            boxes = []
        else:
            img = cv2.resize(img, (self.img_width, self.img_height))
            # ⭐ 不转换，保持 BGR
            boxes = self.load_labels(img_path)

        # BGR → RGB, 归一化到 0~1, 再过 ImageNet 归一化
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        img = normalize(img)
        target = self.boxes_to_target(boxes)

        return img, target


# ==================== 可视化函数（BGR → RGB 仅用于显示）====================

def draw_boxes_on_image(img_tensor, target_tensor, img_width=256, img_height=342):
    """
    在图片上绘制目标框
    输入是 BGR Tensor，显示时转成 RGB
    """
    # 1. (C, H, W) → (H, W, C); 反归一化后是 RGB, 直接显示
    img = denormalize(img_tensor).numpy().copy()
    img = np.clip(img, 0, 1)
    img = np.transpose(img, (1, 2, 0))

    fig, ax = plt.subplots(1, figsize=(10, 14))
    ax.imshow(img)

    grid_width, grid_height = target_tensor.shape[0], target_tensor.shape[1]

    for col in range(grid_width):
        for row in range(grid_height):
            cell = target_tensor[col, row, :]

            if cell[4] > 0.5:
                offset_x = cell[0].item()
                offset_y = cell[1].item()
                w = cell[2].item()
                h = cell[3].item()

                grid_x = col / grid_width
                grid_y = row / grid_height
                cell_width = 1 / grid_width
                cell_height = 1 / grid_height

                center_x = grid_x + offset_x * cell_width
                center_y = grid_y + offset_y * cell_height

                x1 = center_x - w / 2
                y1 = center_y - h / 2

                rect = patches.Rectangle(
                    (x1 * img_width, y1 * img_height),
                    w * img_width,
                    h * img_height,
                    linewidth=2,
                    edgecolor='red',
                    facecolor='none'
                )
                ax.add_patch(rect)

                ax.text(
                    x1 * img_width,
                    y1 * img_height - 5,
                    f'{cell[4].item():.2f}',
                    color='red',
                    fontsize=8,
                    backgroundcolor='white'
                )

    ax.set_title(f'检测框数量: {torch.count_nonzero(target_tensor[:, :, 4]).item()}')
    ax.axis('off')
    return fig


# ==================== 使用示例 ====================

if __name__ == "__main__":
    from torch.utils.data import DataLoader

    # 设置 matplotlib 中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False

    dataset = YOLODataset(
        image_dir="./dataset/images/train",
        label_dir="./dataset/labels/train",
        img_width=256,
        img_height=342,
        grid_height=11,
        grid_width=8,
        mosaic_prob=0.5,
    )

    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    for images, targets in dataloader:
        print(f"图片: {images.shape}")
        print(f"目标: {targets.shape}")

        img = images[0]
        target = targets[0]
        print(target.shape)

        num_boxes = torch.count_nonzero(target[:, :, 4]).item()
        print(f"该图中有 {num_boxes} 个框")

        fig = draw_boxes_on_image(img, target, img_width=256, img_height=342)
        plt.show()
        break