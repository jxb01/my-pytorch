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


def compute_iou(box1, box2):
    """计算两个边界框的 IoU"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    if union_area == 0:
        return 0.0
    return inter_area / union_area


def cxcywh_to_xyxy(box, img_w, img_h):
    """归一化 [cx, cy, w, h] → 像素坐标 [x1, y1, x2, y2]"""
    cx, cy, w, h = box
    x1 = int((cx - w / 2) * img_w)
    y1 = int((cy - h / 2) * img_h)
    x2 = int((cx + w / 2) * img_w)
    y2 = int((cy + h / 2) * img_h)
    return [max(0, x1), max(0, y1), min(img_w, x2), min(img_h, y2)]


class Bottledataset(torch.utils.data.Dataset):
    def __init__(self, img_root, labels_root, transform=None):
        super().__init__()
        self.img_root = img_root
        self.labels_root = labels_root
        self.transform = transform
        self.img_filenames = [f for f in os.listdir(img_root) if f.endswith(('.jpg', '.png'))]
        self.labels_filenames = [f for f in os.listdir(labels_root) if f.endswith('.txt')]

    def __len__(self):
        return len(self.labels_filenames)

    def __getitem__(self, index):
        label_path = os.path.join(self.labels_root, self.labels_filenames[index])
        with open(label_path, "r") as f:
            parts = f.readline().strip().split()
        cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        labels = torch.tensor([cx, cy, w, h], dtype=torch.float32)

        img_name = os.path.splitext(self.labels_filenames[index])[0] + ".jpg"
        img_path = os.path.join(self.img_root, img_name)
        img = Image.open(img_path).convert('RGB')

        if self.transform:
            img = self.transform(img)
        return img, labels


def evaluate_model(model, dataloader, device):
    """在测试集上评估模型"""
    model.eval()
    total_iou = 0
    num_samples = 0

    with torch.no_grad():
        for img, labels in dataloader:
            img = img.to(device)
            labels = labels.to(device)
            outputs = model(img)

            for i in range(len(outputs)):
                pred = outputs[i].cpu().numpy()
                true = labels[i].cpu().numpy()
                pred_box = cxcywh_to_xyxy(pred, 256, 342)
                true_box = cxcywh_to_xyxy(true, 256, 342)
                iou = compute_iou(pred_box, true_box)
                total_iou += iou
                num_samples += 1

    return total_iou / num_samples if num_samples > 0 else 0


def visualize_random_predictions(model, dataset, device, num_samples=6):
    """随机选几张图片，画出预测框和真实框"""
    model.eval()

    # 反归一化
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    # 随机选索引
    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))

    # 创建画布
    cols = 3
    rows = (num_samples + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(12, 4 * rows))
    axes = axes.flatten() if num_samples > 1 else [axes]

    for idx, sample_idx in enumerate(indices):
        img_tensor, true_label = dataset[sample_idx]

        # 预测
        with torch.no_grad():
            pred_label = model(img_tensor.unsqueeze(0).to(device))[0].cpu().numpy()

        # 反归一化图像
        img_np = img_tensor.numpy()
        img_np = img_np * std[:, None, None] + mean[:, None, None]
        img_np = np.clip(img_np, 0, 1).transpose(1, 2, 0)

        axes[idx].imshow(img_np)

        # 转换坐标
        true_box = cxcywh_to_xyxy(true_label.numpy(), 256, 342)
        pred_box = cxcywh_to_xyxy(pred_label, 256, 342)

        # 绘制真实框（绿色）
        rect_true = patches.Rectangle(
            (true_box[0], true_box[1]),
            true_box[2] - true_box[0],
            true_box[3] - true_box[1],
            linewidth=2, edgecolor='g', facecolor='none', label='True'
        )
        axes[idx].add_patch(rect_true)

        # 绘制预测框（红色虚线）
        rect_pred = patches.Rectangle(
            (pred_box[0], pred_box[1]),
            pred_box[2] - pred_box[0],
            pred_box[3] - pred_box[1],
            linewidth=2, edgecolor='r', facecolor='none', linestyle='--', label='Pred'
        )
        axes[idx].add_patch(rect_pred)

        # 计算 IoU
        iou = compute_iou(pred_box, true_box)
        axes[idx].set_title(f'IoU: {iou:.3f}', fontsize=12)
        axes[idx].axis('off')

    # 只显示第一个图的图例
    axes[0].legend(loc='upper right')

    # 隐藏多余的子图
    for idx in range(num_samples, len(axes)):
        axes[idx].axis('off')

    plt.suptitle('Random Predictions (Green=True, Red=Pred)', fontsize=16)
    plt.tight_layout()
    plt.show()


def draw_prediction(frame, pred, color=(0, 255, 0), thickness=2):
    """在帧上绘制预测框"""
    h, w = frame.shape[:2]
    box = cxcywh_to_xyxy(pred, w, h)
    cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, thickness)
    cv2.putText(frame, "Bottle", (box[0], box[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return frame


def letterbox_image(image, target_size=(342, 256), color=(114, 114, 114)):
    """
    对摄像头帧进行 Letterbox 处理，返回处理后的图像和变换参数
    """
    h, w = image.shape[:2]
    target_h, target_w = target_size

    # 计算缩放比例，保持宽高比
    scale = min(target_h / h, target_w / w)
    new_w = int(w * scale)
    new_h = int(h * scale)

    # 缩放
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # 计算灰边偏移
    dw = (target_w - new_w) // 2
    dh = (target_h - new_h) // 2

    # 创建画布并填充
    padded = np.full((target_h, target_w, 3), color, dtype=np.uint8)
    padded[dh:dh + new_h, dw:dw + new_w] = resized

    return padded, scale, (dw, dh), (h, w)


def real_time_detection(model, transform, device):
    """实时摄像头检测（带 Letterbox）"""
    print("\n" + "=" * 50)
    print("按 'q' 退出，按 's' 截图")
    print("=" * 50)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("无法打开摄像头！")
        return

    model.eval()
    fps_counter = 0
    fps_start = cv2.getTickCount()
    fps_display = 0

    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 水平翻转
            frame = cv2.flip(frame, 1)
            original_frame = frame.copy()

            # 1. Letterbox 处理
            padded_frame, scale, (dw, dh), (orig_h, orig_w) = letterbox_image(
                frame, target_size=(342, 256)
            )

            # 2. 预处理（和训练时一致）
            frame_rgb = cv2.cvtColor(padded_frame, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb)
            frame_tensor = transform(frame_pil).unsqueeze(0).to(device)

            # 3. 推理（输出归一化坐标）
            output = model(frame_tensor)
            pred = output[0].cpu().numpy()
            pred = np.clip(pred, 0.0, 1.0)  # 防御性裁剪

            # 4. 将预测坐标从 Letterbox 空间还原到原始摄像头帧
            cx, cy, w, h = pred
            # 转成像素坐标（相对于 Letterbox 画布）
            x1_lb = int((cx - w / 2) * 256)
            y1_lb = int((cy - h / 2) * 342)
            x2_lb = int((cx + w / 2) * 256)
            y2_lb = int((cy + h / 2) * 342)

            # 减去灰边偏移
            x1 = x1_lb - dw
            y1 = y1_lb - dh
            x2 = x2_lb - dw
            y2 = y2_lb - dh

            # 除以缩放比例，还原到原始尺寸
            x1 = int(x1 / scale)
            y1 = int(y1 / scale)
            x2 = int(x2 / scale)
            y2 = int(y2 / scale)

            # 边界裁剪
            x1 = max(0, min(x1, orig_w))
            y1 = max(0, min(y1, orig_h))
            x2 = max(0, min(x2, orig_w))
            y2 = max(0, min(y2, orig_h))

            # 5. 在原始帧上绘制框
            cv2.rectangle(original_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(original_frame, "Bottle", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # 显示信息
            cv2.putText(original_frame, f"FPS: {fps_display:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

            cv2.imshow("Bottle Detection (Letterbox)", original_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                cv2.imwrite("screenshot_letterbox.jpg", original_frame)
                print("截图已保存")

            # FPS 计算放在循环末尾
            fps_counter += 1
            if fps_counter % 30 == 0:
                fps_end = cv2.getTickCount()
                fps_display = 30 / ((fps_end - fps_start) / cv2.getTickFrequency())
                fps_start = fps_end

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # 和训练相同的预处理
    transform = transforms.Compose([
        transforms.Resize((342, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 加载模型
    model = Net().to(device)
    if os.path.isfile("model.pth"):
        model.load_state_dict(torch.load("model.pth"))
        print("✅ Best model loaded!")
    else:
        print("❌ No model found!")
        exit()

    # 选择模式
    print("\n1 - 测试集评估")
    print("2 - 实时摄像头检测")
    print("3 - 两者都运行")
    print("4 - 随机可视化预测结果")
    choice = input("选择 (1/2/3/4): ").strip() or "3"

    if choice in ["1", "3"]:
        test_dataset = Bottledataset(
            transform=transform,
            img_root="dataset\\images\\valid",
            labels_root="dataset\\labels\\valid"
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True
        )
        print(f"Test batches: {len(test_loader)}")

        avg_iou = evaluate_model(model, test_loader, device)
        print(f"\n📊 Average IoU on test set: {avg_iou:.4f}")

    if choice == "4":
        # ✅ 随机可视化
        test_dataset = Bottledataset(
            transform=transform,
            img_root="dataset\\images\\valid",
            labels_root="dataset\\labels\\valid"
        )
        print(f"\n📷 随机选择 6 张图片进行可视化...")
        visualize_random_predictions(model, test_dataset, device, num_samples=6)


    if choice in ["2", "3"]:
        real_time_detection(model, transform, device)