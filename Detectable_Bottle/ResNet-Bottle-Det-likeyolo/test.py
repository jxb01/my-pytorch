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
from dataset import YOLODataset


def decode_target(target, grid_width=8, grid_height=11, conf_threshold=0.3):
    """
    从 YOLO 输出 (8, 11, 5) 中提取所有框
    target: (grid_width, grid_height, 5) 或 (batch, grid_width, grid_height, 5)
    """
    if target.dim() == 4:
        target = target[0]

    boxes = []
    for col in range(grid_width):
        for row in range(grid_height):
            cell = target[col, row]
            conf = cell[4].item()

            if conf < conf_threshold:
                continue

            offset_x = cell[0].item()
            offset_y = cell[1].item()
            w = cell[2].item()
            h = cell[3].item()

            grid_x = col / grid_width
            grid_y = row / grid_height
            cell_w = 1 / grid_width
            cell_h = 1 / grid_height

            center_x = grid_x + offset_x * cell_w
            center_y = grid_y + offset_y * cell_h

            boxes.append([center_x, center_y, w, h, conf])

    return boxes


def cxcywh_to_xyxy(box, img_w, img_h):
    """归一化 [cx, cy, w, h] → 像素坐标 [x1, y1, x2, y2]"""
    cx, cy, w, h = box[:4]
    x1 = int((cx - w / 2) * img_w)
    y1 = int((cy - h / 2) * img_h)
    x2 = int((cx + w / 2) * img_w)
    y2 = int((cy + h / 2) * img_h)
    return [max(0, x1), max(0, y1), min(img_w, x2), min(img_h, y2)]


def compute_iou_xyxy(box1, box2):
    """计算两个边界框的 IoU (角点格式 [x1, y1, x2, y2])"""
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


def nms(boxes, iou_threshold=0.5, img_w=256, img_h=342):
    """
    对多个框进行非极大值抑制

    Args:
        boxes: list of [cx, cy, w, h, conf] (归一化坐标)
        iou_threshold: IoU 阈值
        img_w, img_h: 用于计算像素坐标

    Returns:
        keep: 保留的框列表
    """
    if len(boxes) == 0:
        return []

    # 按置信度降序排列
    boxes = sorted(boxes, key=lambda x: x[4], reverse=True)
    keep = []

    while boxes:
        # 保留置信度最高的框
        best = boxes.pop(0)
        keep.append(best)

        # 移除与当前框 IoU 过高的框
        remaining = []
        best_xyxy = cxcywh_to_xyxy(best, img_w, img_h)
        for b in boxes:
            b_xyxy = cxcywh_to_xyxy(b, img_w, img_h)
            iou = compute_iou_xyxy(best_xyxy, b_xyxy)
            if iou < iou_threshold:
                remaining.append(b)
        boxes = remaining

    return keep


def evaluate_model(model, dataloader, device, conf_threshold=0.3, iou_threshold=0.5):
    """在测试集上评估模型（带 NMS）"""
    model.eval()
    total_iou = 0
    num_samples = 0

    with torch.no_grad():
        for img, target in dataloader:
            img = img.to(device)
            target = target.to(device)
            outputs = model(img)  # (batch, 8, 11, 5)

            for i in range(outputs.shape[0]):
                pred_boxes = decode_target(outputs[i], conf_threshold=conf_threshold)
                pred_boxes = nms(pred_boxes, iou_threshold)

                true_boxes = decode_target(target[i], conf_threshold=conf_threshold)
                true_boxes = nms(true_boxes, iou_threshold)

                if len(pred_boxes) == 0 or len(true_boxes) == 0:
                    continue

                # 取置信度最高的预测框和真实框比较
                best_pred = max(pred_boxes, key=lambda x: x[4])
                best_true = max(true_boxes, key=lambda x: x[4])

                pred_xyxy = cxcywh_to_xyxy(best_pred, 256, 342)
                true_xyxy = cxcywh_to_xyxy(best_true, 256, 342)
                iou = compute_iou_xyxy(pred_xyxy, true_xyxy)
                total_iou += iou
                num_samples += 1

    return total_iou / num_samples if num_samples > 0 else 0


def visualize_random_predictions(model, dataset, device, num_samples=6,
                                 conf_threshold=0.3, iou_threshold=0.5):
    """随机选几张图片，画出预测框和真实框（带 NMS）"""
    model.eval()

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))

    cols = 3
    rows = (num_samples + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(12, 4 * rows))
    axes = axes.flatten() if num_samples > 1 else [axes]

    for idx, sample_idx in enumerate(indices):
        img_tensor, target_tensor = dataset[sample_idx]

        with torch.no_grad():
            output = model(img_tensor.unsqueeze(0).to(device))[0]
            pred_boxes = decode_target(output, conf_threshold=conf_threshold)
            pred_boxes = nms(pred_boxes, iou_threshold)

            true_boxes = decode_target(target_tensor, conf_threshold=conf_threshold)
            true_boxes = nms(true_boxes, iou_threshold)

        # 反归一化图像（tensor 是 ImageNet 归一化的 RGB）
        img_np = img_tensor.numpy()
        img_np = img_np * std[:, None, None] + mean[:, None, None]
        img_np = np.clip(img_np, 0, 1).transpose(1, 2, 0)  # RGB 直接显示

        axes[idx].imshow(img_np)

        # 真实框（绿色）
        for box in true_boxes:
            xyxy = cxcywh_to_xyxy(box, 256, 342)
            rect = patches.Rectangle(
                (xyxy[0], xyxy[1]), xyxy[2] - xyxy[0], xyxy[3] - xyxy[1],
                linewidth=2, edgecolor='g', facecolor='none'
            )
            axes[idx].add_patch(rect)

        # 预测框（红色虚线）
        for box in pred_boxes:
            xyxy = cxcywh_to_xyxy(box, 256, 342)
            rect = patches.Rectangle(
                (xyxy[0], xyxy[1]), xyxy[2] - xyxy[0], xyxy[3] - xyxy[1],
                linewidth=2, edgecolor='r', facecolor='none', linestyle='--'
            )
            axes[idx].add_patch(rect)
            axes[idx].text(xyxy[0], xyxy[1] - 5, f'{box[4]:.2f}',
                           color='red', fontsize=8, backgroundcolor='white')

        axes[idx].set_title(f'Pred: {len(pred_boxes)} / True: {len(true_boxes)}')
        axes[idx].axis('off')

    for idx in range(num_samples, len(axes)):
        axes[idx].axis('off')

    plt.suptitle(f'YOLO Predictions (NMS IoU={iou_threshold}, Conf={conf_threshold})', fontsize=16)
    plt.tight_layout()
    plt.show()


def letterbox_image(image, target_size=(342, 256), color=(114, 114, 114)):
    """对摄像头帧进行 Letterbox 处理"""
    h, w = image.shape[:2]
    target_h, target_w = target_size

    scale = min(target_h / h, target_w / w)
    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    dw = (target_w - new_w) // 2
    dh = (target_h - new_h) // 2

    padded = np.full((target_h, target_w, 3), color, dtype=np.uint8)
    padded[dh:dh + new_h, dw:dw + new_w] = resized

    return padded, scale, (dw, dh), (h, w)


def real_time_detection(model, transform, device, conf_threshold=0.3, iou_threshold=0.5):
    """实时摄像头检测（带 NMS）"""
    print("\n" + "=" * 50)
    print(f"置信度阈值: {conf_threshold}, NMS IoU: {iou_threshold}")
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

            frame = cv2.flip(frame, 1)
            original_frame = frame.copy()

            # Letterbox 处理
            padded_frame, scale, (dw, dh), (orig_h, orig_w) = letterbox_image(
                frame, target_size=(342, 256)
            )

            # 预处理
            frame_rgb = cv2.cvtColor(padded_frame, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb)
            frame_tensor = transform(frame_pil).unsqueeze(0).to(device)

            # 推理
            output = model(frame_tensor)[0]  # (8, 11, 5)

            # 解码 + NMS
            boxes = decode_target(output, conf_threshold=conf_threshold)
            boxes = nms(boxes, iou_threshold, img_w=256, img_h=342)

            # 绘制框
            for box in boxes:
                cx, cy, w, h, conf = box
                # 从 Letterbox 空间还原到原始帧
                x1_lb = int((cx - w / 2) * 256)
                y1_lb = int((cy - h / 2) * 342)
                x2_lb = int((cx + w / 2) * 256)
                y2_lb = int((cy + h / 2) * 342)

                x1 = int((x1_lb - dw) / scale)
                y1 = int((y1_lb - dh) / scale)
                x2 = int((x2_lb - dw) / scale)
                y2 = int((y2_lb - dh) / scale)

                x1 = max(0, min(x1, orig_w))
                y1 = max(0, min(y1, orig_h))
                x2 = max(0, min(x2, orig_w))
                y2 = max(0, min(y2, orig_h))

                cv2.rectangle(original_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(original_frame, f'Bottle {conf:.2f}', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cv2.putText(original_frame, f"FPS: {fps_display:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            cv2.putText(original_frame, f"Objects: {len(boxes)}", (10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

            cv2.imshow("YOLO Detection (NMS)", original_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                cv2.imwrite("screenshot.jpg", original_frame)
                print("截图已保存")

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

    transform = transforms.Compose([
        transforms.Resize((342, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 加载模型
    model = Net().to(device)
    if os.path.isfile("model_long.pth"):
        model.load_state_dict(torch.load("model_long.pth", map_location=device))
        model.eval()
        print("✅ 模型加载成功！")
    else:
        print("❌ 未找到 model.pth！")
        exit()

    print("\n1 - 测试集评估")
    print("2 - 实时摄像头检测")
    print("3 - 随机可视化预测结果")
    print("4 - 全部运行")
    choice = input("选择 (1/2/3/4): ").strip() or "3"

    # 可调参数
    CONF_THRESHOLD = 0.3
    NMS_IOU = 0.5

    if choice in ["1", "4"]:
        # 评估用验证集（不再用训练集, 否则指标虚高）
        test_dataset = YOLODataset(
            image_dir="./dataset/images/valid",
            label_dir="./dataset/labels/valid",
            img_width=256,
            img_height=342,
            grid_height=11,
            grid_width=8,
            mosaic_prob=0,
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True
        )
        avg_iou = evaluate_model(model, test_loader, device,
                                 conf_threshold=CONF_THRESHOLD,
                                 iou_threshold=NMS_IOU)
        print(f"\n📊 Average IoU on test set: {avg_iou:.4f}")

    if choice in ["2", "4"]:
        real_time_detection(model, transform, device,
                            conf_threshold=CONF_THRESHOLD,
                            iou_threshold=NMS_IOU)

    if choice in ["3", "4"]:
        test_dataset = YOLODataset(
            image_dir="./dataset/images/valid",
            label_dir="./dataset/labels/valid",
            img_width=256,
            img_height=342,
            grid_height=11,
            grid_width=8,
            mosaic_prob=0,
        )
        visualize_random_predictions(model, test_dataset, device, num_samples=6,
                                     conf_threshold=CONF_THRESHOLD,
                                     iou_threshold=NMS_IOU)