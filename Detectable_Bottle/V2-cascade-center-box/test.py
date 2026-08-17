import cv2
import torch
import torch.nn as nn
from torchvision import transforms
from model import Net
import os
import numpy as np
from torchvision.transforms import v2


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
    def __init__(self, img_root, labels_root, transform):
        super(Bottledataset, self).__init__()
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

        cx = float(parts[1])
        cy = float(parts[2])
        width = float(parts[3])
        height = float(parts[4])
        labels = torch.tensor([cx, cy, width, height], dtype=torch.float32)

        img_name = os.path.splitext(self.labels_filenames[index])[0] + ".jpg"
        img_path = os.path.join(self.img_root, img_name)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

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

                pred_box = cxcywh_to_xyxy(pred, 224, 224)
                true_box = cxcywh_to_xyxy(true, 224, 224)

                iou = compute_iou(pred_box, true_box)
                total_iou += iou
                num_samples += 1

    return total_iou / num_samples if num_samples > 0 else 0


def preprocess_frame(frame, transform, device):
    """预处理摄像头帧"""
    # BGR → RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 应用变换
    frame_tensor = transform(frame_rgb)

    # 添加 batch 维度
    frame_tensor = frame_tensor.unsqueeze(0).to(device)

    return frame_tensor


def draw_prediction(frame, pred, color=(0, 255, 0), thickness=2):
    """在帧上绘制预测框"""
    h, w = frame.shape[:2]

    # 归一化坐标 → 像素坐标
    box = cxcywh_to_xyxy(pred, w, h)

    # 绘制矩形
    cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, thickness)

    # 添加标签
    label = f"Bottle"
    cv2.putText(frame, label, (box[0], box[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return frame


def real_time_detection(model, transform, device):
    """实时摄像头检测"""
    print("\n" + "=" * 50)
    print("📷 启动实时检测")
    print("按 'q' 退出")
    print("=" * 50)

    # 打开摄像头
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ 无法打开摄像头！")
        return

    # 设置分辨率
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # FPS 计算
    fps_counter = 0
    fps_start_time = cv2.getTickCount()
    fps_display = 0

    model.eval()

    with torch.no_grad():
        while True:
            # 读取帧
            ret, frame = cap.read()
            if not ret:
                print("❌ 无法读取摄像头帧")
                break

            # 镜像翻转（更自然）
            frame = cv2.flip(frame, 1)

            # 预处理
            input_tensor = preprocess_frame(frame, transform, device)

            # 推理
            output = model(input_tensor)
            pred = output[0].cpu().numpy()

            # 绘制预测框
            frame = draw_prediction(frame, pred, color=(0, 255, 0), thickness=2)

            # 显示预测值
            cx, cy, w, h = pred
            info_text = f"cx={cx:.3f} cy={cy:.3f} w={w:.3f} h={h:.3f}"
            cv2.putText(frame, info_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # FPS 计算
            fps_counter += 1
            if fps_counter % 30 == 0:
                fps_end_time = cv2.getTickCount()
                time_elapsed = (fps_end_time - fps_start_time) / cv2.getTickFrequency()
                fps_display = 30 / time_elapsed
                fps_start_time = fps_end_time

            # 显示 FPS
            cv2.putText(frame, f"FPS: {fps_display:.1f}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

            # 显示
            cv2.imshow("Bottle Detection - Real Time", frame)

            # 按键处理
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("退出实时检测")
                break
            elif key == ord('s'):
                # 保存截图
                cv2.imwrite("detection_screenshot.jpg", frame)
                print("📸 截图已保存")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("device is", device)

    # ✅ 和训练相同的预处理
    transform1 = v2.Compose([
        v2.ToImage(),
        v2.Resize((224, 224)),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 加载模型
    model = Net().to(device)
    if os.path.isfile("best_model.pth"):
        model.load_state_dict(torch.load("best_model.pth"))
        print("✅ Best model loaded!")
    elif os.path.isfile("model.pth"):
        model.load_state_dict(torch.load("model.pth"))
        print("✅ Model loaded!")
    else:
        print("❌ No model found!")
        exit()

    # ========== 选择模式 ==========
    print("\n选择运行模式:")
    print("1 - 测试集评估")
    print("2 - 实时摄像头检测")
    print("3 - 两者都运行")

    choice = input("请输入选择 (1/2/3，默认 3): ").strip() or "3"

    if choice in ["1", "3"]:
        # 测试集评估
        test_dataset = Bottledataset(
            transform=transform1,
            img_root="data\\test\\images",
            labels_root="data\\test\\labels"
        )
        test_loader = torch.utils.data.DataLoader(
            dataset=test_dataset, batch_size=64, shuffle=False,
            num_workers=4, pin_memory=True
        )
        print(f"Test batches: {len(test_loader)}")

        avg_iou = evaluate_model(model, test_loader, device)
        print(f"\n📊 Average IoU on test set: {avg_iou:.4f}")

        # 可视化几个样本
        print("\n📷 显示测试样本...")
        model.eval()
        with torch.no_grad():
            for i, (img, labels) in enumerate(test_loader):
                if i >= 2:
                    break
                img = img.to(device)
                outputs = model(img)

                for j in range(min(4, len(outputs))):
                    pred = outputs[j].cpu().numpy()
                    true = labels[j].cpu().numpy()
                    print(f"  Sample {j}: Pred={pred}, True={true}")

    if choice in ["2", "3"]:
        # 实时摄像头检测
        real_time_detection(model, transform1, device)