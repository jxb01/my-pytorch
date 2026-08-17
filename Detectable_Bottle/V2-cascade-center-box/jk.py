import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.transforms import v2
from torchvision.tv_tensors import BoundingBoxes
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os


def test_v2_transforms():
    """全面测试 v2 转换的正确性"""

    print("=" * 70)
    print("v2 转换验证测试套件")
    print("=" * 70)

    # ==================== 测试 1: 基本张量转换 ====================
    print("\n📋 测试 1: 基本张量转换")
    print("-" * 50)

    # 模拟一张非正方形图像 (480x640)
    img_tensor = torch.randint(0, 256, (3, 480, 640), dtype=torch.uint8)

    transform_basic = v2.Compose([
        v2.Resize((224, 224)),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    try:
        result = transform_basic(img_tensor)
        print(f"✓ 张量转换成功")
        print(f"  输入形状: {img_tensor.shape}")
        print(f"  输出形状: {result.shape}")
        print(f"  输出范围: [{result.min():.3f}, {result.max():.3f}]")
        assert result.shape == (3, 224, 224), "形状错误"
    except Exception as e:
        print(f"❌ 张量转换失败: {e}")

    # ==================== 测试 2: PIL 图像转换 ====================
    print("\n📋 测试 2: PIL 图像转换")
    print("-" * 50)

    # 创建测试 PIL 图像
    pil_img = Image.fromarray(np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8))

    transform_pil = v2.Compose([
        v2.ToImage(),
        v2.Resize((224, 224)),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    try:
        result = transform_pil(pil_img)
        print(f"✓ PIL 图像转换成功")
        print(f"  输入大小: {pil_img.size}")
        print(f"  输出形状: {result.shape}")
        assert result.shape == (3, 224, 224), "形状错误"
    except Exception as e:
        print(f"❌ PIL 图像转换失败: {e}")

    # ==================== 测试 3: XYXY 格式 BoundingBoxes ====================
    print("\n📋 测试 3: XYXY 格式 BoundingBoxes 转换")
    print("-" * 50)

    # 创建测试数据
    img = torch.randint(0, 256, (3, 480, 640), dtype=torch.uint8)
    # XYXY 格式：左上角 (100,100)，右下角 (300,300)
    boxes = BoundingBoxes(
        torch.tensor([[100, 100, 300, 300]], dtype=torch.float32),
        format="XYXY",
        canvas_size=(480, 640)
    )

    transform_xyxy = v2.Compose([
        v2.Resize((224, 224)),
    ])

    try:
        img_out, boxes_out = transform_xyxy(img, boxes)
        print(f"✓ XYXY BoundingBoxes 转换成功")
        print(f"  原始框 (XYXY): {boxes.data.tolist()}")
        print(f"  变换后框 (XYXY): {boxes_out.data.tolist()}")

        # 手动计算期望值
        scale_h = 224 / 480
        scale_w = 224 / 640
        expected = [
            100 * scale_w,  # x1
            100 * scale_h,  # y1
            300 * scale_w,  # x2
            300 * scale_h,  # y2
        ]
        print(f"  期望值 (手工计算): {expected}")

        # 检查误差
        actual = boxes_out.data[0].tolist()
        for i, (act, exp) in enumerate(zip(actual, expected)):
            error = abs(act - exp)
            status = "✓" if error < 1.0 else "⚠️"
            print(f"  {status} 坐标[{i}]: 实际={act:.2f}, 期望={exp:.2f}, 误差={error:.4f}")

    except Exception as e:
        print(f"❌ XYXY BoundingBoxes 转换失败: {e}")

    # ==================== 测试 4: CXCYWH 格式 BoundingBoxes ====================
    print("\n📋 测试 4: CXCYWH 格式 BoundingBoxes 转换")
    print("-" * 50)

    # CXCYWH 格式：中心 (0.5, 0.5)，宽高 (0.3, 0.3)
    boxes_cxcywh = BoundingBoxes(
        torch.tensor([[0.5, 0.5, 0.3, 0.3]], dtype=torch.float32),
        format="CXCYWH",
        canvas_size=(480, 640)
    )

    transform_cxcywh = v2.Compose([
        v2.Resize((224, 224)),
    ])

    try:
        img_out, boxes_out = transform_cxcywh(img.clone(), boxes_cxcywh)
        print(f"✓ CXCYWH BoundingBoxes 转换成功")
        print(f"  原始框 (CXCYWH): {boxes_cxcywh.data.tolist()}")
        print(f"  变换后框 (CXCYWH): {boxes_out.data.tolist()}")

        # 检查宽高是否仍然为正
        w, h = boxes_out.data[0, 2], boxes_out.data[0, 3]
        if w > 0 and h > 0:
            print(f"  ✓ 宽高为正: w={w:.4f}, h={h:.4f}")
        else:
            print(f"  ❌ 宽高异常: w={w:.4f}, h={h:.4f}")

        # 检查中心点是否在合理范围
        cx, cy = boxes_out.data[0, 0], boxes_out.data[0, 1]
        if 0 <= cx <= 1 and 0 <= cy <= 1:
            print(f"  ✓ 中心点在 [0,1] 内: cx={cx:.4f}, cy={cy:.4f}")
        else:
            print(f"  ⚠️ 中心点超出范围: cx={cx:.4f}, cy={cy:.4f}")

    except Exception as e:
        print(f"❌ CXCYWH BoundingBoxes 转换失败: {e}")

    # ==================== 测试 5: 完整数据增强流程 ====================
    print("\n📋 测试 5: 完整数据增强流程（模拟你的原始代码）")
    print("-" * 50)

    # 这是你原始代码的 transform
    transform_full = v2.Compose([
        v2.ToImage(),
        v2.Resize((224, 224)),
        v2.RandomHorizontalFlip(p=0.5),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 测试多次以确保随机增强也没问题
    for test_idx in range(5):
        # 创建随机 PIL 图像和标签
        pil_test = Image.fromarray(np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8))

        boxes_test = BoundingBoxes(
            torch.tensor([[0.5, 0.5, 0.3, 0.3]], dtype=torch.float32),
            format="CXCYWH",
            canvas_size=(480, 640)
        )

        try:
            img_out, boxes_out = transform_full(pil_test, boxes_test)

            # 检查输出
            checks = []

            # 1. 形状检查
            checks.append(("形状正确", img_out.shape == (3, 224, 224)))

            # 2. 标签范围检查
            box_data = boxes_out.data[0]
            checks.append(("标签在 [0,1]", (box_data >= 0).all() and (box_data <= 1).all()))

            # 3. 宽高检查
            checks.append(("宽高为正", box_data[2] > 0 and box_data[3] > 0))

            # 4. 中心点检查
            checks.append(("中心点合理", 0 <= box_data[0] <= 1 and 0 <= box_data[1] <= 1))

            all_passed = all(passed for _, passed in checks)

            if all_passed:
                print(f"  测试 {test_idx + 1}: ✓ 全部通过")
            else:
                print(f"  测试 {test_idx + 1}: ❌ 有问题")
                for name, passed in checks:
                    print(f"    {'✓' if passed else '❌'} {name}")

        except Exception as e:
            print(f"  测试 {test_idx + 1}: ❌ 异常: {e}")

    # ==================== 测试 6: v1 vs v2 一致性 ====================
    print("\n📋 测试 6: v1 和 v2 对图像处理的一致性")
    print("-" * 50)

    # 固定随机种子
    torch.manual_seed(42)
    np.random.seed(42)

    # 创建相同的输入
    pil_img1 = Image.fromarray(np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8))
    pil_img2 = pil_img1.copy()

    # v1 transform
    transform_v1 = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # v2 transform
    transform_v2 = v2.Compose([
        v2.ToImage(),
        v2.Resize((224, 224)),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    result_v1 = transform_v1(pil_img1)
    result_v2 = transform_v2(pil_img2)

    diff = (result_v1 - result_v2).abs().max().item()
    mean_diff = (result_v1 - result_v2).abs().mean().item()

    print(f"  v1 输出范围: [{result_v1.min():.4f}, {result_v1.max():.4f}]")
    print(f"  v2 输出范围: [{result_v2.min():.4f}, {result_v2.max():.4f}]")
    print(f"  最大差异: {diff:.6f}")
    print(f"  平均差异: {mean_diff:.6f}")

    if diff < 0.01:
        print(f"  ✓ v1 和 v2 结果几乎一致（差异 < 0.01）")
    else:
        print(f"  ⚠️ v1 和 v2 存在较大差异")

    # ==================== 测试 7: 可视化验证 ====================
    print("\n📋 测试 7: 可视化验证（检查框是否正确）")
    print("-" * 50)

    # 创建白色背景图像
    img_vis = Image.new('RGB', (640, 480), color='white')

    # 定义一个框：左上角 (100,100)，右下角 (300,300)
    boxes_vis = BoundingBoxes(
        torch.tensor([[100, 100, 300, 300]], dtype=torch.float32),
        format="XYXY",
        canvas_size=(480, 640)
    )

    # 应用 Resize
    transform_vis = v2.Compose([
        v2.ToImage(),
        v2.Resize((224, 224)),
    ])

    img_out_vis, boxes_out_vis = transform_vis(img_vis, boxes_vis)

    # 转换回 PIL 用于显示
    img_display = (img_out_vis.permute(1, 2, 0).numpy()).astype(np.uint8)

    # 绘制框
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # 原始图像 + 框
    ax1.imshow(img_vis)
    rect1 = patches.Rectangle(
        (100, 100), 200, 200,
        linewidth=2, edgecolor='r', facecolor='none'
    )
    ax1.add_patch(rect1)
    ax1.set_title(f'原始图像 (640×480)\n框: [100,100,300,300]')
    ax1.axis('off')

    # 变换后图像 + 框
    ax2.imshow(img_display)
    box_data = boxes_out_vis.data[0]
    x1, y1, x2, y2 = box_data.tolist()
    rect2 = patches.Rectangle(
        (x1, y1), x2 - x1, y2 - y1,
        linewidth=2, edgecolor='r', facecolor='none'
    )
    ax2.add_patch(rect2)
    ax2.set_title(f'Resize 后 (224×224)\n框: [{x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f}]')
    ax2.axis('off')

    plt.suptitle('v2 BoundingBoxes Resize 验证', fontsize=14)
    plt.tight_layout()
    plt.show()

    print(f"  原始框: [100, 100, 300, 300]")
    print(f"  变换后: [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]")
    print(f"  期望值: [{100 * 224 / 640:.1f}, {100 * 224 / 480:.1f}, {300 * 224 / 640:.1f}, {300 * 224 / 480:.1f}]")
    print(f"  ✓ 可视化窗口已显示，请检查框的位置是否正确")

    # ==================== 总结 ====================
    print("\n" + "=" * 70)
    print("验证完成！")
    print("=" * 70)
    print("\n如果所有测试都通过 ✓，说明 v2 转换工作正常")
    print("如果出现 ❌ 或 ⚠️，请检查对应的测试")


def test_real_data():
    """测试你的真实数据"""
    print("\n" + "=" * 70)
    print("测试你的真实数据集")
    print("=" * 70)

    # 检查你的数据目录
    img_root = "data\\train\\images"
    labels_root = "data\\train\\labels"

    if not os.path.exists(img_root) or not os.path.exists(labels_root):
        print("❌ 数据目录不存在，跳过真实数据测试")
        return

    # 获取第一个文件
    label_files = [f for f in os.listdir(labels_root) if f.endswith('.txt')]
    if not label_files:
        print("❌ 没有找到标签文件")
        return

    sample_label = label_files[0]
    label_path = os.path.join(labels_root, sample_label)

    # 读取标签
    with open(label_path, 'r') as f:
        parts = f.readline().strip().split()

    cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])

    # 读取图像
    img_name = sample_label.replace('.txt', '.jpg')
    img_path = os.path.join(img_root, img_name)

    if not os.path.exists(img_path):
        # 尝试 .png
        img_name = sample_label.replace('.txt', '.png')
        img_path = os.path.join(img_root, img_name)

    if not os.path.exists(img_path):
        print(f"❌ 找不到图像: {img_name}")
        return

    img = Image.open(img_path).convert('RGB')
    print(f"\n真实样本: {img_name}")
    print(f"  原始图像大小: {img.size}")
    print(f"  原始标签 (CXCYWH): [{cx:.4f}, {cy:.4f}, {w:.4f}, {h:.4f}]")

    # 测试 v2 转换
    boxes = BoundingBoxes(
        torch.tensor([[cx, cy, w, h]], dtype=torch.float32),
        format="CXCYWH",
        canvas_size=(img.height, img.width)
    )

    transform = v2.Compose([
        v2.ToImage(),
        v2.Resize((224, 224)),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    try:
        img_out, boxes_out = transform(img, boxes)
        box_out = boxes_out.data[0]

        print(f"  输出图像形状: {img_out.shape}")
        print(f"  输出标签: [{box_out[0]:.4f}, {box_out[1]:.4f}, {box_out[2]:.4f}, {box_out[3]:.4f}]")

        # 检查标签合理性
        checks = []
        checks.append(("宽>0", box_out[2] > 0))
        checks.append(("高>0", box_out[3] > 0))
        checks.append(("中心在[0,1]", 0 <= box_out[0] <= 1 and 0 <= box_out[1] <= 1))
        checks.append(("尺寸合理", box_out[2] <= 1 and box_out[3] <= 1))

        all_ok = True
        for check_name, check_result in checks:
            status = "✓" if check_result else "❌"
            print(f"  {status} {check_name}")
            if not check_result:
                all_ok = False

        if all_ok:
            print(f"\n✓ 真实数据转换正确！")
        else:
            print(f"\n❌ 真实数据转换有问题！")

    except Exception as e:
        print(f"❌ 真实数据转换失败: {e}")


if __name__ == '__main__':
    # 运行所有测试
    test_v2_transforms()
    test_real_data()