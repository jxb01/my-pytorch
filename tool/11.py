import cv2
import os

import cv2
import numpy as np


def letterbox_image_cv2(frame, target_size=640):
    """
    推理/离线预处理用的 Letterbox:
    保持原图长宽比例缩放到 target_size，并在四周补灰边 (114, 114, 114)
    """
    h, w, _ = frame.shape
    # 1. 计算缩放比例：取最小的那个比例，保证长边缩放到 target_size
    scale = min(target_size / w, target_size / h)
    new_w, new_h = int(w * scale), int(h * scale)

    # 2. 等比例缩放原图
    resized = cv2.resize(frame, (new_w, new_h))

    # 3. 计算四周需要补的灰边尺寸
    delta_w = target_size - new_w
    delta_h = target_size - new_h
    top = delta_h // 2
    bottom = delta_h - top
    left = delta_w // 2
    right = delta_w - left

    # 4. 使用 cv2.copyMakeBorder 补灰边
    # 颜色 (114, 114, 114) 是行业标准灰，不干扰模型视觉
    color = [114, 114, 114]
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    # 5. 返回补边后的图，以及偏移量（还原坐标时会用到）
    return padded, left, top, scale
def preprocess_images_offline(input_dir, output_dir, target_size=640):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for file in os.listdir(input_dir):
        if file.endswith(".jpg") or file.endswith(".png"):
            img = cv2.imread(os.path.join(input_dir, file))
            # 调用你的 letterbox 函数
            padded_img, _, _, _ = letterbox_image_cv2(img, target_size)
            cv2.imwrite(os.path.join(output_dir, file), padded_img)
    print("所有图片预处理完成！")


# 把训练集的路径传进去
preprocess_images_offline("D:\\my pytorch\\Detectable_Bottle\\V2-cascade-center-box\\data1\\train\\images", "D:\\my pytorch\\Detectable_Bottle\\V2-cascade-center-box\\data1\\train\\images")