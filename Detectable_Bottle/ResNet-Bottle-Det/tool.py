import os
import torch
import torchvision.models as models
imglist=os.listdir("dataset\\images\\train")
labellist=os.listdir("dataset\\labels\\train")
# 不要删除，而是只保留最大的目标
for labelname in labellist:
    label_path = os.path.join("dataset\\labels\\train", labelname)

    with open(label_path, "r") as f:
        lines = f.readlines()

    if len(lines) > 1:
        # 找面积最大的目标
        max_area = 0
        best_line = lines[0]

        for line in lines:
            parts = line.strip().split()
            w = float(parts[3])
            h = float(parts[4])
            area = w * h
            if area > max_area:
                max_area = area
                best_line = line

        # 只保留最大的
        with open(label_path, "w") as f:
            f.write(best_line)

        print(f"✏️  修改: {labelname} ({len(lines)} → 1 个目标)")
import os

label_dir = "dataset\\labels\\train"
image_dir = "dataset\\images\\train"

empty_count = 0

for labelname in os.listdir(label_dir):
    label_path = os.path.join(label_dir, labelname)

    # 检查文件是否为空
    if os.path.getsize(label_path) == 0:
        print(f"🗑️  空标签: {labelname}")
        os.remove(label_path)
        empty_count += 1

        # 同时删除对应图像
        img_name = os.path.splitext(labelname)[0] + ".jpg"
        img_path = os.path.join(image_dir, img_name)
        if os.path.exists(img_path):
            os.remove(img_path)
            print(f"    图像已删除: {img_name}")

print(f"\n删除 {empty_count} 个空文件")
# import torch
# import os
#
# # 查看 torch 缓存目录
# print(torch.hub.get_dir())
# # 输出类似：C:\Users\你的用户名\.cache\torch\hub
#
# # 查看具体文件
# cache_dir = torch.hub.get_dir()
# checkpoint_dir = os.path.join(cache_dir, 'checkpoints')
# print(f"模型保存在：{checkpoint_dir}")
#
# # 列出已下载的模型
# if os.path.exists(checkpoint_dir):
#     files = os.listdir(checkpoint_dir)
#     print(f"已下载的模型文件：{files}")