import cv2
import os


def extract_frames(video_path, output_folder, frame_interval=30):
    """
    从视频中按指定帧间隔截图保存
    :param video_path: 视频文件路径
    :param output_folder: 保存图片的文件夹路径
    :param frame_interval: 每隔多少帧保存一张（30帧约等于每秒1张）
    """
    # 1. 如果输出文件夹不存在，则自动创建
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"✅ 已创建文件夹: {output_folder}")

    # 2. 打开视频文件
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ 错误：无法打开视频文件 {video_path}")
        return

    # 3. 获取视频总帧率和每秒帧数（供参考）
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"🎥 视频信息: 总帧数 {total_frames}, FPS {fps:.2f}")
    print(f"⏳ 开始按每 {frame_interval} 帧（约 1 秒）保存一张图片...")

    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # 视频读取完毕

        # 4. 判断当前帧号是否满足保存条件
        if frame_count % frame_interval == 0:
            # 生成文件名，用 4 位数字补零，例如 0001.jpg, 0002.jpg
            filename = f"{saved_count + 1:04d}.jpg"
            save_path = os.path.join(output_folder, filename)

            # 保存图片
            cv2.imwrite(save_path, frame)
            saved_count += 1

            # 打印进度（防止看起来像死机）
            if saved_count % 10 == 0:
                print(f"📸 已保存 {saved_count} 张图片...")

        frame_count += 1

    # 5. 释放资源
    cap.release()
    print(f"🎉 完成！共从视频中提取并保存了 {saved_count} 张图片到: {output_folder}")


# ==========================================
# 👇 这里是配置区，根据你的路径修改
# ==========================================
if __name__ == '__main__':
    # 替换成你的视频路径（注意：路径中如果有中文，最好改成英文）
    video_file = "../Detectable_Bottle/V2-cascade-center-box/data/video.mp4"

    # 替换成你希望图片保存到的位置（建议直接放到训练集的文件夹里）
    save_dir = "data/train/images1"

    extract_frames(video_file, save_dir, frame_interval=10)