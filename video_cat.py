import cv2
import numpy as np
from tqdm import tqdm

def side_by_side_video(video1_path, video2_path, output_path, fps=None):
    # 1. 打开两个视频
    cap1 = cv2.VideoCapture(video1_path)
    cap2 = cv2.VideoCapture(video2_path)

    if not cap1.isOpened() or not cap2.isOpened():
        print("错误：无法打开其中一个视频文件")
        return

    # 2. 获取视频属性（以第一个视频为基准）
    w1 = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    h1 = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps1 = cap1.get(cv2.CAP_PROP_FPS)
    total_frames = int(min(cap1.get(cv2.CAP_PROP_FRAME_COUNT), cap2.get(cv2.CAP_PROP_FRAME_COUNT)))

    # 如果没有指定输出 FPS，就用第一个视频的
    if fps is None:
        fps = fps1

    # 3. 定义输出视频
    # 宽度是两个视频之和，高度取最大值（或者将第二个视频缩放到第一个的高度）
    combined_size = (w1 * 2, h1) 
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, combined_size)

    print(f"正在合并视频: {video1_path} + {video2_path}")
    print(f"输出分辨率: {combined_size}, 总帧数: {total_frames}")

    # 4. 循环读取并拼接
    for _ in tqdm(range(total_frames)):
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        if not ret1 or not ret2:
            break

        # 确保第二个视频的大小和第一个一致（防止报错）
        if (frame2.shape[1], frame2.shape[0]) != (w1, h1):
            frame2 = cv2.resize(frame2, (w1, h1))

        # --- 核心步骤：水平拼接 ---
        # np.hstack 将两个矩阵横向合并
        combined_frame = np.hstack((frame1, frame2))

        # 可以在画面中央画一根黑线作为分割
        cv2.line(combined_frame, (w1, 0), (w1, h1), (0, 0, 0), 2)

        out.write(combined_frame)

    # 5. 释放资源
    cap1.release()
    cap2.release()
    out.release()
    print(f"\n合并完成！保存至: {output_path}")

if __name__ == "__main__":
    # 使用示例
    video_left = "motion_test.mp4"    # 左侧：原视频
    video_right = "my_new_video.mp4"      # 右侧：光流视频
    result_name = "comparison_result.mp4"
    
    side_by_side_video(video_left, video_right, result_name)