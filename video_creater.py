import cv2
import numpy as np

def generate_moving_square_video(output_path, size=512, square_size=100, fps=30, duration=3):
    """
    生成一个白色方块在黑色背景上移动的视频
    :param output_path: 视频保存路径
    :param size: 视频分辨率 (size x size)
    :param square_size: 方块的边长
    :param fps: 帧率
    :param duration: 持续秒数
    """
    # 1. 定义视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (size, size))

    total_frames = fps * duration
    
    # 2. 计算方块的起始位置和每帧位移
    # 计划：从左上角 (0, 0) 移动到右下角
    # 留出余量，防止方块移出屏幕
    max_dist = size - square_size
    step = max_dist / total_frames

    print(f"开始生成视频: {size}x{size}, 共 {total_frames} 帧")

    for i in range(total_frames):
        # 创建一个全黑的底图
        frame = np.zeros((size, size, 3), dtype=np.uint8)

        # 计算当前帧方块左上角的坐标 (x, y)
        curr_x = int(i * step)
        curr_y = int(i * step)

        # 在底图上画一个白色的实心方块
        # cv2.rectangle(图像, 左上角, 右下角, 颜色, 粗细(-1代表实心))
        top_left = (curr_x, curr_y)
        bottom_right = (curr_x + square_size, curr_y + square_size)
        cv2.rectangle(frame, top_left, bottom_right, (255, 255, 255), -1)

        # 写入视频
        out.write(frame)

    out.release()
    print(f"视频已保存至: {output_path}")

if __name__ == "__main__":
    generate_moving_square_video("motion_test.mp4")