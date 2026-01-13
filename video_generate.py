import cv2
import os

def combine_images_to_video(image_dir, output_video_path, fps=30):
    """
    将文件夹内的图片合成为视频
    :param image_dir: 图片所在的文件夹路径
    :param output_video_path: 输出视频的路径 (例如 'output.mp4')
    :param fps: 帧率 (每秒播放多少张图)
    """
    # 1. 获取文件夹内所有图片，并【务必排序】
    # 排序确保 frame_0001 在 frame_0002 前面
    images = [img for img in os.listdir(image_dir) if img.endswith((".jpg", ".png", ".jpeg"))]
    images.sort() 

    if not images:
        print("文件夹内没有找到图片！")
        return

    # 2. 读取第一张图片来获取画面的宽和高
    first_image_path = os.path.join(image_dir, images[0])
    frame = cv2.imread(first_image_path)
    height, width, layers = frame.shape
    size = (width, height) # 注意：OpenCV 的 size 是 (宽, 高)

    # 3. 初始化视频写入器
    # 'mp4v' 是常用的 MP4 编码器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, size)

    print(f"正在合成视频，总计 {len(images)} 帧...")

    # 4. 逐张读取并写入
    for image_name in images:
        image_path = os.path.join(image_dir, image_name)
        frame = cv2.imread(image_path)
        
        # 如果图片尺寸不一致，强制缩放到第一张图的大小（防止报错）
        if (frame.shape[1], frame.shape[0]) != size:
            frame = cv2.resize(frame, size)
            
        video_writer.write(frame)
        print(f"已写入: {image_name}")

    # 5. 释放资源
    video_writer.release()
    print(f"\n合成成功！视频保存在: {output_video_path}")

# --- 使用示例 ---
if __name__ == "__main__":
    input_folder = "flow_results"  # 你的图片文件夹
    output_name = "my_new_video.mp4"   # 生成的视频名
    target_fps = 24                    # 电影级帧率选 24，丝滑选 60
    
    combine_images_to_video(input_folder, output_name, target_fps)