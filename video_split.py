import cv2
import os

def split_video_to_frames(video_path, output_dir, num_frames):
    """
    将视频分解为指定数量的图片
    :param video_path: 视频文件路径
    :param output_dir: 保存图片的目录
    :param num_frames: 想要提取的图片总数
    """
    # 1. 打开视频文件
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("无法打开视频文件")
        return

    # 2. 获取视频基本信息
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps
    
    print(f"视频总帧数: {total_frames}")
    print(f"视频时长: {duration:.2f} 秒")

    # 如果视频总帧数比要求的数目还少，就提取所有帧
    if num_frames > total_frames:
        num_frames = total_frames
        print(f"要求数量超过总帧数，将提取全部 {total_frames} 帧")

    # 3. 计算采样的步长 (Interval)
    # 比如 300 帧取 10 张，步长就是 30
    step = total_frames // num_frames

    # 创建输出文件夹
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    count = 0
    extracted_count = 0

    # 4. 开始循环提取
    for i in range(num_frames):
        # 计算当前要提取的帧的索引
        frame_id = i * step
        
        # 将视频指针跳转到指定帧
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        
        # 读取这一帧
        ret, frame = cap.read()
        
        if ret:
            # 构建保存路径，补零命名方便排序 (例如 0001.jpg, 0002.jpg)
            file_name = f"frame_{extracted_count:04d}.jpg"
            save_path = os.path.join(output_dir, file_name)
            
            # 保存图片
            cv2.imwrite(save_path, frame)
            extracted_count += 1
            print(f"已保存: {save_path} (来自视频第 {frame_id} 帧)")
        else:
            break

    # 5. 释放资源
    cap.release()
    print(f"\n任务完成！共提取 {extracted_count} 张图片到目录: {output_dir}")

# --- 使用示例 ---
if __name__ == "__main__":
    # 你可以把这里换成你自己的视频路径
    my_video = "test_video.mp4" 
    my_output = "extracted_images"
    target_number = 1000 # 我想提取 20 张
    
    split_video_to_frames(my_video, my_output, target_number)