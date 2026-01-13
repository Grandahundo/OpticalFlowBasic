import torch
import numpy as np
import cv2
import torch.nn.functional as F
from torchvision.models.optical_flow import raft_small, Raft_Small_Weights
from pathlib import Path
from tqdm import tqdm # 建议安装: pip install tqdm (进度条)

# --- 1. 环境准备 ---
device = "cuda" if torch.cuda.is_available() else "cpu"
# 使用最新的 Weights API 加载模型
weights = Raft_Small_Weights.DEFAULT
model = raft_small(weights=weights).to(device).eval()

# --- 2. 核心功能函数 ---

def flow_to_image(flow):
    """
    将 [2, H, W] 的光流张量转为 BGR 彩色图
    """
    if torch.is_tensor(flow):
        flow = flow.permute(1, 2, 0).detach().cpu().numpy()
    
    h, w = flow.shape[:2]
    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    
    # 计算极坐标
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    
    # 颜色映射：方向->色调，饱和度全满，位移大小->亮度
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 1] = 255
    # 使用归一化让动态范围最大化，解决“黑乎乎一坨”的问题
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def preprocess_image(img_path):
    """
    读取并预处理图像，确保符合 RAFT 输入要求
    """
    img = cv2.imread(str(img_path))
    if img is None:
        return None
    
    # 转为 Tensor: [H, W, C] -> [1, C, H, W]
    img_tensor = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0).to(device)
    
    # 尺寸填充/缩放：确保 H, W 是 8 的倍数
    _, _, h, w = img_tensor.shape
    new_h = ((h + 7) // 8) * 8
    new_w = ((w + 7) // 8) * 8
    
    if h != new_h or w != new_w:
        img_tensor = F.interpolate(img_tensor, size=(new_h, new_w), mode='bilinear', align_corners=False)
        
    return img_tensor

@torch.no_grad()
def get_flow(img1, img2):
    """
    计算两帧之间的光流
    """
    # RAFT 预测，返回迭代列表，取最后一个
    list_of_flows = model(img1, img2)
    return list_of_flows[-1][0]

# --- 3. 主循环逻辑 ---

def process_video_frames(input_dir, output_dir):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 获取所有图片并排序
    frame_paths = sorted(list(input_path.glob("frame_*.jpg")))
    
    if len(frame_paths) < 2:
        print("图片数量不足，无法计算光流")
        return

    print(f"开始处理: 共 {len(frame_paths)-1} 对图像")
    
    # 使用 tqdm 显示进度条
    for i in tqdm(range(len(frame_paths) - 1)):
        p1, p2 = frame_paths[i], frame_paths[i+1]
        
        # 预处理
        img1 = preprocess_image(p1)
        img2 = preprocess_image(p2)
        
        if img1 is None or img2 is None:
            continue
            
        # 计算光流
        flow = get_flow(img1, img2)
        
        # 可视化
        vis_flow = flow_to_image(flow)
        
        # 保存：保持和原图序号对应
        save_name = output_path / f"flow_{i:04d}.png"
        # 在 preprocess_image 里面增加
        img = cv2.GaussianBlur(vis_flow, (3, 3), 0)
        cv2.imwrite(str(save_name), vis_flow)

if __name__ == "__main__":
    # 配置路径
    INPUT_DIR = "extracted_images"
    OUTPUT_DIR = "flow_results"
    
    process_video_frames(INPUT_DIR, OUTPUT_DIR)