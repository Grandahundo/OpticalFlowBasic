import torch
import numpy as np
import cv2 # 用于图像处理和可视化
from torchvision.models.optical_flow import raft_small

# 1. 准备模型
device = "cuda" if torch.cuda.is_available() else "cpu"
model = raft_small(pretrained=True).to(device).eval()

# 2. 读取两张相邻的图片 (或是同一段视频的两帧)
# 注意：RAFT 要求输入是 [Batch, 3, H, W]，且 H, W 必须是 8 的倍数

def produce_flow_by_frame(img1, img2):
    # 3. 推理
    with torch.no_grad():
        import torch.nn.functional as F

        # 1. 获取原始尺寸
        n, c, h, w = img1.shape

        # 2. 计算最接近的 8 的倍数
        # 480 已经是 8 的倍数，268 会被调整为 272 (或者 264)
        new_h = ((h + 7) // 8) * 8
        new_w = ((w + 7) // 8) * 8

        # 3. 缩放图片
        if h != new_h or w != new_w:
            print(f"调整尺寸: ({h}, {w}) -> ({new_h}, {new_w})")
            img1 = F.interpolate(img1, size=(new_h, new_w), mode='bilinear', align_corners=False)
            img2 = F.interpolate(img2, size=(new_h, new_w), mode='bilinear', align_corners=False)

        # 4. 现在可以运行模型了
        list_of_flows = model(img1, img2)
            # list_of_flows = model(img1, img2)
        flow = list_of_flows[-1][0] # 拿到最终的 [2, H, W] 位移图

    # 4. 可视化 (这是重点！)
    # def flow_to_image(flow):
    #     # flow 形状是 (2, H, W) -> 转为 (H, W, 2)
    #     flow_uv = flow.permute(1, 2, 0).cpu().numpy()
        
    #     # 使用 OpenCV 的 HSV 转换法
    #     hsv = np.zeros((flow_uv.shape[0], flow_uv.shape[1], 3), dtype=np.uint8)
    #     mag, ang = cv2.cartToPolar(flow_uv[..., 0], flow_uv[..., 1]) # 笛卡尔坐标转极坐标
    #     hsv[..., 0] = ang * 180 / np.pi / 2  # 方向决定色调
    #     hsv[..., 1] = 255                    # 饱和度设为最大
    #     hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX) # 速度快慢决定亮度
        
    #     return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    def flow_to_image(flow):
        """
        将光流图转为漂亮的彩色图
        :param flow: torch.Tensor 或 numpy 数组, 形状为 (2, H, W) 或 (H, W, 2)
        """
        # 确保格式为 (H, W, 2)
        if len(flow.shape) == 3 and flow.shape[0] == 2:
            flow = flow.permute(1, 2, 0).detach().cpu().numpy()
        
        h, w = flow.shape[:2]
        hsv = np.zeros((h, w, 3), dtype=np.uint8)
        
        # 1. 计算极坐标：幅度(mag)代表速度，角度(ang)代表方向
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        mag = mag * 10
        
        # 2. 设置色调（根据方向）
        hsv[..., 0] = ang * 180 / np.pi / 2
        
        # 3. 设置饱和度（设为最大，让颜色浓郁）
        hsv[..., 1] = 255
        
        # 4. 设置亮度（关键改进点！）
        # 不要直接用 mag，而是将 mag 缩放到 0-255 范围
        # 这样即使只有 1 像素的位移，也会被拉伸到 255 (最亮)
        if mag.max() > 0:
            # 使用 cv2.normalize 动态拉伸对比度
            hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        else:
            hsv[..., 2] = 0
            
        # 5. 转回 BGR 方便 OpenCV 显示和保存
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    # 保存看结果
    vis_flow = flow_to_image(flow)
    return vis_flow

for i in range(1, 131):
    path1 = f"extracted_images/frame_{i - 1:04d}.jpg"
    lst_img = torch.from_numpy(cv2.imread(path1)).permute(2, 0, 1).float().unsqueeze(0).to(device)
    path2 = f"extracted_images/frame_{i:04d}.jpg"
    cur_img = torch.from_numpy(cv2.imread(path1)).permute(2, 0, 1).float().unsqueeze(0).to(device)
    vis_flow = produce_flow_by_frame(lst_img, cur_img)
    cv2.imwrite(f'flow_results/frame_{i - 1:04d}.png', vis_flow)