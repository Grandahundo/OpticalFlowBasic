# Learn Optical Flow Basic (RAFT)

本项目是一个用于学习和实验计算机视觉中**光流（Optical Flow）**算法的基础仓库。主要使用 **RAFT (Recurrent All-Pairs Field Transforms)** 模型进行位移分析，并提供了一套完整的视频处理工具链。

## 🛠 功能模块说明

仓库中的脚本涵盖了从视频处理到光流可视化的全过程：

### 1. 视频预处理
- `video_split.py`: 将输入视频（如 `test_video.mp4`）分解为指定数量的序列图片，保存在 `extracted_images/` 目录中。

### 2. 光流计算 (核心)
- `main.py`: 调用 `torchvision` 中的 RAFT 模型，读取序列图片并计算两帧之间的像素级位移，生成彩色光流图并存入 `flow_results/`。
- `video_generate.py`: 一个用于生成白色方块在黑色背景上移动的测试视频，专门用于验证光流对纯色物体追踪的鲁棒性。

### 3. 后处理与可视化
- `video_creater.py` / `video_generate.py`: 将生成的 `flow_results/` 图片序列合成为完整的光流视频。
- `video_cat.py`: 将原始视频与光流视频进行**左右对比（Side-by-Side）**拼接，便于直观分析。

---

## 🚀 实验流程

按照以下顺序运行脚本即可完成一次完整的光流分析：

1. **分解视频**: 准备一个视频并运行 `video_split.py`。
2. **计算光流**: 运行 `main.py` 生成位移热力图。
3. **合成结果**: 运行 `video_creater.py`。
4. **制作对比图**: 运行 `video_cat.py` 生成最终的 `comparison_result.mp4`。

---

## 📈 实验观察

在实验中，我们使用了基于 HSV 色彩空间的可视化方案：
- **颜色 (Hue)**: 代表像素移动的方向。
- **亮度 (Value)**: 代表移动的速度。
- **优势**: 相比传统方法，本项目使用的 RAFT 模型在纯色区域（无纹理区域）表现出了极强的预测能力，能够通过边缘运动推断出物体内部的位移。

---

## 🎥 效果展示 (Result)

### 左右对比演示 (Comparison Result)

https://github.com/user-attachments/assets/426ed4ad-62f8-495e-b0c3-fa2667959175


*(注：如果无法直接播放，请检查视频路径是否正确)*


