#!/usr/bin/env python3
"""
步骤0：多视角视频生成脚本

功能：
    从图像序列和mask生成多视角视频，用于后续3D重建
    
流程：
    1. 应用mask抠前景（白色背景）
    2. 智能采样（stride=1，最多45帧，调整为4k+1）
    3. 视觉稳定化预处理（正方形画布，动态跟随物体中心，等比缩放+padding到576×576）
    4. SV4D模型生成多视角（4或8个视角）

输入结构：
    input_images_dir/          # 原始图像目录
    ├── 00000.jpg
    ├── 00001.jpg
    └── ...
    
    input_masks_dir/           # 前景mask目录
    ├── 00000.png
    ├── 00001.png
    └── ...

输出结构：
    output_folder/
    ├── downsampled/           # 采样后的数据
    │   ├── original/          # 原始图像（带背景）
    │   │   ├── 00000.png
    │   │   └── ...
    │   ├── object/            # 前景图像（白色背景）
    │   │   ├── 00000.png
    │   │   └── ...
    │   └── mask/              # 前景mask
    │       ├── 00000.png
    │       └── ...
    ├── multiview_images/      # 多视角数据
    │   ├── v001/              # 视角1图像序列
    │   │   ├── 00000.png
    │   │   └── ...
    │   ├── v002/              # 视角2图像序列
    │   ├── ...
    │   └── multiview_videos/  # 多视角视频
    │       ├── 000000_input.mp4     # 预处理后的输入视频
    │       ├── 000000_v001.mp4      # 视角1视频
    │       ├── 000000_v002.mp4      # 视角2视频
    │       └── ...

主要参数：
    --input_images_dir: 输入图像目录
    --input_masks_dir: 输入mask目录
    --output_folder: 输出目录
    --model_type: sv4d2 (4视角) 或 sv4d2_8views (8视角)
    --num_frames: 限制处理帧数（用于debug）
    --num_steps: 扩散模型采样步数（默认50）
    --seed: 随机种子（默认23）

使用示例：
    # 单个场景
    export CUDA_VISIBLE_DEVICES=0
    python scripts/0_0_gen_multiviews.py \
        --input_images_dir data/DAVIS/JPEGImages/480p/bear \
        --input_masks_dir data/DAVIS/Annotations/480p/bear \
        --output_folder outputs/multiview/bear \
        --model_type sv4d2
    
    export CUDA_VISIBLE_DEVICES=1
    python scripts/0_0_gen_multiviews.py \
        --input_images_dir data/DAVIS/JPEGImages/480p/camel \
        --input_masks_dir data/DAVIS/Annotations/480p/camel \
        --output_folder outputs/multiview/camel \
        --model_type sv4d2
        
    # 批量处理
    for scene in camel bear cows hike; do
        python scripts/0_0_gen_multiviews.py \
            --input_images_dir data/DAVIS/JPEGImages/480p/$scene \
            --input_masks_dir data/DAVIS/Annotations/480p/$scene \
            --output_folder outputs/multiview/$scene \
            --model_type sv4d2
    done
"""

import os
import sys
from glob import glob
from typing import List, Optional, Dict, Tuple, Any
from pathlib import Path

from tqdm import tqdm

# 保存原始工作目录（在切换之前）
ORIGINAL_CWD = os.getcwd()

# 添加generative-models到Python路径并切换工作目录
generative_models_path = os.path.realpath(os.path.join(os.path.dirname(__file__), "../generative-models"))
sys.path.append(generative_models_path)
os.chdir(generative_models_path)

import numpy as np
import torch
import cv2
from fire import Fire
from PIL import Image
import imageio

from scripts.demo.sv4d_helpers import (
    load_model,
    read_video,
    run_img2vid,
)
from sgm.modules.encoders.modules import VideoPredictionEmbedderWithEncoder

# ============================================================================
# 常量定义
# ============================================================================
VAE_FACTOR = 8
LATENT_CHANNELS = 4
DEFAULT_IMAGE_SIZE = 576
DEFAULT_NUM_STEPS = 50
DEFAULT_SEED = 23
DEFAULT_ENCODING_T = 8
DEFAULT_DECODING_T = 4
DEFAULT_FPS = 10
DEFAULT_IMAGE_FRAME_RATIO = 0.85

# 图像文件扩展名
IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']

# 日志格式
LOG_PROCESS = "🔧"
LOG_SUCCESS = "✅"
LOG_WARNING = "⚠️"
LOG_ERROR = "❌"
LOG_INFO = "ℹ️"

# ============================================================================
# 自定义的 preprocess_video 辅助函数
# ============================================================================

def read_gif(input_path, n_frames):
    """读取GIF文件并转换为RGB图像序列
    
    Args:
        input_path: GIF文件路径
        n_frames: 最多读取帧数
        
    Returns:
        List[Image]: RGB图像列表
    """
    from PIL import ImageSequence
    frames = []
    video = Image.open(input_path)
    for img in ImageSequence.Iterator(video):
        frames.append(img.convert("RGB"))
        if len(frames) == n_frames:
            break
    return frames


def read_mp4(input_path, n_frames):
    """读取MP4文件并转换为RGB图像序列
    
    Args:
        input_path: MP4文件路径
        n_frames: 最多读取帧数
        
    Returns:
        List[Image]: RGB图像列表
    """
    frames = []
    vidcap = cv2.VideoCapture(input_path)
    success, image = vidcap.read()
    while success:
        frames.append(Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)))
        success, image = vidcap.read()
        if len(frames) == n_frames:
            break
    return frames


def preprocess_video(
    input_path,
    mask_dir,
    n_frames=21,
    W=576,
    H=576,
    output_folder=None,
    image_frame_ratio=0.9,
    base_count=0,
):
    """视频预处理：动态跟随物体中心，缩放到576×576
    
    【正处理】原始尺寸 -> 576×576：
    
    阶段1：全局分析
        - 计算每帧物体中心 C_i = (center_x, center_y)
        - 统计全局最大尺寸 max_size = max(所有帧的宽高)
        - 计算画布边长 side_len_0 = max_size / image_frame_ratio
          （默认0.9，留10%余量，如max_size=400 -> side_len_0=444）
    
    阶段2：逐帧变换
        - 动态裁剪：以C_i为中心裁剪side_len_0×side_len_0正方形
        - 越界补白：创建白色画布，粘贴重叠区域
        - 缩放到576：resize(side_len_0×side_len_0 -> 576×576)
    
    【注意】side_len_0取决于实际物体尺寸，每个视频可能不同
    
    Args:
        input_path: 输入路径（目录/GIF/MP4）
        mask_dir: mask目录（用于计算物体中心和尺寸）
        n_frames: 处理帧数（默认21）
        W, H: 输出尺寸（默认576×576，SV4D标准输入）
        output_folder: 输出目录
        image_frame_ratio: 留白比例（默认0.9）
        base_count: 输出文件序号（默认0）
    
    Returns:
        str: 输出视频路径 (multiview_images/multiview_videos/{base_count:06d}_input.mp4)
    """
    
    if output_folder is None:
        output_folder = os.path.dirname(input_path)
        
    path = Path(input_path)
    is_video_file = False
    all_img_paths = []
    
    if path.is_file():
        if any([input_path.endswith(x) for x in [".gif", ".mp4"]]):
            is_video_file = True
        else:
            raise ValueError("Path is not a valid video file.")
    elif path.is_dir():
        all_img_paths = sorted([
            f for f in path.iterdir()
            if f.is_file() and f.suffix.lower() in [".jpg", ".jpeg", ".png"]
        ])[:n_frames]
    elif "*" in input_path:
        all_img_paths = sorted(glob(input_path))[:n_frames]
    else:
        raise ValueError(f"Invalid input path: {input_path}")
    
    # 读取图像
    if is_video_file and input_path.endswith(".gif"):
        images = read_gif(input_path, n_frames)[:n_frames]
    elif is_video_file and input_path.endswith(".mp4"):
        images = read_mp4(input_path, n_frames)[:n_frames]
    else:
        images = [Image.open(img_path) for img_path in all_img_paths]
    
    if len(images) != n_frames:
        raise ValueError(f"Input contains {len(images)} frames, expected {n_frames} frames.")
    
    # ========== 视觉稳定化方案：动态跟随物体中心 ==========
    images_v0 = []
    
    # 获取原始图像尺寸
    sample_image_arr = np.array(images[0])
    in_h, in_w = sample_image_arr.shape[:2]
    original_size = (in_w, in_h)  # (width, height)
    
    # ========== 第一阶段：全局维度分析 ==========
    frame_centers = []  # 保存每帧物体的几何中心
    max_size = 0  # 全局最大尺寸S
    
    # 获取mask文件列表
    mask_files = sorted([
        f for f in Path(mask_dir).iterdir()
        if f.is_file() and f.suffix.lower() in [".jpg", ".jpeg", ".png"]
    ])[:n_frames]
    
    if len(mask_files) != len(images):
        raise ValueError(f"mask文件数量({len(mask_files)})与图像数量({len(images)})不匹配")
    
    print(f"{LOG_INFO} 使用mask文件: {mask_dir}")
    
    for idx, image in enumerate(images):
        # 读取mask文件
        mask = cv2.imread(str(mask_files[idx]), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"无法读取mask文件: {mask_files[idx]}")
        
        # 确保mask是二值的
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        
        # 计算当前帧的bbox
        x, y, w, h = cv2.boundingRect(mask)
        
        # 计算当前帧物体的几何中心
        center_x = x + w // 2
        center_y = y + h // 2
        frame_centers.append((center_x, center_y))
        
        # 更新全局最大尺寸S
        max_size = max(max_size, w, h)
    
    # 计算最终画布边长（正方形）- side_len_0
    # 这个值取决于实际物体尺寸，步骤1的side_len_1可能不同
    side_len = int(max_size / image_frame_ratio) if image_frame_ratio is not None else max_size
    
    print(f"{LOG_INFO} 正方形画布尺寸: {side_len}×{side_len} (max_size={max_size}, image_frame_ratio={image_frame_ratio})")
    
    # ========== 第二阶段：逐帧变换执行 ==========
    for frame_idx, image in enumerate(images):
        # 2.1 转换为RGB数组
        image_arr = np.array(image.convert("RGB"))
        
        # 2.2 动态裁剪定位：以当前帧的中心C_i为中心点
        center_x_i, center_y_i = frame_centers[frame_idx]
        
        # 计算裁剪窗口（以物体中心为基准）
        x_start = center_x_i - side_len // 2
        y_start = center_y_i - side_len // 2
        x_end = x_start + side_len
        y_end = y_start + side_len
        
        # 2.3 越界处理与补白：创建白色画布并粘贴重叠部分
        canvas = np.ones((side_len, side_len, 3), dtype=np.uint8) * 255
        
        # 计算原图与裁剪窗口的交集
        src_x_start = max(0, x_start)
        src_y_start = max(0, y_start)
        src_x_end = min(in_w, x_end)
        src_y_end = min(in_h, y_end)
        
        # 计算在画布上的粘贴位置
        dst_x_start = src_x_start - x_start
        dst_y_start = src_y_start - y_start
        dst_x_end = dst_x_start + (src_x_end - src_x_start)
        dst_y_end = dst_y_start + (src_y_end - src_y_start)
        
        # 从原图提取重叠区域
        if src_x_end > src_x_start and src_y_end > src_y_start:
            cropped_region = image_arr[src_y_start:src_y_end, src_x_start:src_x_end]
            # 粘贴到画布
            canvas[dst_y_start:dst_y_end, dst_x_start:dst_x_end] = cropped_region
        
        # 2.4 最终缩放到目标尺寸
        final_576 = cv2.resize(canvas, (W, H), interpolation=cv2.INTER_LANCZOS4)
        
        images_v0.append(final_576)
    
    # 保存处理后的视频
    multiview_videos_dir = os.path.join(output_folder, "multiview_images", "multiview_videos")
    os.makedirs(multiview_videos_dir, exist_ok=True)
    processed_file = os.path.join(multiview_videos_dir, f"{base_count:06d}_input.mp4")
    imageio.mimwrite(processed_file, images_v0, fps=10)
    
    return processed_file


sv4d2_configs = {
    "sv4d2": {
        "T": 12,  # number of frames per sample
        "V": 4,  # number of views per sample
        "model_config": "scripts/sampling/configs/sv4d2.yaml",
        "version_dict": {
            "T": 12 * 4,
            "options": {
                "discretization": 1,
                "cfg": 2.0,
                "min_cfg": 2.0,
                "num_views": 4,
                "sigma_min": 0.002,
                "sigma_max": 700.0,
                "rho": 7.0,
                "guider": 2,
                "force_uc_zero_embeddings": [
                    "cond_frames",
                    "cond_frames_without_noise",
                    "cond_view",
                    "cond_motion",
                ],
                "additional_guider_kwargs": {
                    "additional_cond_keys": ["cond_view", "cond_motion"]
                },
            },
        },
    },
    "sv4d2_8views": {
        "T": 5,  # number of frames per sample
        "V": 8,  # number of views per sample
        "model_config": "scripts/sampling/configs/sv4d2_8views.yaml",
        "version_dict": {
            "T": 5 * 8,
            "options": {
                "discretization": 1,
                "cfg": 2.5,
                "min_cfg": 1.5,
                "num_views": 8,
                "sigma_min": 0.002,
                "sigma_max": 700.0,
                "rho": 7.0,
                "guider": 5,
                "force_uc_zero_embeddings": [
                    "cond_frames",
                    "cond_frames_without_noise",
                    "cond_view",
                    "cond_motion",
                ],
                "additional_guider_kwargs": {
                    "additional_cond_keys": ["cond_view", "cond_motion"]
                },
            },
        },
    },
}


def apply_mask_to_image(image_path: str, mask_path: str) -> np.ndarray:
    """应用mask抠前景，生成白色背景图像
    
    Args:
        image_path: 原始图像路径
        mask_path: mask路径
        
    Returns:
        np.ndarray: RGB数组（前景保留，背景填白）
    """
    # 读取原图和mask
    original_img = cv2.imread(str(image_path))
    mask_img = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    
    if original_img is None or mask_img is None:
        raise ValueError(f"无法读取图像或mask: {image_path}, {mask_path}")
    
    # BGR转RGB
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    
    # 提取灰度mask
    if len(mask_img.shape) == 3:
        mask_gray = mask_img[:, :, 3] if mask_img.shape[2] == 4 else cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)
    else:
        mask_gray = mask_img
    
    # 归一化mask到0-255
    if mask_gray.max() < 255:
        mask_gray = (mask_gray.astype(np.float32) / mask_gray.max() * 255).astype(np.uint8)
    
    # 确保尺寸一致
    if original_img.shape[:2] != mask_gray.shape:
        mask_gray = cv2.resize(mask_gray, (original_img.shape[1], original_img.shape[0]))
    
    # 应用mask: result = original * mask + white * (1 - mask)
    mask_normalized = mask_gray.astype(np.float32) / 255.0
    result = original_img.astype(np.float32) * mask_normalized[:, :, None]
    result += 255.0 * (1 - mask_normalized[:, :, None])
    
    return np.clip(result, 0, 255).astype(np.uint8)




def _calculate_inference_count(n_frames: int, T: int = 12, S_max: int = 11) -> int:
    """计算推理次数：(I-1)*S_max + T >= n_frames
    
    Args:
        n_frames: 总帧数
        T: 窗口大小（默认12）
        S_max: 步长（默认11，overlap=1）
        
    Returns:
        int: 推理次数
    """
    if n_frames <= T:
        return 1
    
    # 计算覆盖 n_frames 所需的最少推理次数
    # 公式: (I-1) * S_max + T >= n_frames
    inference_count = 2
    while ((inference_count - 1) * S_max + T) < n_frames:
        inference_count += 1
    
    return inference_count


def _generate_sampling_windows(n_frames: int, T: int = 12, S_max: int = 11) -> List[List[int]]:
    """生成推理窗口：前I-1个按S_max移动，最后对齐末尾
    
    Args:
        n_frames: 总帧数
        T: 窗口大小（默认12）
        S_max: 步长（默认11）
        
    Returns:
        List[List[int]]: 窗口列表 [[start, end], ...]
    """
    inference_count = _calculate_inference_count(n_frames, T, S_max)
    windows = []
    
    if inference_count == 1:
        windows.append([0, T - 1])
    else:
        # 前 I-1 个窗口按 S_max 移动
        for i in range(inference_count - 1):
            start = i * S_max
            windows.append([start, start + T - 1])
        
        # 最后窗口对齐到末尾
        last_start = n_frames - T
        windows.append([last_start, n_frames - 1])
    
    return windows


def _calculate_window_overlap(windows: List[List[int]]) -> int:
    """计算最后两个窗口的重叠帧数
    
    Args:
        windows: 窗口列表 [[start, end], ...]
        
    Returns:
        int: 重叠帧数（窗口数<=1时返回0）
    """
    if len(windows) <= 1:
        return 0
    
    return windows[-2][1] - windows[-1][0] + 1


def _validate_windows_coverage(windows: List[List[int]], n_frames: int) -> bool:
    """验证窗口完整覆盖所有帧（从0到n_frames-1，无间隙）
    
    Args:
        windows: 窗口列表 [[start, end], ...]
        n_frames: 总帧数
        
    Returns:
        bool: 验证通过返回True
    """
    if not windows:
        return False
    
    # 检查第一个窗口是否从0开始
    if windows[0][0] != 0:
        print(f"{LOG_WARNING} 窗口未从帧0开始: {windows[0]}")
        return False
    
    # 检查最后一个窗口是否覆盖到最后一帧
    if windows[-1][1] != n_frames - 1:
        print(f"{LOG_WARNING} 窗口未覆盖到最后一帧({n_frames-1}): {windows[-1]}")
        return False
    
    # 检查窗口之间是否有间隙
    for i in range(len(windows) - 1):
        current_end = windows[i][1]
        next_start = windows[i + 1][0]
        
        if next_start > current_end + 1:
            gap = next_start - current_end - 1
            print(f"{LOG_WARNING} 窗口{i}和{i+1}之间有{gap}帧间隙")
            return False
    
    return True


def calculate_video_strategy(total_frames: int) -> Dict[str, Any]:
    """计算采样策略：stride=1 -> 截断45帧 -> 推理窗口 -> 调整为4k+1
    
    Args:
        total_frames: 总帧数
    
    Returns:
        Dict[str, Any]: {stride, sampled_n, truncated_n, final_n, 
                        inference_i, windows, loss_rate, indices, last_overlap}
    """
    T = 12
    MAX_TRUNCATE = 45
    S_max = 11  # overlap = 1
    
    # 阶段1: 固定stride=1进行采样（不降采样）
    stride = 1
    sampled_n = total_frames
    stride_reason = f"固定stride=1采样"
    
    # 阶段2: 截断到45
    truncated_n = min(sampled_n, MAX_TRUNCATE)
    
    # 阶段3: 计算推理配置（基于截断后的帧数）
    inference_i = _calculate_inference_count(truncated_n, T, S_max)
    windows = _generate_sampling_windows(truncated_n, T, S_max)
    last_overlap = _calculate_window_overlap(windows)
    
    # 阶段4: Infer后调整到4k+1
    infer_output = truncated_n  # Infer输出与截断后帧数相同
    k = (infer_output - 1) // 4
    final_n = 4 * k + 1
    
    # 生成采样索引列表（最终使用的帧）
    indices = [i * stride for i in range(final_n)]
    
    # 计算损失率（基于最终使用的帧数）
    used_frames = (final_n - 1) * stride + 1
    loss_rate = (total_frames - used_frames) / total_frames
    
    # 验证窗口覆盖完整性
    is_valid = _validate_windows_coverage(windows, truncated_n)
    if not is_valid:
        print(f"{LOG_ERROR} 窗口覆盖验证失败！")
    
    # 输出日志
    print(f"{LOG_INFO} Stride选择: {stride_reason}")
    print(f"{LOG_INFO} 采样流程: {total_frames}帧 -> stride={stride} -> {sampled_n}帧", end="")
    if truncated_n < sampled_n:
        print(f" -> 截断到{truncated_n}帧", end="")
    print(f" -> Infer({inference_i}次) -> 调整到{final_n}帧(4×{k}+1)")
    
    # 格式化窗口显示
    windows_str = ", ".join([f"[{w[0]}:{w[1]}]" for w in windows])
    print(f"{LOG_INFO} 推理窗口: {windows_str}")
    
    return {
        "stride": stride,
        "sampled_n": sampled_n,
        "truncated_n": truncated_n,
        "final_n": final_n,
        "inference_i": inference_i,
        "windows": windows,
        "loss_rate": loss_rate,
        "indices": indices,
        "last_overlap": last_overlap,
    }


def process_and_downsample(
    input_images_dir: str,
    input_masks_dir: str,
    output_folder: str,
    num_frames: Optional[int] = None,
) -> Tuple[str, int, Dict[str, Any]]:
    """处理并采样：应用mask抠前景 + 智能采样到4k+1帧
    
    Args:
        input_images_dir: 输入图像目录
        input_masks_dir: 输入mask目录
        output_folder: 输出根目录
        num_frames: 限制帧数（debug用，默认None）
    
    Returns:
        Tuple[str, int, Dict]: (前景图像目录, 实际帧数, 采样策略)
    """
    print(f"\n{LOG_PROCESS} 处理图像和mask...")
    
    # 读取所有图像文件
    image_files = []
    for ext in IMAGE_EXTENSIONS:
        image_files.extend(glob(os.path.join(input_images_dir, f'*{ext}')))
    
    image_files = sorted(image_files)
    total_frames = len(image_files)
    
    if total_frames == 0:
        raise ValueError(f"在 {input_images_dir} 中未找到图像文件")
    
    print(f"{LOG_INFO} 找到 {total_frames} 帧图像")
    
    # 使用固定约束策略计算采样方案
    strategy = calculate_video_strategy(total_frames)
    sampled_indices = strategy["indices"]
    
    # Debug模式：只取前N帧
    if num_frames is not None and num_frames < len(sampled_indices):
        sampled_indices = sampled_indices[:num_frames]
        print(f"{LOG_WARNING} Debug模式: 只处理前 {num_frames} 帧")
        
        # 重新计算策略（基于截断后的帧数）
        T = 12
        S_max = 11
        truncated_n = num_frames
        inference_i = _calculate_inference_count(truncated_n, T, S_max)
        windows = _generate_sampling_windows(truncated_n, T, S_max)
        last_overlap = _calculate_window_overlap(windows)
        
        # 更新strategy
        strategy["final_n"] = num_frames
        strategy["indices"] = sampled_indices
        strategy["truncated_n"] = truncated_n
        strategy["inference_i"] = inference_i
        strategy["windows"] = windows
        strategy["last_overlap"] = last_overlap
        
        print(f"{LOG_INFO} 重新计算推理窗口: {len(windows)}个窗口")
    
    actual_frames = len(sampled_indices)
    
    # 创建输出目录
    downsampled_dir = os.path.join(output_folder, "downsampled")
    downsampled_original_dir = os.path.join(downsampled_dir, "original")
    downsampled_object_dir = os.path.join(downsampled_dir, "object")
    downsampled_mask_dir = os.path.join(downsampled_dir, "mask")
    
    for dir_path in [downsampled_original_dir, downsampled_object_dir, downsampled_mask_dir]:
        os.makedirs(dir_path, exist_ok=True)
    
    # 处理每一帧
    for output_idx, input_idx in enumerate(tqdm(sampled_indices, desc="处理帧")):
        image_path = image_files[input_idx]
        image_stem = os.path.splitext(os.path.basename(image_path))[0]
        
        # 查找对应的mask文件
        mask_path = _find_mask_file(input_masks_dir, image_stem)
        
        # 读取并保存原始图像
        original_image = Image.open(image_path).convert('RGB')
        original_image.save(os.path.join(downsampled_original_dir, f"{output_idx:05d}.png"))
        
        # 应用mask抠前景
        masked_image = apply_mask_to_image(image_path, mask_path)
        Image.fromarray(masked_image).save(os.path.join(downsampled_object_dir, f"{output_idx:05d}.png"))
        
        # 读取并保存mask
        mask_img = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        mask_gray = _extract_gray_mask(mask_img)
        cv2.imwrite(os.path.join(downsampled_mask_dir, f"{output_idx:05d}.png"), mask_gray)
    
    print(f"{LOG_SUCCESS} 已保存 {actual_frames} 帧到输出目录")
    
    return downsampled_object_dir, actual_frames, strategy


def _find_mask_file(masks_dir: str, image_stem: str) -> str:
    """根据图像文件名查找对应mask
    
    Args:
        masks_dir: mask目录
        image_stem: 图像文件名（不含扩展名）
        
    Returns:
        str: mask文件路径
    """
    mask_path = os.path.join(masks_dir, f"{image_stem}.png")
    if os.path.exists(mask_path):
        return mask_path
    
    for ext in IMAGE_EXTENSIONS:
        alt_path = os.path.join(masks_dir, f"{image_stem}{ext}")
        if os.path.exists(alt_path):
            return alt_path
    
    raise FileNotFoundError(f"找不到对应的mask文件: {mask_path}")


def _extract_gray_mask(mask_img: np.ndarray) -> np.ndarray:
    """提取灰度mask
    
    Args:
        mask_img: mask数组（彩色或灰度）
        
    Returns:
        np.ndarray: 灰度mask
    """
    if len(mask_img.shape) == 3:
        return cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)
    return mask_img


def _prepare_camera_params(
    sv4d2_model: str,
    elevations_deg: Any,
    azimuths_deg: Optional[List[float]],
    n_views: int,
) -> Tuple[List[float], np.ndarray, np.ndarray, np.ndarray]:
    """准备相机参数（俯仰角、方位角）
    
    Args:
        sv4d2_model: 模型名（"sv4d2"或"sv4d2_8views"）
        elevations_deg: 俯仰角（单值或列表）
        azimuths_deg: 方位角列表（None使用默认）
        n_views: 视角数
        
    Returns:
        Tuple: (elevations_deg, azimuths_deg, polars_rad, azimuths_rad)
    """
    # 处理elevations_deg
    if isinstance(elevations_deg, (float, int)):
        elevations_deg = [elevations_deg] * n_views
    
    assert len(elevations_deg) == n_views, \
        f"elevations_deg需要{n_views}个值，实际{len(elevations_deg)}个"
    
    # 处理azimuths_deg
    if azimuths_deg is None:
        azimuths_deg = (
            np.array([0, 60, 120, 180, 240])
            if sv4d2_model == "sv4d2"
            else np.array([0, 30, 75, 120, 165, 210, 255, 300, 330])
        )
    
    assert len(azimuths_deg) == n_views, \
        f"azimuths_deg需要{n_views}个值，实际{len(azimuths_deg)}个"
    
    # 转换为弧度
    polars_rad = np.array([np.deg2rad(90 - e) for e in elevations_deg])
    azimuths_rad = np.array([np.deg2rad((a - azimuths_deg[-1]) % 360) for a in azimuths_deg])
    
    return elevations_deg, azimuths_deg, polars_rad, azimuths_rad


def _initialize_img_matrix(
    n_frames: int,
    n_views: int,
    images_v0: torch.Tensor,
    device: str,
    H: int,
    W: int,
) -> List[List[Optional[torch.Tensor]]]:
    """初始化图像矩阵：第0帧所有视角用零填充，所有帧视角0用实际数据填充
    
    Args:
        n_frames: 帧数
        n_views: 视角数
        images_v0: 输入视角图像序列
        device: 设备
        H, W: 图像尺寸
        
    Returns:
        List[List[Optional[Tensor]]]: 图像矩阵 [n_frames][n_views]
    """
    images_t0 = torch.zeros(n_views, 3, H, W).float().to(device)
    subsampled_views = np.arange(n_views)
    
    img_matrix = [[None] * n_views for _ in range(n_frames)]
    
    # 初始化第0帧的所有视角
    for i, v in enumerate(subsampled_views):
        img_matrix[0][i] = images_t0[v].unsqueeze(0)
    
    # 初始化所有帧的视角0（输入视角）
    for t in range(n_frames):
        img_matrix[t][0] = images_v0[t]
    
    return img_matrix


def generate_multiview_video(
    input_images_dir: str,
    model_path: str,
    output_folder: str,
    n_frames: int,
    strategy: Dict[str, Any],
    num_steps: int = DEFAULT_NUM_STEPS,
    img_size: int = DEFAULT_IMAGE_SIZE,
    seed: int = DEFAULT_SEED,
    encoding_t: int = DEFAULT_ENCODING_T,
    decoding_t: int = DEFAULT_DECODING_T,
    device: str = "cuda",
    elevations_deg: Optional[List[float]] = 0.0,
    azimuths_deg: Optional[List[float]] = None,
    image_frame_ratio: Optional[float] = DEFAULT_IMAGE_FRAME_RATIO,
    verbose: Optional[bool] = False,
) -> Dict[str, Any]:
    """SV4D生成多视角：预处理 -> 加载模型 -> 按窗口推理 -> 返回矩阵
    
    Args:
        input_images_dir: 输入图像目录
        model_path: SV4D模型路径
        output_folder: 输出目录
        n_frames: 帧数
        strategy: 采样策略（含窗口配置）
        num_steps: 采样步数（默认50）
        img_size: 图像尺寸（默认576）
        seed: 随机种子（默认23）
        encoding_t: 编码批大小（默认8）
        decoding_t: 解码批大小（默认4）
        device: 设备（默认"cuda"）
        elevations_deg: 俯仰角（默认0.0）
        azimuths_deg: 方位角（默认None）
        image_frame_ratio: 留白比例（默认0.9）
        verbose: 详细输出（默认False）
    
    Returns:
        Dict: {img_matrix, view_indices, n_frames, H, W}
    """
    import json
    
    print(f"\n{LOG_PROCESS} 生成多视角视频...")
    print(f"{LOG_INFO} 模型: {os.path.basename(model_path)}")
    
    # 获取模型配置
    model_name = os.path.splitext(os.path.basename(model_path))[0]
    assert model_name in sv4d2_configs, f"未知模型: {model_name}"
    
    config = sv4d2_configs[model_name]
    T, V = config["T"], config["V"]
    model_config = config["model_config"]
    version_dict = config["version_dict"].copy()
    
    H, W = img_size, img_size
    n_views = V + 1
    
    # 更新version_dict
    version_dict.update({
        "H": H, "W": W,
        "C": LATENT_CHANNELS,
        "f": VAE_FACTOR,
        "options": {**version_dict["options"], "num_steps": num_steps}
    })
    
    torch.manual_seed(seed)
    
    # 预处理输入视频
    print(f"{LOG_INFO} 预处理输入图像...")
    
    # 构建mask目录路径（使用已经处理好的mask）
    mask_dir = os.path.join(output_folder, "downsampled", "mask")
    if not os.path.exists(mask_dir):
        raise FileNotFoundError(f"mask目录不存在: {mask_dir}")
    
    processed_input_path = preprocess_video(
        input_images_dir, mask_dir, n_frames=n_frames,
        W=W, H=H, output_folder=output_folder,
        image_frame_ratio=image_frame_ratio, base_count=0,
    )
    images_v0 = read_video(processed_input_path, n_frames=n_frames, device=device)
    
    # 准备相机参数
    elevations_deg, azimuths_deg, polars_rad, azimuths_rad = _prepare_camera_params(
        model_name, elevations_deg, azimuths_deg, n_views
    )
    
    # 初始化图像矩阵
    img_matrix = _initialize_img_matrix(n_frames, n_views, images_v0, device, H, W)
    
    # 加载模型
    print(f"{LOG_INFO} 加载模型...")
    model, _ = load_model(model_config, device, version_dict["T"], num_steps, verbose, model_path)
    model.en_and_decode_n_samples_a_time = decoding_t
    for emb in model.conditioner.embedders:
        if isinstance(emb, VideoPredictionEmbedderWithEncoder):
            emb.en_and_decode_n_samples_a_time = encoding_t
    
    # 使用策略中的窗口（已经计算好）
    windows = strategy["windows"]
    t0_list = [w[0] for w in windows]  # 提取每个窗口的起始帧索引
    
    # 格式化窗口显示
    windows_str = ", ".join([f"[{w[0]}:{w[1]}]" for w in windows])
    
    print(f"{LOG_INFO} 采样窗口: {len(t0_list)}个 (推理次数I={strategy['inference_i']})")
    print(f"{LOG_INFO} 窗口详情: {windows_str}")
    
    # 采样多视角
    v0 = 0
    view_indices = np.arange(V) + 1
    subsampled_views = np.arange(n_views)
    
    for idx, t0 in enumerate(tqdm(t0_list, desc="采样进度")):
        if t0 + T > n_frames:
            t0 = n_frames - T
        
        frame_indices = t0 + np.arange(T)
        image = img_matrix[t0][v0]
        cond_motion = torch.cat([img_matrix[t][v0] for t in frame_indices], 0)
        cond_view = torch.cat([img_matrix[t0][v] for v in view_indices], 0)
        
        # 准备相机条件
        polars = polars_rad[subsampled_views[1:]][None].repeat(T, 0).flatten()
        azims = azimuths_rad[subsampled_views[1:]][None].repeat(T, 0).flatten()
        polars = (polars - polars_rad[v0] + np.pi / 2) % (np.pi * 2)
        azims = (azims - azimuths_rad[v0]) % (np.pi * 2)
        
        # 运行采样
        samples = run_img2vid(
            version_dict, model, image, seed,
            polars, azims, cond_motion, cond_view,
            decoding_t, cond_mv=(t0 != 0),
        )
        samples = samples.view(T, V, 3, H, W)
        
        # 更新img_matrix
        for i, t in enumerate(frame_indices):
            for j, v in enumerate(view_indices):
                img_matrix[t][v] = samples[i, j][None] * 2 - 1
    
    # 验证完整性
    all_frames_filled = True
    for v_idx in view_indices:
        none_count = sum(1 for t in range(n_frames) if img_matrix[t][v_idx] is None)
        if none_count > 0:
            none_indices = [t for t in range(n_frames) if img_matrix[t][v_idx] is None]
            print(f"{LOG_ERROR} 视角{v_idx}: {none_count}帧未填充 {none_indices}")
            all_frames_filled = False
    
    if all_frames_filled:
        print(f"{LOG_SUCCESS} 多视角视频生成完成 - 所有帧已填充")
    else:
        print(f"{LOG_WARNING} 多视角视频生成完成 - 存在未填充帧")
    
    return {
        "img_matrix": img_matrix,
        "view_indices": view_indices,
        "n_frames": n_frames,
        "H": H, 
        "W": W,
    }


def save_multiview_outputs(
    multiview_data: Dict[str, Any],
    output_folder: str,
    fps: int = DEFAULT_FPS,
) -> None:
    """保存多视角图像序列和视频
    
    Args:
        multiview_data: 多视角数据字典
        output_folder: 输出目录
        fps: 帧率（默认10）
    """
    print(f"\n{LOG_PROCESS} 保存输出...")
    
    img_matrix = multiview_data["img_matrix"]
    view_indices = multiview_data["view_indices"]
    n_frames = multiview_data["n_frames"]
    
    # 创建输出目录
    multiview_images_dir = os.path.join(output_folder, "multiview_images")
    multiview_videos_dir = os.path.join(multiview_images_dir, "multiview_videos")
    os.makedirs(multiview_videos_dir, exist_ok=True)
    
    # 保存每个视角
    for v in view_indices:
        frames = [img_matrix[t][v] for t in range(n_frames) if img_matrix[t][v] is not None]
        
        # 转换为numpy数组
        img_grid = [
            (((img[0].permute(1, 2, 0) + 1) / 2).cpu().numpy() * 255.0).astype(np.uint8)
            for img in frames
        ]
        
        # 保存视频
        vid_file = os.path.join(multiview_videos_dir, f"000000_v{v:03d}.mp4")
        imageio.mimwrite(vid_file, img_grid, fps=fps)
        
        # 保存图像序列
        view_dir = os.path.join(multiview_images_dir, f"v{v:03d}")
        os.makedirs(view_dir, exist_ok=True)
        for idx, img_array in enumerate(img_grid):
            Image.fromarray(img_array).save(os.path.join(view_dir, f"{idx:05d}.png"))
        
        print(f"{LOG_SUCCESS} 视角{v}: {len(img_grid)}帧")


def main(
    input_images_dir: str = "data/DAVIS/JPEGImages/480p/camel",
    input_masks_dir: str = "data/DAVIS/Annotations/480p/camel",
    output_folder: str = "outputs/multiview/camel",
    num_frames: Optional[int] = None,
    model_type: str = "sv4d2",
    model_path: Optional[str] = None,
    num_steps: int = DEFAULT_NUM_STEPS,
    img_size: int = DEFAULT_IMAGE_SIZE,
    seed: int = DEFAULT_SEED,
    encoding_t: int = DEFAULT_ENCODING_T,
    decoding_t: int = DEFAULT_DECODING_T,
    device: str = "cuda",
    elevations_deg: Optional[List[float]] = 0.0,
    azimuths_deg: Optional[List[float]] = None,
    image_frame_ratio: Optional[float] = DEFAULT_IMAGE_FRAME_RATIO,
    verbose: Optional[bool] = False,
    fps: int = DEFAULT_FPS,
) -> None:
    """主流程：图像序列 -> 多视角视频
    
    流程：处理采样 -> SV4D生成多视角 -> 保存输出
    
    Args:
        input_images_dir: 输入图像目录
        input_masks_dir: 输入mask目录
        output_folder: 输出目录
        num_frames: 限制帧数（debug用）
        model_type: 模型类型（sv4d2=4视角, sv4d2_8views=8视角）
        model_path: 模型路径（默认checkpoints/{model_type}.safetensors）
        num_steps: 采样步数（默认50）
        img_size: 图像尺寸（默认576）
        seed: 随机种子（默认23）
        encoding_t: 编码批大小（默认8）
        decoding_t: 解码批大小（默认4）
        device: 设备（默认cuda）
        elevations_deg: 俯仰角（默认0.0）
        azimuths_deg: 方位角（默认None）
        image_frame_ratio: 留白比例（默认0.9）
        verbose: 详细输出（默认False）
        fps: 视频帧率（默认10）
    """
    # 将相对路径转换为绝对路径（相对于原始工作目录）
    # 因为脚本已经切换到 generative-models 目录，所以需要使用原始工作目录来解析相对路径
    if not os.path.isabs(input_images_dir):
        input_images_dir = os.path.abspath(os.path.join(ORIGINAL_CWD, input_images_dir))
    if not os.path.isabs(input_masks_dir):
        input_masks_dir = os.path.abspath(os.path.join(ORIGINAL_CWD, input_masks_dir))
    if not os.path.isabs(output_folder):
        output_folder = os.path.abspath(os.path.join(ORIGINAL_CWD, output_folder))
    
    print("=" * 80)
    print("🚀 步骤0: 多视角视频生成")
    print("=" * 80)
    print(f"输入: {input_images_dir}")
    print(f"输出: {output_folder}")
    print(f"模型: {model_type}")
    if num_frames:
        print(f"{LOG_WARNING} Debug模式: 只处理前{num_frames}帧")
    print("=" * 80)
    
    # 设置默认模型路径（相对于generative-models目录）
    if model_path is None:
        model_path = f"checkpoints/{model_type}.safetensors"
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    # 检查并删除已存在的输出目录
    if os.path.exists(output_folder):
        import shutil
        print(f"\n{LOG_WARNING} 输出目录已存在，删除旧数据: {output_folder}")
        shutil.rmtree(output_folder)
    
    os.makedirs(output_folder, exist_ok=True)
    
    # 步骤1: 处理图像并采样
    downsampled_object_dir, actual_frames, strategy = process_and_downsample(
        input_images_dir, input_masks_dir, output_folder, num_frames,
    )
    
    # 步骤2: 生成多视角视频
    multiview_data = generate_multiview_video(
        downsampled_object_dir, model_path, output_folder, actual_frames, strategy,
        num_steps, img_size, seed, encoding_t, decoding_t, device,
        elevations_deg, azimuths_deg, image_frame_ratio, verbose,
    )
    
    # 步骤3: 保存输出
    save_multiview_outputs(multiview_data, output_folder, fps)
    
    print("\n" + "=" * 80)
    print(f"🎉 完成! 实际帧数: {actual_frames}")
    print(f"输出目录: {output_folder}")
    print("=" * 80)


if __name__ == "__main__":
    Fire(main)

