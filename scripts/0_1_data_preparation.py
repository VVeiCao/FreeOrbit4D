#!/usr/bin/env python3
"""
步骤1：数据准备脚本

功能：
    将步骤0生成的多视角图像逆变换到训练所需尺寸，生成训练数据集
    
流程：
    1. 逆变换：576×576 -> 等比缩放 -> 居中放入画布 -> rembg（硬边缘）
    2. 智能裁剪到目标尺寸
    3. 生成多视角、原始、前景、scene图像及对应mask

输入结构（步骤0的输出）：
    input_dir/
    ├── downsampled/
    │   ├── original/          # 原始图像（带背景）
    │   │   ├── 00000.png
    │   │   └── ...
    │   ├── object/            # 前景图像（白色背景）
    │   │   ├── 00000.png
    │   │   └── ...
    │   └── mask/              # 前景mask
    │       ├── 00000.png
    │       └── ...
    ├── multiview_images/
    │   ├── v001/              # 视角1图像序列
    │   │   ├── 00000.png
    │   │   └── ...
    │   ├── v002/              # 视角2图像序列
    │   └── ...
    └── preprocess_transform_params.json  # 变换参数（可选，不再使用）

输出结构：
    output_dir/
    ├── 00000/                 # 第1帧数据
    │   ├── images/
    │   │   ├── 00000_0.png           # 多视角图像（视角0=原始）
    │   │   ├── 00000_1.png           # 视角1
    │   │   ├── 00000_2.png           # 视角2
    │   │   ├── ...
    │   │   ├── 00000_original.png    # 原始图像
    │   │   └── 00000_foreground.png  # 前景图像
    │   ├── masks/
    │   │   ├── 00000_0_mask.png      # 多视角mask
    │   │   ├── 00000_1_mask.png
    │   │   ├── ...
    │   │   ├── 00000_original_mask.png
    │   │   ├── 00000_foreground_mask.png
    │   │   └── 00000_background_mask.png
    │   └── scene/
    │       ├── 00000_scene_image.png
    │       └── 00000_scene_mask.png
    ├── 00001/                 # 第2帧数据
    └── ...

主要参数：
    --input_dir: 输入目录（步骤0的输出）
    --output_dir: 输出目录
    --output_height: 输出图像高度（默认480）
    --output_width: 输出图像宽度（默认832）
    --debug: Debug模式（只处理前3帧）

使用示例：
    # 单个场景
    export CUDA_VISIBLE_DEVICES=0
    python scripts/0_1_data_preparation.py \
        --input_dir outputs/multiview/camel \
        --output_dir outputs/prepared/camel
    
    # 批量处理
    for scene in camel bear cows hike; do
        export CUDA_VISIBLE_DEVICES=0
        python scripts/0_1_data_preparation.py \
            --input_dir outputs/multiview/$scene \
            --output_dir outputs/prepared/$scene
    done
"""

import os
import sys
import argparse
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import cv2
import rembg
from PIL import Image
from tqdm import tqdm

# ============================================================================
# 常量定义
# ============================================================================
DEFAULT_OUTPUT_HEIGHT = 480
DEFAULT_OUTPUT_WIDTH = 832
DEBUG_FRAMES = 3

# 日志格式
LOG_PROCESS = "🔧"
LOG_SUCCESS = "✅"
LOG_WARNING = "⚠️"
LOG_ERROR = "❌"
LOG_INFO = "ℹ️"


# ============================================================================
# 逆变换工具
# ============================================================================

def simple_inverse_transform(
    frame_576: Image.Image,
    transform_params: Dict[str, Any],
    target_height: int,
    target_width: int,
    rembg_session: Any
) -> Tuple[Image.Image, np.ndarray]:
    """简化逆变换: 576×576 -> upscale -> pad -> rembg (硬边缘模式)"""
    side_len = transform_params['side_len']
    
    upscaled = frame_576.resize((side_len, side_len), Image.LANCZOS)
    upscaled_arr = np.array(upscaled)
    
    canvas = np.ones((target_height, target_width, 3), dtype=np.uint8) * 255
    start_y = (target_height - side_len) // 2
    start_x = (target_width - side_len) // 2
    
    if start_y >= 0 and start_x >= 0:
        canvas[start_y:start_y+side_len, start_x:start_x+side_len] = upscaled_arr
    else:
        crop_y = max(0, -start_y)
        crop_x = max(0, -start_x)
        crop_h = min(side_len - crop_y, target_height)
        crop_w = min(side_len - crop_x, target_width)
        canvas[0:crop_h, 0:crop_w] = upscaled_arr[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
    
    canvas_pil = Image.fromarray(canvas)
    
    # ========== 硬边缘模式：直接获取二值mask ==========
    # 使用 only_mask=True 直接获取二值mask（硬边缘，无羽化）
    # 显式关闭 alpha_matting 以避免边缘羽化
    alpha_mask = rembg.remove(canvas_pil, session=rembg_session, only_mask=True, alpha_matting=False)
    alpha_mask = np.array(alpha_mask)
    
    # 强制二值化：确保只有0和255，没有中间灰度值
    _, alpha_mask = cv2.threshold(alpha_mask, 127, 255, cv2.THRESH_BINARY)
    
    # 使用mask合成白底RGB图像
    result_rgb = canvas.copy()
    # mask为0的地方（背景）填充为白色
    result_rgb[alpha_mask < 127] = 255
    
    result_rgb = Image.fromarray(result_rgb)
    
    return result_rgb, alpha_mask


# ============================================================================
# 图像处理工具
# ============================================================================

def smart_crop_and_resize(image: Any, height: int, width: int) -> Any:
    """智能居中裁剪并调整尺寸（保持宽高比）"""
    is_pil = isinstance(image, Image.Image)
    image_arr = np.array(image) if is_pil else image
    is_mask = len(image_arr.shape) == 2
    
    if is_mask:
        image_arr = image_arr[:, :, np.newaxis]
    
    image_height, image_width = image_arr.shape[:2]
    
    if image_height == height and image_width == width:
        return image if is_pil else (image_arr if not is_mask else image_arr[:, :, 0])
    
    if image_height / image_width < height / width:
        cropped_width = int(image_height / height * width)
        left = (image_width - cropped_width) // 2
        image_arr = image_arr[:, left:left + cropped_width]
    else:
        cropped_height = int(image_width / width * height)
        top = (image_height - cropped_height) // 2
        image_arr = image_arr[top:top + cropped_height, :]
    
    if is_mask:
        return cv2.resize(image_arr[:, :, 0], (width, height), interpolation=cv2.INTER_NEAREST)
    else:
        result_pil = Image.fromarray(image_arr).resize((width, height), Image.LANCZOS)
        return result_pil if is_pil else np.array(result_pil)


def apply_smart_crop(image: Any, mask: Optional[np.ndarray], 
                     target_height: Optional[int], target_width: Optional[int]) -> Tuple[Any, Optional[np.ndarray]]:
    """对图像和mask同时应用智能裁剪"""
    if target_height is None or target_width is None:
        return image, mask
    cropped_image = smart_crop_and_resize(image, target_height, target_width)
    cropped_mask = smart_crop_and_resize(mask, target_height, target_width) if mask is not None else None
    return cropped_image, cropped_mask


# ============================================================================
# Mask处理工具
# ============================================================================

def create_pseudo_mask(width: int, height: int) -> np.ndarray:
    """创建全白mask（255）"""
    return np.ones((height, width), dtype=np.uint8) * 255


def invert_mask(mask: np.ndarray) -> np.ndarray:
    """反转mask（255-mask）"""
    return 255 - mask


def generate_scene(original_image: np.ndarray, foreground_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """生成scene数据：前景bbox填白，返回(scene_image, scene_mask)"""
    if len(foreground_mask.shape) == 3:
        foreground_mask = foreground_mask[:, :, 0]
    
    foreground_pixels = np.where(foreground_mask > 127)
    
    if len(foreground_pixels[0]) == 0:
        return original_image.copy(), np.ones_like(foreground_mask) * 255
    
    y_min, y_max = foreground_pixels[0].min(), foreground_pixels[0].max()
    x_min, x_max = foreground_pixels[1].min(), foreground_pixels[1].max()
    
    scene_image = original_image.copy()
    scene_image[y_min:y_max+1, x_min:x_max+1] = 255
    
    updated_mask = foreground_mask.copy()
    updated_mask[y_min:y_max+1, x_min:x_max+1] = 255
    
    return scene_image, ~updated_mask


# ============================================================================
# 数据验证
# ============================================================================

def detect_and_validate_input(input_dir: str) -> Dict[str, Any]:
    """检测并验证输入数据结构"""
    print(f"\n{LOG_PROCESS} 验证输入数据...")
    input_dir = Path(input_dir)
    
    # 检查必需目录
    downsampled_dir = input_dir / "downsampled"
    required_dirs = {
        'downsampled_original': downsampled_dir / "original",
        'downsampled_object': downsampled_dir / "object",
        'downsampled_mask': downsampled_dir / "mask",
        'multiview_images': input_dir / "multiview_images"
    }
    
    for name, path in required_dirs.items():
        if not path.exists():
            raise FileNotFoundError(f"找不到目录: {path}")
    
    # 检测视角目录
    multiview_dir = required_dirs['multiview_images']
    view_dirs = sorted([d for d in multiview_dir.iterdir() 
                       if d.is_dir() and d.name.startswith('v')])
    if not view_dirs:
        raise ValueError("找不到视角目录 (v001, v002, ...)")
    
    # 检测帧数 - 以multiview_images的帧数为准
    view_frames_list = [len(list(view_dir.glob("*.png"))) for view_dir in view_dirs]
    if len(set(view_frames_list)) != 1:
        raise ValueError(f"多视角帧数不一致: {view_frames_list}")
    
    n_frames = view_frames_list[0]
    
    # 检查downsampled目录的帧数
    downsampled_frames = [
        len(list(required_dirs['downsampled_original'].glob("*.png"))),
        len(list(required_dirs['downsampled_object'].glob("*.png"))),
        len(list(required_dirs['downsampled_mask'].glob("*.png")))
    ]
    
    # downsampled目录的帧数必须一致
    if len(set(downsampled_frames)) != 1:
        raise ValueError(f"downsampled目录帧数不一致: {downsampled_frames}")
    
    downsampled_n_frames = downsampled_frames[0]
    
    # 检查downsampled是否有足够的帧
    if downsampled_n_frames < n_frames:
        raise ValueError(f"downsampled目录帧数不足: 需要{n_frames}帧，实际只有{downsampled_n_frames}帧")
    
    # 如果downsampled帧数更多，给出警告
    if downsampled_n_frames > n_frames:
        print(f"{LOG_WARNING} downsampled目录有{downsampled_n_frames}帧，但multiview只有{n_frames}帧，将只使用前{n_frames}帧")
    
    print(f"{LOG_SUCCESS} 检测到 {len(view_dirs)} 个视角, {n_frames} 帧")
    
    return {
        'input_dir': input_dir,
        'downsampled_original_dir': required_dirs['downsampled_original'],
        'downsampled_object_dir': required_dirs['downsampled_object'],
        'downsampled_mask_dir': required_dirs['downsampled_mask'],
        'multiview_images_dir': required_dirs['multiview_images'],
        'view_dirs': view_dirs,
        'n_views': len(view_dirs),
        'n_frames': n_frames,
    }


# ============================================================================
# 数据完整性验证
# ============================================================================

def validate_data_integrity(data_info: Dict[str, Any],
                           n_frames_to_process: int) -> bool:
    """验证所有必需文件是否存在且尺寸一致"""
    print(f"\n{LOG_PROCESS} 验证数据完整性...")
    
    all_valid = True
    
    # 验证所有帧文件都存在
    for frame_idx in range(n_frames_to_process):
        frame_name = f"{frame_idx:05d}.png"
        
        if not (data_info['downsampled_original_dir'] / frame_name).exists():
            print(f"{LOG_ERROR} 缺少原始图像: {frame_name}")
            all_valid = False
        
        if not (data_info['downsampled_object_dir'] / frame_name).exists():
            print(f"{LOG_ERROR} 缺少前景图像: {frame_name}")
            all_valid = False
        
        if not (data_info['downsampled_mask_dir'] / frame_name).exists():
            print(f"{LOG_ERROR} 缺少前景mask: {frame_name}")
            all_valid = False
        
        for view_dir in data_info['view_dirs']:
            if not (view_dir / frame_name).exists():
                print(f"{LOG_ERROR} 缺少多视角图像: {view_dir.name}/{frame_name}")
                all_valid = False
    
    # 验证图像尺寸一致性
    sample_original = Image.open(data_info['downsampled_original_dir'] / "00000.png")
    sample_foreground = Image.open(data_info['downsampled_object_dir'] / "00000.png")
    
    if sample_original.size != sample_foreground.size:
        print(f"{LOG_ERROR} 原始图像和前景图像尺寸不一致: {sample_original.size} vs {sample_foreground.size}")
        all_valid = False
    
    # 验证多视角图像尺寸
    sample_multiview = Image.open(data_info['view_dirs'][0] / "00000.png")
    if sample_multiview.size != (576, 576):
        print(f"{LOG_WARNING} 多视角图像尺寸不是576×576: {sample_multiview.size}")
    
    if all_valid:
        print(f"{LOG_SUCCESS} 数据完整性验证通过")
    else:
        print(f"{LOG_ERROR} 数据完整性验证失败")
    
    return all_valid


# ============================================================================
# 帧处理函数
# ============================================================================

def _process_multiview_images(frame_name: str, frame_idx: int, data_info: Dict[str, Any],
                             rembg_session: Any, transform_params: Dict[str, Any],
                             target_params: Dict[str, int], images_dir: str, masks_dir: str) -> None:
    """处理多视角图像：576×576 -> upscale -> pad -> rembg"""
    target_height = target_params['height']
    target_width = target_params['width']
    
    for view_idx, view_dir in enumerate(data_info['view_dirs']):
        view_image_path = view_dir / f"{frame_name}.png"
        if not view_image_path.exists():
            raise FileNotFoundError(f"找不到多视角图像: {view_image_path}")
        
        view_image = Image.open(view_image_path)
        
        # 逆变换：576×576 -> upscale -> pad -> rembg（硬边缘）
        view_rgb, mask = simple_inverse_transform(
            view_image, transform_params,
            target_height, target_width, 
            rembg_session
        )
        
        view_rgb.save(os.path.join(images_dir, f"{frame_name}_{view_idx}.png"))
        cv2.imwrite(os.path.join(masks_dir, f"{frame_name}_{view_idx}_mask.png"), mask)


def _process_original_image(frame_name: str, data_info: Dict[str, Any], target_params: Dict[str, int],
                           images_dir: str, masks_dir: str) -> Tuple[Image.Image, np.ndarray]:
    """处理原始图像：裁剪并生成全白mask"""
    target_height = target_params['height']
    target_width = target_params['width']
    
    original_img = Image.open(data_info['downsampled_original_dir'] / f"{frame_name}.png").convert('RGB')
    original_mask = create_pseudo_mask(original_img.width, original_img.height)
    
    original_img_cropped, original_mask_cropped = apply_smart_crop(
        original_img, original_mask, target_height, target_width
    )
    
    original_img_cropped.save(os.path.join(images_dir, f"{frame_name}_original.png"))
    cv2.imwrite(os.path.join(masks_dir, f"{frame_name}_original_mask.png"), original_mask_cropped)
    
    return original_img_cropped, original_mask_cropped


def _process_foreground_image(frame_name: str, data_info: Dict[str, Any], target_params: Dict[str, int],
                             images_dir: str, masks_dir: str) -> np.ndarray:
    """处理前景图像：裁剪并保存前景/背景mask"""
    target_height = target_params['height']
    target_width = target_params['width']
    
    foreground_img = Image.open(data_info['downsampled_object_dir'] / f"{frame_name}.png").convert('RGB')
    mask_img = cv2.imread(str(data_info['downsampled_mask_dir'] / f"{frame_name}.png"), cv2.IMREAD_UNCHANGED)
    
    if len(mask_img.shape) == 3:
        foreground_mask = (
            mask_img[:, :, 3] if mask_img.shape[2] == 4 else
            mask_img[:, :, 0] if mask_img[:, :, 0].max() > 0 else
            cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)
        )
    else:
        foreground_mask = mask_img
    
    if foreground_mask.max() < 255:
        foreground_mask = (foreground_mask.astype(np.float32) / foreground_mask.max() * 255).astype(np.uint8)
    
    foreground_img_cropped, foreground_mask_cropped = apply_smart_crop(
        foreground_img, foreground_mask, target_height, target_width
    )
    
    foreground_img_cropped.save(os.path.join(images_dir, f"{frame_name}_foreground.png"))
    cv2.imwrite(os.path.join(masks_dir, f"{frame_name}_foreground_mask.png"), foreground_mask_cropped)
    cv2.imwrite(os.path.join(masks_dir, f"{frame_name}_background_mask.png"), invert_mask(foreground_mask_cropped))
    
    return foreground_mask_cropped


def _generate_scene_data(frame_name: str, original_img_cropped: Image.Image,
                        foreground_mask_cropped: np.ndarray, scene_dir: str) -> None:
    """生成scene数据：前景bbox填白"""
    scene_image, scene_mask = generate_scene(np.array(original_img_cropped), foreground_mask_cropped)
    Image.fromarray(scene_image).save(os.path.join(scene_dir, f"{frame_name}_scene_image.png"))
    Image.fromarray(scene_mask).save(os.path.join(scene_dir, f"{frame_name}_scene_mask.png"))


def process_single_frame(frame_idx: int, data_info: Dict[str, Any], rembg_session: Any,
                        output_dir: str, transform_params: Dict[str, Any],
                        target_params: Dict[str, int]) -> None:
    """处理单帧：多视角图像 + 原始图像 + 前景图像 + scene数据"""
    frame_name = f"{frame_idx:05d}"
    
    frame_dir = os.path.join(output_dir, frame_name)
    images_dir = os.path.join(frame_dir, "images")
    masks_dir = os.path.join(frame_dir, "masks")
    scene_dir = os.path.join(frame_dir, "scene")
    
    for dir_path in [images_dir, masks_dir, scene_dir]:
        os.makedirs(dir_path, exist_ok=True)
    
    _process_multiview_images(frame_name, frame_idx, data_info, rembg_session, transform_params,
                             target_params, images_dir, masks_dir)
    original_img_cropped, _ = _process_original_image(frame_name, data_info, target_params, images_dir, masks_dir)
    foreground_mask_cropped = _process_foreground_image(frame_name, data_info, target_params, images_dir, masks_dir)
    _generate_scene_data(frame_name, original_img_cropped, foreground_mask_cropped, scene_dir)


def main():
    """主函数：读取多视角数据 -> 逆变换 -> rembg -> 生成训练数据集"""
    parser = argparse.ArgumentParser(description='步骤1: 数据准备脚本')
    
    parser.add_argument('--input_dir', type=str, required=True,
                       help='输入根目录')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='输出目录')
    parser.add_argument('--n_frames', type=int, default=None,
                       help='处理帧数（默认全部）')
    parser.add_argument('--debug', action='store_true',
                       help='Debug模式（只处理前3帧）')
    parser.add_argument('--output_height', type=int, default=DEFAULT_OUTPUT_HEIGHT,
                       help='输出图像高度')
    parser.add_argument('--output_width', type=int, default=DEFAULT_OUTPUT_WIDTH,
                       help='输出图像宽度')
    parser.add_argument('--auto_divisible', type=int, default=None,
                       help='自动调整尺寸使其能被指定数字整除')
    parser.add_argument('--overwrite', type=bool, default=True,
                       help='自动覆盖已存在的输出目录')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("🚀 步骤1: 数据准备")
    print("=" * 80)
    print(f"输入: {args.input_dir}")
    print(f"输出: {args.output_dir}")
    print("=" * 80)
    
    # 检查并处理输出目录
    if os.path.exists(args.output_dir):
        if args.overwrite:
            print(f"\n{LOG_WARNING} 删除旧输出目录: {args.output_dir}")
            shutil.rmtree(args.output_dir)
        else:
            print(f"\n{LOG_ERROR} 输出目录已存在，请使用 --overwrite True")
            sys.exit(1)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 初始化
    print(f"\n{LOG_PROCESS} 初始化rembg...")
    rembg_session = rembg.new_session()
    
    # 验证数据
    data_info = detect_and_validate_input(args.input_dir)
    
    # 确定处理帧数
    n_frames_to_process = args.n_frames or data_info['n_frames']
    if args.debug:
        n_frames_to_process = min(DEBUG_FRAMES, n_frames_to_process)
        print(f"\n{LOG_WARNING} Debug模式: 只处理前{n_frames_to_process}帧")
    
    # 读取简化的变换参数（必需）
    import json
    json_path = os.path.join(args.input_dir, 'preprocess_transform_params.json')
    
    if not os.path.exists(json_path):
        raise FileNotFoundError(
            f"找不到变换参数文件: {json_path}\n"
            f"请确保使用新版本的 0_0_gen_multiviews.py 生成数据"
        )
    
    print(f"\n{LOG_INFO} 读取变换参数: {json_path}")
    
    with open(json_path, 'r') as f:
        transform_params = json.load(f)
    
    # 验证必需字段
    required_fields = ['side_len', 'target_size', 'original_size']
    for field in required_fields:
        if field not in transform_params:
            raise ValueError(f"JSON缺少必需字段: {field}")
    
    print(f"{LOG_INFO} 填充尺寸: {transform_params['side_len']}, "
          f"目标尺寸: {transform_params['target_size']}, "
          f"原始尺寸: {transform_params['original_size'][1]}x{transform_params['original_size'][0]}")
    
    # 计算裁剪参数
    sample_img = Image.open(list(data_info['downsampled_original_dir'].glob("*.png"))[0])
    orig_width, orig_height = sample_img.size
    
    if args.auto_divisible:
        target_height = (orig_height // args.auto_divisible) * args.auto_divisible
        target_width = (orig_width // args.auto_divisible) * args.auto_divisible
    else:
        target_height, target_width = args.output_height, args.output_width
    
    target_params = {'height': target_height, 'width': target_width}
    print(f"\n{LOG_INFO} 目标输出尺寸: {target_width}×{target_height}")
    print(f"{LOG_INFO} 处理流程: 576×576 → {transform_params['side_len']}×{transform_params['side_len']} → pad到 {target_width}×{target_height}")
    
    # 验证数据完整性
    if not validate_data_integrity(data_info, n_frames_to_process):
        print(f"\n{LOG_ERROR} 数据验证失败，请检查输入数据")
        if not args.debug:
            sys.exit(1)
        else:
            print(f"{LOG_WARNING} Debug模式，继续处理...")
    
    # 处理所有帧
    print(f"\n{LOG_PROCESS} 开始处理帧...")
    for frame_idx in tqdm(range(n_frames_to_process), desc="处理进度"):
        try:
            process_single_frame(
                frame_idx, data_info, rembg_session, 
                args.output_dir, transform_params, target_params
            )
        except Exception as e:
            print(f"\n{LOG_ERROR} 处理帧{frame_idx:05d}失败: {str(e)}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
    # 打印处理摘要
    print("\n" + "=" * 80)
    print(f"{LOG_SUCCESS} 处理完成！")
    print("=" * 80)
    print(f"处理统计:")
    print(f"  - 处理帧数: {n_frames_to_process}")
    print(f"  - 视角数量: {data_info['n_views']}")
    print(f"  - 输出尺寸: {target_width}×{target_height}")
    print(f"  - 输出目录: {args.output_dir}")
    print("=" * 80)


if __name__ == '__main__':
    main()

