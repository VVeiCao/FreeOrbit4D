#!/usr/bin/env python3
"""
步骤1.0：前景点云生成脚本 - 逐帧生成前景点云（5视角，480p分辨率）

功能：
    逐帧处理，每帧生成5视角合并的前景点云和完整mapping
    
流程：
    1. 逐帧处理（每帧单独VGGT推理）
    2. 5视角：foreground(第1位), 0, 1, 2, 3
    3. 分辨率变换（以480×832为例）：
       原始480×832 → 等比缩放299×518 → padding到308×518（能被14整除）
       → 推理308×518 → 去padding到299×518 → 等比缩放回480×832
    4. 提取点云（point_map）
    5. 保存PLY和NPZ文件

输入结构（步骤0.1的输出）：
    prepared_dir/
    ├── {frame_id}/
    │   ├── images/
    │   │   ├── {frame_id}_foreground.png
    │   │   ├── {frame_id}_0.png
    │   │   ├── {frame_id}_1.png
    │   │   ├── {frame_id}_2.png
    │   │   └── {frame_id}_3.png
    │   └── masks/
    │       ├── {frame_id}_foreground_mask.png
    │       ├── {frame_id}_0_mask.png
    │       ├── {frame_id}_1_mask.png
    │       ├── {frame_id}_2_mask.png
    │       └── {frame_id}_3_mask.png

输出：
    - {frame_id}/pointcloud/{frame_id}_foreground_5_views.ply    # 5视角合并点云
    - {frame_id}/pointcloud/{frame_id}_foreground_5_views.npz    # 完整mapping

使用示例：
    # 单个场景
    python scripts/1_0_run_foreground.py \
        --folder outputs/prepared/camel

    python scripts/1_0_run_foreground.py \
        --folder outputs/prepared/bear
    
    # 批量处理
    for scene in camel bear cows hike; do
        python scripts/1_0_run_foreground.py \
            --folder outputs/prepared/$scene
    done

主要参数：
    --folder: 数据目录（步骤0.1的输出）
    --num_frames: 处理帧数（默认全部）
"""
import os
import sys
import open3d as o3d
import torch
import argparse
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm

# 添加page-4d到Python路径（用于导入模型）
page_4d_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'page-4d')
sys.path.insert(0, page_4d_path)

# 导入模型相关
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.models.vggt import VGGT

# 导入本地工具函数
from utils.image_loader import load_and_preprocess_images_aspect_ratio, load_and_preprocess_masks_aspect_ratio

def scale_intrinsics(intrinsics, old_size, new_size):
    """缩放相机内参到新分辨率
    
    Args:
        intrinsics: (B, 3, 3) 或 (3, 3) 张量
        old_size: (H_old, W_old) 元组
        new_size: (H_new, W_new) 元组
    
    Returns:
        intrinsics_scaled: 缩放后的内参
    """
    scale_x = new_size[1] / old_size[1]  # W_new / W_old
    scale_y = new_size[0] / old_size[0]  # H_new / H_old
    
    intrinsics_scaled = intrinsics.clone()
    intrinsics_scaled[..., 0, 0] *= scale_x  # fx
    intrinsics_scaled[..., 1, 1] *= scale_y  # fy
    intrinsics_scaled[..., 0, 2] *= scale_x  # cx
    intrinsics_scaled[..., 1, 2] *= scale_y  # cy
    
    return intrinsics_scaled

def process_frame(model, frame_id, image_paths, mask_paths, device, dtype, args):
    """处理单帧，生成5视角点云和mapping（原始尺寸输出）"""
    
    # ===== 步骤1: 加载并预处理（保持宽高比，padding到能被14整除） =====
    images_padded, padded_sizes, scaled_sizes, original_sizes, pad_coords = \
        load_and_preprocess_images_aspect_ratio(image_paths, target_long_edge=518)
    masks_padded, _, _, _, _ = \
        load_and_preprocess_masks_aspect_ratio(mask_paths, target_long_edge=518)
    
    # 获取尺寸信息（使用第一张图像的尺寸，假设所有图像尺寸相同）
    H_padded, W_padded = padded_sizes[0].tolist()  # 推理尺寸，例如 (308, 518)
    H_scaled, W_scaled = scaled_sizes[0].tolist()  # 缩放后尺寸，例如 (299, 518)
    H_orig, W_orig = original_sizes[0].tolist()    # 原始尺寸，例如 (480, 832)
    pad_top, pad_bottom, pad_left, pad_right = pad_coords[0].tolist()
    
    # ===== 步骤2: 准备模型输入 =====
    images_inference = images_padded.to(device).unsqueeze(0)  # (1, 5, 3, H_padded, W_padded)
    masks_inference = masks_padded.to(device).unsqueeze(0)    # (1, 5, 1, H_padded, W_padded)
    
    # ===== 步骤3: 推理 =====
    with torch.no_grad():
        with torch.amp.autocast('cuda', dtype=dtype):
            aggregated_tokens_list, ps_idx = model.aggregator(images_inference)
            pose_enc = model.camera_head(aggregated_tokens_list)[-1]
            extrinsic_infer, intrinsic_infer = pose_encoding_to_extri_intri(pose_enc, images_inference.shape[-2:])
        
        with torch.cuda.amp.autocast(enabled=False):
            depth_map_infer, depth_conf_infer = model.depth_head(aggregated_tokens_list, images_inference, ps_idx)
            point_map_infer, point_conf_infer = model.point_head(aggregated_tokens_list, images_inference, ps_idx)
    
    # ===== 步骤4: 去除padding，resize回原始尺寸 =====
    # 去除padding：裁剪到缩放后尺寸
    depth_map_infer_squeezed = depth_map_infer.squeeze(-1)  # (1, 5, H_padded, W_padded)
    depth_map_scaled = depth_map_infer_squeezed[:, :, pad_top:pad_top+H_scaled, pad_left:pad_left+W_scaled]
    depth_conf_scaled = depth_conf_infer[:, :, pad_top:pad_top+H_scaled, pad_left:pad_left+W_scaled]
    
    # Point map: (1, 5, H_padded, W_padded, 3) → 裁剪
    point_map_infer_reshaped = point_map_infer.squeeze(0).permute(0, 3, 1, 2)  # (5, 3, H_padded, W_padded)
    point_map_scaled = point_map_infer_reshaped[:, :, pad_top:pad_top+H_scaled, pad_left:pad_left+W_scaled]
    point_conf_scaled = point_conf_infer[:, :, pad_top:pad_top+H_scaled, pad_left:pad_left+W_scaled]
    
    # Resize到原始尺寸
    depth_map_orig = F.interpolate(depth_map_scaled, size=(H_orig, W_orig), mode='bilinear', align_corners=False)
    depth_conf_orig = F.interpolate(depth_conf_scaled, size=(H_orig, W_orig), mode='bilinear', align_corners=False)
    point_map_orig = F.interpolate(point_map_scaled, size=(H_orig, W_orig), mode='bilinear', align_corners=False)
    point_conf_orig = F.interpolate(point_conf_scaled, size=(H_orig, W_orig), mode='bilinear', align_corners=False)
    
    point_map_orig = point_map_orig.permute(0, 2, 3, 1)  # (5, H_orig, W_orig, 3)
    
    # ===== 步骤5: 生成点云（在原始尺寸空间，使用point_map） =====
    world_points_orig = point_map_orig  # (5, H_orig, W_orig, 3)
    conf_orig = point_conf_orig.squeeze(0)  # (5, H_orig, W_orig)
    
    # 准备图像和mask（去除padding并resize到原始尺寸）
    images_scaled = images_padded[:, :, pad_top:pad_top+H_scaled, pad_left:pad_left+W_scaled]
    masks_scaled = masks_padded[:, :, pad_top:pad_top+H_scaled, pad_left:pad_left+W_scaled]
    
    images_orig = F.interpolate(images_scaled, size=(H_orig, W_orig), mode='bilinear', align_corners=False)
    masks_orig = F.interpolate(masks_scaled, size=(H_orig, W_orig), mode='bilinear', align_corners=False)
    
    images_orig_cpu = images_orig.cpu().permute(0, 2, 3, 1)  # (5, H_orig, W_orig, 3)
    masks_orig_cpu = masks_orig.cpu()  # (5, 1, H_orig, W_orig)
    
    # ===== 步骤6: 保存结果 =====
    
    view_pointmaps = np.zeros((5, H_orig, W_orig, 3), dtype=np.float32)
    view_masks = np.zeros((5, H_orig, W_orig), dtype=bool)
    view_colormaps = np.zeros((5, H_orig, W_orig, 3), dtype=np.float32)
    view_confmaps = np.zeros((5, H_orig, W_orig), dtype=np.float32)
    foreground_points_list, foreground_colors_list = [], []
    view_names = ['foreground', '0', '1', '2', '3']
    
    for view_idx in range(5):
        # 获取原始尺寸的数据
        points_map = world_points_orig[view_idx].cpu().numpy()  # (H_orig, W_orig, 3)
        colors_map = images_orig_cpu[view_idx].numpy()  # (H_orig, W_orig, 3)
        conf_map = (conf_orig[view_idx].cpu().squeeze(-1) if conf_orig[view_idx].dim() > 2
                   else conf_orig[view_idx].cpu()).numpy()  # (H_orig, W_orig)
        mask_binary = (masks_orig_cpu[view_idx].squeeze(0).numpy() > 0.5)  # (H_orig, W_orig)
        
        # 保存完整mapping（不过滤）
        view_pointmaps[view_idx] = points_map
        view_masks[view_idx] = mask_binary
        view_colormaps[view_idx] = colors_map
        view_confmaps[view_idx] = conf_map
        
        # mask过滤（用于PLY）
        foreground_points_list.append(points_map[mask_binary])
        foreground_colors_list.append(colors_map[mask_binary])
    
    all_fg_points = np.concatenate(foreground_points_list)
    all_fg_colors = np.concatenate(foreground_colors_list)
    
    # 计算调整后的相机参数（对应原始尺寸）
    # 推理尺寸 → 原始尺寸
    intrinsics_final_list = []
    extrinsics_final_list = []
    for view_idx in range(5):
        intrinsic_orig_view = scale_intrinsics(
            intrinsic_infer[0][view_idx].unsqueeze(0),
            old_size=(H_padded, W_padded),
            new_size=(H_orig, W_orig)
        )
        intrinsics_final_list.append(intrinsic_orig_view.squeeze(0).cpu().numpy())
        extrinsics_final_list.append(extrinsic_infer[0][view_idx].cpu().numpy())
    
    # 保存结果
    pointcloud_dir = os.path.join(args.folder, frame_id, 'pointcloud')
    os.makedirs(pointcloud_dir, exist_ok=True)
    
    # 保存PLY（mask过滤）
    if len(all_fg_points) > 0:
        fg_pcd = o3d.geometry.PointCloud()
        fg_pcd.points = o3d.utility.Vector3dVector(all_fg_points)
        fg_pcd.colors = o3d.utility.Vector3dVector(all_fg_colors)
        o3d.io.write_point_cloud(os.path.join(pointcloud_dir, f"{frame_id}_foreground_5_views.ply"), fg_pcd)
    
    # 保存NPZ（完整mapping + 相机参数）
    mapping_path = os.path.join(pointcloud_dir, f"{frame_id}_foreground_5_views.npz")
    np.savez(mapping_path,
            foreground_pointmaps=view_pointmaps,
            foreground_masks=view_masks,
            colors=view_colormaps,
            confidences=view_confmaps,
            image_names=view_names,
            original_size=[H_orig, W_orig],
            intrinsics=np.array(intrinsics_final_list),
            extrinsics=np.array(extrinsics_final_list))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='步骤1.0: 逐帧生成前景点云（5视角，VGGT模型）')
    parser.add_argument('--folder', type=str, required=True, help='数据目录（步骤0.1的输出）')
    parser.add_argument('--num_frames', type=int, default=None, help='处理帧数（默认全部）')
    args = parser.parse_args()
    
    # 扫描帧目录
    subdirs = sorted([d for d in os.listdir(args.folder) 
                     if os.path.isdir(os.path.join(args.folder, d)) 
                     and os.path.exists(os.path.join(args.folder, d, "images"))])
    if args.num_frames:
        subdirs = subdirs[:args.num_frames]
    
    print("=" * 80)
    print("🚀 步骤1.0: 前景点云生成（VGGT + Point Map）")
    print("=" * 80)
    print(f"数据目录: {args.folder}")
    print(f"帧数: {len(subdirs)}")
    print("=" * 80 + "\n")
    
    # 加载模型
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("加载 VGGT 模型...")
    model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
    model.eval()
    print("✓ VGGT 模型加载完成\n")
    
    view_order = ['foreground', '0', '1', '2', '3']
    
    # 逐帧处理（带进度条）
    failed_frames = []
    for subdir in tqdm(subdirs, desc="处理进度", unit="帧"):
        frame_id = subdir
        
        # 构建路径
        image_dir = os.path.join(args.folder, subdir, "images")
        mask_dir = os.path.join(args.folder, subdir, "masks")
        image_paths = [os.path.join(image_dir, f"{frame_id}_{v}.png") for v in view_order]
        mask_paths = [os.path.join(mask_dir, f"{frame_id}_{v}_mask.png") for v in view_order]
        
        try:
            process_frame(model, frame_id, image_paths, mask_paths, device, dtype, args)
        except Exception as e:
            failed_frames.append((frame_id, str(e)))
    
    # 输出汇总
    print("\n" + "=" * 80)
    print("✅ 处理完成！")
    print("=" * 80)
    print(f"处理统计:")
    print(f"  - 总帧数: {len(subdirs)}")
    print(f"  - 成功: {len(subdirs) - len(failed_frames)}")
    if failed_frames:
        print(f"  - 失败: {len(failed_frames)}")
        print(f"\n失败帧详情:")
        for frame_id, error in failed_frames[:5]:  # 只显示前5个
            print(f"  - {frame_id}: {error}")
        if len(failed_frames) > 5:
            print(f"  ... 还有 {len(failed_frames) - 5} 个失败帧")
    print(f"  - 输出目录: {args.folder}")
    print("=" * 80 + "\n")


