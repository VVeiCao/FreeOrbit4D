#!/usr/bin/env python3
"""
步骤1.1：背景点云生成脚本 - 批量生成全局背景和每帧前景点云

功能：
    使用形态学膨胀分离前景/背景，生成全局背景点云和每帧前景mapping
    
流程：
    1. 批量推理所有帧（DPG模型，一次性处理）
    2. 分辨率变换：原始 → 等比缩放 → padding(14整除) → 推理 → 去padding → 缩放回原始
    3. 形态学膨胀foreground_mask（默认5像素）
    4. 分离前景/背景（膨胀内=前景，膨胀外=背景）
    5. 背景点云处理：Confidence过滤 → Voxel下采样 → 离群点去除
    6. 保存前景/背景点云和相机参数

输入结构（步骤0.1的输出）：
    prepared_dir/
    ├── {frame_id}/
    │   ├── images/
    │   │   └── {frame_id}_original.png       # 原始图像
    │   └── masks/
    │       └── {frame_id}_foreground_mask.png  # 前景mask

输出结构：
    prepared_dir/
    ├── {frame_id}/
    │   ├── pointcloud/
    │   │   └── {frame_id}_foreground_singleview.ply  # 单视角前景点云
    │   ├── resized_input/
    │   │   ├── {frame_id}_original_resized.png       # resize后图像
    │   │   └── {frame_id}_foreground_mask_resized.png  # resize后mask
    │   └── {frame_id}_v4_1_foreground_mapping.npz     # 前景mapping+相机参数
    ├── global_background.ply                           # 全局背景点云
    └── global_camera.json                              # 全局相机参数

使用示例：
    # 单个场景
    python scripts/1_1_run_background.py \
        --folder outputs/prepared/camel
    
    # 批量处理
    for scene in camel bear cows hike; do
        python scripts/1_1_run_background.py \
            --folder outputs/prepared/$scene
    done

主要参数：
    --folder: 数据目录（步骤0.1的输出）
    --num_frames: 处理帧数（默认全部）
    --point_source: 点云来源（point_map/backproject，默认backproject）
    --pad_pixels: mask膨胀像素数（默认5）
    --conf_threshold: 背景Confidence阈值（默认1.0）
    --voxel_size: Voxel下采样体素大小（默认0.001）
    --outlier_nb_neighbors: 离群点检测近邻数（默认500）
    --outlier_std_ratio: 离群点检测标准差倍数（默认1.5）
"""
import os
import sys
import cv2
import open3d as o3d
import torch
import argparse
import json
import numpy as np
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

# 添加page-4d到Python路径（用于导入模型）
page_4d_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'page-4d')
sys.path.insert(0, page_4d_path)

# 导入模型相关
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt_t_mask_mlp_fin10.models.vggt import VGGT as DPG

# 导入本地工具函数
from utils.image_loader import load_and_preprocess_images_aspect_ratio, load_and_preprocess_masks_aspect_ratio

def backproject_depth_to_points_batch(depths, intrinsics, extrinsics_3x4):
    """深度图反投影到世界坐标
    
    Args:
        depths: 深度图 (B, H, W)
        intrinsics: 相机内参 (B, 3, 3)
        extrinsics_3x4: 相机外参 (B, 3, 4)
    
    Returns:
        Tensor: 世界坐标点 (B, H*W, 3)
    """
    B, H, W = depths.shape
    device = depths.device
    
    # 创建像素网格
    y, x = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
    pixels = torch.stack([x, y, torch.ones_like(x)], dim=-1).reshape(1, H*W, 3).expand(B, -1, -1).float()
    
    # 像素坐标 → 相机坐标
    cam_coords = torch.bmm(pixels, torch.inverse(intrinsics).transpose(1, 2))
    cam_points = cam_coords * depths.reshape(B, -1, 1)
    cam_points_h = torch.cat([cam_points, torch.ones((B, cam_points.shape[1], 1), device=device, dtype=cam_points.dtype)], dim=-1)
    
    # 相机坐标 → 世界坐标
    bottom = torch.tensor([[0, 0, 0, 1]], device=device, dtype=cam_points.dtype).expand(B, 1, 4)
    T_wc = torch.inverse(torch.cat([extrinsics_3x4, bottom], dim=1))
    world_points = torch.bmm(cam_points_h, T_wc.transpose(1, 2))[:, :, :3]
    
    return world_points

def scale_intrinsics(intrinsics, old_size, new_size):
    """缩放相机内参
    
    Args:
        intrinsics: 内参矩阵 (B, 3, 3) 或 (3, 3)
        old_size: 原始尺寸 (H, W)
        new_size: 新尺寸 (H, W)
    
    Returns:
        Tensor: 缩放后的内参
    """
    scale_x = new_size[1] / old_size[1]  # W_new / W_old
    scale_y = new_size[0] / old_size[0]  # H_new / H_old
    
    intrinsics_scaled = intrinsics.clone()
    intrinsics_scaled[..., 0, 0] *= scale_x  # fx
    intrinsics_scaled[..., 1, 1] *= scale_y  # fy
    intrinsics_scaled[..., 0, 2] *= scale_x  # cx
    intrinsics_scaled[..., 1, 2] *= scale_y  # cy
    
    return intrinsics_scaled

def compute_padded_mask(foreground_mask, pad_pixels=20):
    """形态学膨胀前景mask
    
    Args:
        foreground_mask: 前景mask (H, W)
        pad_pixels: 膨胀像素数（默认20）
    
    Returns:
        np.ndarray: 膨胀后的mask (H, W) bool数组
    """
    if not np.any(foreground_mask):
        return np.zeros_like(foreground_mask, dtype=bool)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (pad_pixels*2+1, pad_pixels*2+1))
    padded_mask_uint8 = cv2.dilate((foreground_mask.astype(np.uint8)*255), kernel, iterations=1)
    padded_mask = padded_mask_uint8 > 0
    
    return padded_mask

def process(model, image_names, mask_names, frame_ids, device, point_source='backproject', args=None):
    """批量处理：生成全局背景点云 + 每帧前景点云
    
    流程：
        1. 批量推理所有帧（DPG模型）
        2. 分辨率变换：原始 → 缩放 → padding → 推理 → 去padding → 还原
        3. 提取点云（point_map或backproject）
        4. 逐帧处理：膨胀mask → 分离前景/背景 → 保存
        5. 合并背景：Confidence过滤 → Voxel下采样 → 离群点去除
    
    Args:
        model: DPG模型
        image_names: 图像路径列表
        mask_names: mask路径列表
        frame_ids: 帧ID列表
        device: 计算设备
        point_source: 点云来源（point_map/backproject）
        args: 命令行参数
    """
    
    # ===== 步骤1: 加载并预处理（保持宽高比，padding到能被14整除） =====
    images_padded, padded_sizes, scaled_sizes, original_sizes, pad_coords = \
        load_and_preprocess_images_aspect_ratio(image_names, target_long_edge=518)
    masks_padded, _, _, _, _ = \
        load_and_preprocess_masks_aspect_ratio(mask_names, target_long_edge=518)
    
    # 获取尺寸信息（使用第一张图像的尺寸，假设所有图像尺寸相同）
    H_padded, W_padded = padded_sizes[0].tolist()  # 推理尺寸，例如 (308, 518)
    H_scaled, W_scaled = scaled_sizes[0].tolist()  # 缩放后尺寸，例如 (299, 518)
    H_orig, W_orig = original_sizes[0].tolist()    # 原始尺寸，例如 (480, 832)
    pad_top, pad_bottom, pad_left, pad_right = pad_coords[0].tolist()
    
    # ===== 步骤2: 准备模型输入 =====
    images_inference = images_padded.to(device).unsqueeze(0)  # (1, N, 3, H_padded, W_padded)
    masks_inference = masks_padded.to(device).unsqueeze(0)    # (1, N, 1, H_padded, W_padded)
    
    print(f"✓ 加载 {len(image_names)} 张图片 ({H_orig}×{W_orig} → {H_scaled}×{W_scaled} → padding到 {H_padded}×{W_padded})")
    print(f"✓ 运行推理...")
    
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
    num_frames = point_map_infer.size(1)
    
    # 去除padding：裁剪到缩放后尺寸
    depth_map_infer_squeezed = depth_map_infer.squeeze(-1)  # (1, N, H_padded, W_padded)
    depth_map_scaled = depth_map_infer_squeezed[:, :, pad_top:pad_top+H_scaled, pad_left:pad_left+W_scaled]
    depth_conf_scaled = depth_conf_infer[:, :, pad_top:pad_top+H_scaled, pad_left:pad_left+W_scaled]
    
    # Point map: (1, N, H_padded, W_padded, 3) → 裁剪
    point_map_infer_reshaped = point_map_infer.squeeze(0).permute(0, 3, 1, 2)  # (N, 3, H_padded, W_padded)
    point_map_scaled = point_map_infer_reshaped[:, :, pad_top:pad_top+H_scaled, pad_left:pad_left+W_scaled]
    point_conf_scaled = point_conf_infer[:, :, pad_top:pad_top+H_scaled, pad_left:pad_left+W_scaled]
    
    # Resize到原始尺寸
    depth_map_orig = F.interpolate(depth_map_scaled, size=(H_orig, W_orig), mode='bilinear', align_corners=False)
    depth_conf_orig = F.interpolate(depth_conf_scaled, size=(H_orig, W_orig), mode='bilinear', align_corners=False)
    point_map_orig = F.interpolate(point_map_scaled, size=(H_orig, W_orig), mode='bilinear', align_corners=False)
    point_conf_orig = F.interpolate(point_conf_scaled, size=(H_orig, W_orig), mode='bilinear', align_corners=False)
    
    point_map_orig = point_map_orig.permute(0, 2, 3, 1)  # (N, H_orig, W_orig, 3)
    
    # ===== 步骤5: 生成点云（在原始尺寸空间） =====
    if point_source == 'point_map':
        world_points_orig = point_map_orig  # (N, H_orig, W_orig, 3)
        conf_orig = point_conf_orig.squeeze(0)  # (N, H_orig, W_orig)
    elif point_source == 'backproject':
        # 缩放内参: 推理尺寸 → 原始尺寸
        intrinsic_orig = scale_intrinsics(
            intrinsic_infer[0], 
            old_size=(H_padded, W_padded), 
            new_size=(H_orig, W_orig)
        )
        world_points_orig = backproject_depth_to_points_batch(
            depth_map_orig[0], intrinsic_orig, extrinsic_infer[0]
        )  # (N, H_orig*W_orig, 3)
        world_points_orig = world_points_orig.reshape(num_frames, H_orig, W_orig, 3)
        conf_orig = depth_conf_orig.squeeze(0)  # (N, H_orig, W_orig)
    else:
        raise ValueError(f"未知point_source: {point_source}")
    
    print(f"✓ 点云来源: {point_source}")
    
    # 准备图像和mask（去除padding并resize到原始尺寸）
    images_scaled = images_padded[:, :, pad_top:pad_top+H_scaled, pad_left:pad_left+W_scaled]
    masks_scaled = masks_padded[:, :, pad_top:pad_top+H_scaled, pad_left:pad_left+W_scaled]
    
    images_orig = F.interpolate(images_scaled, size=(H_orig, W_orig), mode='bilinear', align_corners=False)
    masks_orig = F.interpolate(masks_scaled, size=(H_orig, W_orig), mode='bilinear', align_corners=False)
    
    images_orig_cpu = images_orig.cpu().permute(0, 2, 3, 1)  # (N, H_orig, W_orig, 3)
    masks_orig_cpu = masks_orig.cpu()  # (N, 1, H_orig, W_orig)
    
    # ===== 步骤6: 处理每帧 =====
    
    print(f"\n处理 {num_frames} 帧 ({H_orig}×{W_orig}), 膨胀 {args.pad_pixels} 像素\n")
    
    all_background_points, all_background_colors, all_background_confs = [], [], []
    camera_data = {}
    output_dir = args.folder
    
    # 使用进度条处理每帧
    for frame_idx in tqdm(range(num_frames), desc="处理帧", unit="帧"):
        frame_id = frame_ids[frame_idx]
        
        # 计算调整后的相机参数（对应原始尺寸）
        # 推理尺寸 → 原始尺寸
        intrinsic_orig_frame = scale_intrinsics(
            intrinsic_infer[0][frame_idx].unsqueeze(0),
            old_size=(H_padded, W_padded),
            new_size=(H_orig, W_orig)
        )
        
        # 保存相机参数
        camera_data[frame_id] = {
            'extrinsic': extrinsic_infer[0][frame_idx].cpu().numpy().tolist(),
            'intrinsic': intrinsic_orig_frame.squeeze(0).cpu().numpy().tolist()
        }
        
        # 获取原始尺寸的数据
        points_map = world_points_orig[frame_idx].cpu().numpy()  # (H_orig, W_orig, 3)
        colors_map = images_orig_cpu[frame_idx].numpy()  # (H_orig, W_orig, 3)
        conf_map = (conf_orig[frame_idx].cpu().squeeze(-1) if conf_orig[frame_idx].dim() > 2
                   else conf_orig[frame_idx].cpu()).numpy()  # (H_orig, W_orig)
        fg_mask_binary = (masks_orig_cpu[frame_idx].squeeze(0).numpy() > 0.5)  # (H_orig, W_orig)
        
        # 膨胀mask
        padded_mask = compute_padded_mask(fg_mask_binary, pad_pixels=args.pad_pixels)
        bg_mask_2d = ~padded_mask
        fg_mask_2d = fg_mask_binary
        
        # 提取前景/背景点
        fg_points, fg_colors = points_map[fg_mask_2d], colors_map[fg_mask_2d]
        bg_points, bg_colors = points_map[bg_mask_2d], colors_map[bg_mask_2d]
        bg_confs = conf_map[bg_mask_2d]
        
        all_background_points.append(bg_points)
        all_background_colors.append(bg_colors)
        all_background_confs.append(bg_confs)
        
        # 创建输出目录
        frame_dir = os.path.join(output_dir, frame_id)
        pointcloud_dir = os.path.join(frame_dir, 'pointcloud')
        resized_input_dir = os.path.join(frame_dir, 'resized_input')
        os.makedirs(pointcloud_dir, exist_ok=True)
        os.makedirs(resized_input_dir, exist_ok=True)
        
        # 保存resize后的图像和mask（裁剪到原始尺寸）
        resized_image_np = (colors_map * 255).astype(np.uint8)
        resized_mask_np = (fg_mask_binary * 255).astype(np.uint8)
        
        Image.fromarray(resized_image_np).save(os.path.join(resized_input_dir, f"{frame_id}_original_resized.png"))
        Image.fromarray(resized_mask_np).save(os.path.join(resized_input_dir, f"{frame_id}_foreground_mask_resized.png"))
        
        # 保存前景点云
        if len(fg_points) > 0:
            fg_pcd = o3d.geometry.PointCloud()
            fg_pcd.points = o3d.utility.Vector3dVector(fg_points)
            fg_pcd.colors = o3d.utility.Vector3dVector(fg_colors)
            o3d.io.write_point_cloud(os.path.join(pointcloud_dir, f"{frame_id}_foreground_singleview.ply"), fg_pcd)
        
        # 保存前景mapping（v4_1版本 + 相机参数）
        mapping_path = os.path.join(frame_dir, f"{frame_id}_v4_1_foreground_mapping.npz")
        np.savez(mapping_path,
                foreground_pointmap=points_map,
                foreground_mask=fg_mask_2d,
                padded_mask=padded_mask,
                color=colors_map,
                confidence=conf_map,
                pad_pixels=args.pad_pixels,
                original_size=[H_orig, W_orig],
                intrinsic=intrinsic_orig_frame.squeeze(0).cpu().numpy(),
                extrinsic=extrinsic_infer[0][frame_idx].cpu().numpy())
    
    # 合并背景点云
    print(f"\n{'='*80}")
    print(f"处理全局背景点云...")
    print(f"{'='*80}\n")
    
    global_bg_points = np.concatenate(all_background_points)
    global_bg_colors = np.concatenate(all_background_colors)
    global_bg_confs = np.concatenate(all_background_confs)
    
    print(f"✓ 全局背景: {len(global_bg_points):,} 点, conf [{global_bg_confs.min():.1f}, {global_bg_confs.max():.1f}]")
    
    if len(global_bg_points) > 0:
        # Confidence过滤
        print(f"\n🎯 Confidence过滤 (阈值={args.conf_threshold:.1f})...")
        conf_mask = global_bg_confs >= args.conf_threshold
        bg_points_conf = global_bg_points[conf_mask]
        bg_colors_conf = global_bg_colors[conf_mask]
        
        print(f"  {len(global_bg_points):,} → {len(bg_points_conf):,} 点")
        
        if len(bg_points_conf) == 0:
            print(f"  ⚠️  过滤后为空，跳过保存（建议降低阈值）")
            return
        
        # 创建点云
        bg_pcd = o3d.geometry.PointCloud()
        bg_pcd.points = o3d.utility.Vector3dVector(bg_points_conf)
        bg_pcd.colors = o3d.utility.Vector3dVector(
            np.clip(bg_colors_conf, 0, 1) if bg_colors_conf.max() > 1.0 else bg_colors_conf
        )
        
        # Voxel下采样（可选）
        if args.voxel_size:
            print(f"\n📦 Voxel下采样 (size={args.voxel_size:.6f})...")
            bg_pcd_downsampled = bg_pcd.voxel_down_sample(voxel_size=args.voxel_size)
            print(f"  {len(bg_pcd.points):,} → {len(bg_pcd_downsampled.points):,} 点")
        else:
            bg_pcd_downsampled = bg_pcd
        
        # 离群点去除
        print(f"\n🧹 离群点去除...")
        nb_neighbors = min(args.outlier_nb_neighbors, len(bg_pcd_downsampled.points))
        if nb_neighbors > 1:
            bg_pcd_final, _ = bg_pcd_downsampled.remove_statistical_outlier(
                nb_neighbors=nb_neighbors, std_ratio=args.outlier_std_ratio
            )
            print(f"  {len(bg_pcd_downsampled.points):,} → {len(bg_pcd_final.points):,} 点")
        else:
            bg_pcd_final = bg_pcd_downsampled
            print(f"  ⚠️  近邻数不足，跳过")
        
        # 保存
        bg_ply_path = os.path.join(output_dir, "global_background.ply")
        o3d.io.write_point_cloud(bg_ply_path, bg_pcd_final)
        print(f"\n✓ 保存: {bg_ply_path}")
        print(f"  流程: {len(global_bg_points):,} → conf → {len(bg_points_conf):,} " +
              (f"→ voxel → {len(bg_pcd_downsampled.points):,} " if args.voxel_size else "") +
              f"→ 离群点 → {len(bg_pcd_final.points):,}")
    else:
        print(f"\n⚠ 背景点云为空")
    
    # 保存相机参数
    print(f"\n💾 保存相机参数...")
    camera_json_data = {
        "image_size": [H_orig, W_orig],
        "coordinate_system": "y-down (original)"
    }
    camera_json_data.update(camera_data)
    
    camera_json_path = os.path.join(output_dir, "global_camera.json")
    with open(camera_json_path, "w") as f:
        json.dump(camera_json_data, f, indent=4)
    print(f"✓ {len(camera_data)} 帧相机参数 → {camera_json_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='步骤1.1: 生成全局背景点云和每帧前景点云（DPG模型，批量处理）')
    parser.add_argument('--folder', type=str, required=True, help='数据目录（步骤0.1的输出）')
    parser.add_argument('--num_frames', type=int, default=None, help='处理帧数（默认全部）')
    parser.add_argument('--point_source', type=str, default='backproject', 
                       choices=['point_map', 'backproject'], help='点云来源（默认backproject）')
    parser.add_argument('--pad_pixels', type=int, default=5, help='mask膨胀像素数（默认5）')
    parser.add_argument('--conf_threshold', type=float, default=1.0, help='背景Confidence阈值（默认1.0）')
    parser.add_argument('--voxel_size', type=float, default=0.001, help='Voxel下采样体素大小（默认0.001）')
    parser.add_argument('--outlier_nb_neighbors', type=int, default=500, help='离群点检测近邻数（默认500）')
    parser.add_argument('--outlier_std_ratio', type=float, default=1.5, help='离群点检测标准差倍数（默认1.5）')
    args = parser.parse_args()
    
    # 扫描帧目录
    subdirs = sorted([d for d in os.listdir(args.folder) 
                     if os.path.isdir(os.path.join(args.folder, d)) 
                     and os.path.exists(os.path.join(args.folder, d, "images"))])
    if args.num_frames:
        subdirs = subdirs[:args.num_frames]
    
    print("=" * 80)
    print("🚀 步骤1.1: 背景点云生成（DPG + 批量处理）")
    print("=" * 80)
    print(f"数据目录: {args.folder}")
    print(f"帧数: {len(subdirs)}")
    print(f"膨胀: {args.pad_pixels}px")
    print(f"点云来源: {args.point_source}")
    print("=" * 80 + "\n")
    
    # 构建路径列表
    image_names, mask_names, frame_ids = [], [], []
    for subdir in subdirs:
        image_names.append(os.path.join(args.folder, subdir, "images", f"{subdir}_original.png"))
        mask_names.append(os.path.join(args.folder, subdir, "masks", f"{subdir}_foreground_mask.png"))
        frame_ids.append(subdir)
    
    # 加载模型
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 获取项目根目录和权重路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    dpg_weight_path = os.path.join(project_root, 'page-4d', 'weights', 'checkpoint_150.pt')
    
    print("加载 DPG 模型...")
    if not os.path.exists(dpg_weight_path):
        raise FileNotFoundError(f"DPG权重文件不存在: {dpg_weight_path}")
    
    model = DPG()
    model.load_state_dict(torch.load(dpg_weight_path, map_location=device)['model'], strict=False)
    model.to(device).eval()
    print("✓ DPG 模型加载完成\n")
    
    process(model, image_names, mask_names, frame_ids, device, point_source=args.point_source, args=args)
    
    print("\n" + "=" * 80)
    print("✅ 处理完成！")
    print("=" * 80)
    print(f"输出目录: {args.folder}")
    print("=" * 80 + "\n")


