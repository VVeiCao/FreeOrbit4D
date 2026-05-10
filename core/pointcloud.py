"""Point cloud generation module (VGGT foreground + DPG background)."""

import os
import sys
import cv2
import json
import numpy as np
import open3d as o3d
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from tqdm import tqdm

from utils.image_loader import load_and_preprocess_images_aspect_ratio, load_and_preprocess_masks_aspect_ratio
from utils.camera import scale_intrinsics
from utils.depth import backproject_depth_to_points_batch
from utils.pointcloud import save_pointcloud_ply, save_pointcloud_npz
from utils.file_io import find_frame_dirs, ensure_dir


class ForegroundPointCloudGenerator:
    """Foreground point cloud generator using VGGT with 5 views."""
    
    def __init__(self, 
                 device: str = 'cuda',
                 target_long_edge: int = 518,
                 divisor: int = 14):
        """
        Args:
            device: Compute device.
            target_long_edge: Target size for the long edge.
            divisor: Size must be divisible by this value.
        """
        self.device = device
        self.target_long_edge = target_long_edge
        self.divisor = divisor
        self.model = None
        self.dtype = None
    
    def load_model(self):
        """Load VGGT model."""
        page_4d_path = Path(__file__).parent.parent / 'page-4d'
        if str(page_4d_path) not in sys.path:
            sys.path.insert(0, str(page_4d_path))
        
        from vggt.models.vggt import VGGT
        
        print("Loading VGGT-1B model...")
        
        if torch.cuda.is_available():
            capability = torch.cuda.get_device_capability()
            self.dtype = torch.bfloat16 if capability[0] >= 8 else torch.float16
        else:
            self.dtype = torch.float32
        
        self.model = VGGT.from_pretrained("facebook/VGGT-1B").to(self.device)
        self.model.eval()
        
        print("VGGT model loaded.\n")
        return self.model
    
    def process_frame(self, 
                     frame_id: str,
                     image_paths: List[str],
                     mask_paths: List[str],
                     output_dir: str) -> Dict:
        """Process a single frame to generate 5-view point cloud and mapping.
        
        Args:
            frame_id: Frame ID.
            image_paths: List of 5 view image paths.
            mask_paths: List of 5 view mask paths.
            output_dir: Output directory.
        
        Returns:
            Result dict with frame_id, num_points, output_dir.
        """
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri
        
        # Step 1: Load and preprocess
        images_padded, padded_sizes, scaled_sizes, original_sizes, pad_coords = \
            load_and_preprocess_images_aspect_ratio(image_paths, target_long_edge=self.target_long_edge)
        masks_padded, _, _, _, _ = \
            load_and_preprocess_masks_aspect_ratio(mask_paths, target_long_edge=self.target_long_edge)
        
        H_padded, W_padded = padded_sizes[0].tolist()
        H_scaled, W_scaled = scaled_sizes[0].tolist()
        H_orig, W_orig = original_sizes[0].tolist()
        pad_top, pad_bottom, pad_left, pad_right = pad_coords[0].tolist()
        
        # Step 2: Prepare model input
        images_inference = images_padded.to(self.device).unsqueeze(0)
        masks_inference = masks_padded.to(self.device).unsqueeze(0)
        
        # Step 3: Inference
        with torch.no_grad():
            with torch.amp.autocast('cuda', dtype=self.dtype):
                aggregated_tokens_list, ps_idx = self.model.aggregator(images_inference)
                pose_enc = self.model.camera_head(aggregated_tokens_list)[-1]
                extrinsic_infer, intrinsic_infer = pose_encoding_to_extri_intri(
                    pose_enc, images_inference.shape[-2:]
                )
            
            with torch.amp.autocast('cuda', enabled=False):
                depth_map_infer, depth_conf_infer = self.model.depth_head(
                    aggregated_tokens_list, images_inference, ps_idx
                )
                point_map_infer, point_conf_infer = self.model.point_head(
                    aggregated_tokens_list, images_inference, ps_idx
                )
        
        # Step 4: Remove padding, resize to original size
        depth_map_infer_squeezed = depth_map_infer.squeeze(-1)
        depth_map_scaled = depth_map_infer_squeezed[:, :, pad_top:pad_top+H_scaled, pad_left:pad_left+W_scaled]
        depth_conf_scaled = depth_conf_infer[:, :, pad_top:pad_top+H_scaled, pad_left:pad_left+W_scaled]
        
        point_map_infer_reshaped = point_map_infer.squeeze(0).permute(0, 3, 1, 2)
        point_map_scaled = point_map_infer_reshaped[:, :, pad_top:pad_top+H_scaled, pad_left:pad_left+W_scaled]
        point_conf_scaled = point_conf_infer[:, :, pad_top:pad_top+H_scaled, pad_left:pad_left+W_scaled]
        
        depth_map_orig = F.interpolate(depth_map_scaled, size=(H_orig, W_orig), mode='bilinear', align_corners=False)
        depth_conf_orig = F.interpolate(depth_conf_scaled, size=(H_orig, W_orig), mode='bilinear', align_corners=False)
        point_map_orig = F.interpolate(point_map_scaled, size=(H_orig, W_orig), mode='bilinear', align_corners=False)
        point_conf_orig = F.interpolate(point_conf_scaled, size=(H_orig, W_orig), mode='bilinear', align_corners=False)
        
        point_map_orig = point_map_orig.permute(0, 2, 3, 1)
        
        # Step 5: Generate point cloud
        world_points_orig = point_map_orig
        conf_orig = point_conf_orig.squeeze(0)
        
        images_scaled = images_padded[:, :, pad_top:pad_top+H_scaled, pad_left:pad_left+W_scaled]
        masks_scaled = masks_padded[:, :, pad_top:pad_top+H_scaled, pad_left:pad_left+W_scaled]
        
        images_orig = F.interpolate(images_scaled, size=(H_orig, W_orig), mode='bilinear', align_corners=False)
        masks_orig = F.interpolate(masks_scaled, size=(H_orig, W_orig), mode='bilinear', align_corners=False)
        
        images_orig_cpu = images_orig.cpu().permute(0, 2, 3, 1)
        masks_orig_cpu = masks_orig.cpu()
        
        # Step 6: Organize data and save
        view_pointmaps = np.zeros((5, H_orig, W_orig, 3), dtype=np.float32)
        view_masks = np.zeros((5, H_orig, W_orig), dtype=bool)
        view_colormaps = np.zeros((5, H_orig, W_orig, 3), dtype=np.float32)
        view_confmaps = np.zeros((5, H_orig, W_orig), dtype=np.float32)
        foreground_points_list, foreground_colors_list = [], []
        view_names = ['foreground', '0', '1', '2', '3']
        
        for view_idx in range(5):
            points_map = world_points_orig[view_idx].cpu().numpy()
            colors_map = images_orig_cpu[view_idx].numpy()
            conf_map = (conf_orig[view_idx].cpu().squeeze(-1) if conf_orig[view_idx].dim() > 2
                       else conf_orig[view_idx].cpu()).numpy()
            mask_binary = (masks_orig_cpu[view_idx].squeeze(0).numpy() > 0.5)
            
            view_pointmaps[view_idx] = points_map
            view_masks[view_idx] = mask_binary
            view_colormaps[view_idx] = colors_map
            view_confmaps[view_idx] = conf_map
            
            foreground_points_list.append(points_map[mask_binary])
            foreground_colors_list.append(colors_map[mask_binary])
        
        all_fg_points = np.concatenate(foreground_points_list)
        all_fg_colors = np.concatenate(foreground_colors_list)
        
        # Compute camera parameters
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
        
        # Save results
        pointcloud_dir = Path(output_dir) / frame_id / 'pointcloud'
        ensure_dir(pointcloud_dir)
        
        if len(all_fg_points) > 0:
            fg_pcd = o3d.geometry.PointCloud()
            fg_pcd.points = o3d.utility.Vector3dVector(all_fg_points)
            fg_pcd.colors = o3d.utility.Vector3dVector(all_fg_colors)
            o3d.io.write_point_cloud(str(pointcloud_dir / f"{frame_id}_foreground_5_views.ply"), fg_pcd)
        
        mapping_path = pointcloud_dir / f"{frame_id}_foreground_5_views.npz"
        np.savez(str(mapping_path),
                foreground_pointmaps=view_pointmaps,
                foreground_masks=view_masks,
                colors=view_colormaps,
                confidences=view_confmaps,
                image_names=view_names,
                original_size=[H_orig, W_orig],
                intrinsics=np.array(intrinsics_final_list),
                extrinsics=np.array(extrinsics_final_list))
        
        return {
            'frame_id': frame_id,
            'num_points': len(all_fg_points),
            'output_dir': str(pointcloud_dir)
        }
    
    def batch_process(self, 
                     folder: str,
                     num_frames: Optional[int] = None):
        """Batch process all frames.
        
        Args:
            folder: Data directory.
            num_frames: Number of frames to process (None for all).
        
        Returns:
            List of result dicts.
        """
        frame_ids = find_frame_dirs(folder, max_frames=num_frames)
        
        if len(frame_ids) == 0:
            raise ValueError(f"No frame directories found: {folder}")
        
        if self.model is None:
            self.load_model()
        
        view_order = ['foreground', '0', '1', '2', '3']
        
        results = []
        failed_frames = []
        
        for frame_id in tqdm(frame_ids, desc="Foreground", unit="frame"):
            try:
                image_dir = Path(folder) / frame_id / "images"
                mask_dir = Path(folder) / frame_id / "masks"
                image_paths = [str(image_dir / f"{frame_id}_{v}.png") for v in view_order]
                mask_paths = [str(mask_dir / f"{frame_id}_{v}_mask.png") for v in view_order]
                
                result = self.process_frame(frame_id, image_paths, mask_paths, folder)
                results.append(result)
                
            except Exception as e:
                failed_frames.append((frame_id, str(e)))
        
        print(f"\nForeground done: {len(results)}/{len(frame_ids)} frames succeeded.")
        if failed_frames:
            print(f"  Failed: {len(failed_frames)}")
            for frame_id, error in failed_frames[:5]:
                print(f"    {frame_id}: {error}")
            if len(failed_frames) > 5:
                print(f"    ... and {len(failed_frames) - 5} more")
        
        return results
    
    @classmethod
    def from_config(cls, config):
        """Create generator from config."""
        from utils.config import Config
        if not isinstance(config, Config):
            config = Config(config)
        
        return cls(
            device=config.get('common.device', 'cuda'),
            target_long_edge=config.get('stage_1.foreground.target_long_edge', 518),
            divisor=config.get('stage_1.foreground.divisor', 14)
        )


class BackgroundPointCloudGenerator:
    """Background point cloud generator using DPG with single view."""
    
    def __init__(self,
                 device: str = 'cuda',
                 confidence_threshold: float = 50.0,
                 pad_pixels: int = 5,
                 voxel_size: float = 0.001,
                 outlier_nb_neighbors: int = 500,
                 outlier_std_ratio: float = 1.5,
                 point_source: str = 'backproject'):
        """
        Args:
            device: Compute device.
            confidence_threshold: Confidence percentile threshold (%), filters out the lowest X% of points.
            pad_pixels: Mask dilation in pixels.
            voxel_size: Voxel downsampling size.
            outlier_nb_neighbors: Number of neighbors for outlier detection.
            outlier_std_ratio: Std deviation ratio for outlier detection.
            point_source: Point source ('point_map' or 'backproject').
        """
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.pad_pixels = pad_pixels
        self.voxel_size = voxel_size
        self.outlier_nb_neighbors = outlier_nb_neighbors
        self.outlier_std_ratio = outlier_std_ratio
        self.point_source = point_source
        self.model = None
        self.dtype = None
    
    def load_model(self):
        """Load DPG model."""
        page_4d_path = Path(__file__).parent.parent / 'page-4d'
        if str(page_4d_path) not in sys.path:
            sys.path.insert(0, str(page_4d_path))
        
        from vggt_t_mask_mlp_fin10.models.vggt import VGGT as DPG
        
        print("Loading DPG model...")
        
        if torch.cuda.is_available():
            capability = torch.cuda.get_device_capability()
            self.dtype = torch.bfloat16 if capability[0] >= 8 else torch.float16
        else:
            self.dtype = torch.float32
        
        project_root = Path(__file__).parent.parent
        dpg_weight_path = project_root / 'checkpoints' / 'dpg' / 'checkpoint_150.pt'
        
        if not dpg_weight_path.exists():
            raise FileNotFoundError(f"DPG weight file not found: {dpg_weight_path}")
        
        self.model = DPG()
        self.model.load_state_dict(
            torch.load(str(dpg_weight_path), map_location=self.device, weights_only=False)['model'], 
            strict=False
        )
        self.model.to(self.device).eval()
        
        print("DPG model loaded.\n")
        return self.model
    
    def _compute_padded_mask(self, foreground_mask: np.ndarray) -> np.ndarray:
        """Dilate foreground mask morphologically."""
        if not np.any(foreground_mask):
            return np.zeros_like(foreground_mask, dtype=bool)
        
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (self.pad_pixels * 2 + 1, self.pad_pixels * 2 + 1)
        )
        padded_mask_uint8 = cv2.dilate(
            (foreground_mask.astype(np.uint8) * 255),
            kernel,
            iterations=1
        )
        padded_mask = padded_mask_uint8 > 0
        
        return padded_mask
    
    def process_batch(self,
                     image_paths: List[str],
                     mask_paths: List[str],
                     frame_ids: List[str]) -> Tuple[Dict, List[Dict]]:
        """Batch inference: generate global background + per-frame foreground.
        
        Args:
            image_paths: List of image paths.
            mask_paths: List of mask paths.
            frame_ids: List of frame IDs.
        
        Returns:
            (global_background_data, foreground_frames_data)
        """
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri
        
        # Step 1: Preprocess
        images_padded, padded_sizes, scaled_sizes, original_sizes, pad_coords = \
            load_and_preprocess_images_aspect_ratio(image_paths, target_long_edge=518)
        masks_padded, _, _, _, _ = \
            load_and_preprocess_masks_aspect_ratio(mask_paths, target_long_edge=518)
        
        H_padded, W_padded = padded_sizes[0].tolist()
        H_scaled, W_scaled = scaled_sizes[0].tolist()
        H_orig, W_orig = original_sizes[0].tolist()
        pad_top, pad_bottom, pad_left, pad_right = pad_coords[0].tolist()
        
        # Step 2: Inference
        images_inference = images_padded.to(self.device).unsqueeze(0)
        masks_inference = masks_padded.to(self.device).unsqueeze(0)
        
        with torch.no_grad():
            with torch.amp.autocast('cuda', dtype=self.dtype):
                aggregated_tokens_list, ps_idx = self.model.aggregator(images_inference)
                pose_enc = self.model.camera_head(aggregated_tokens_list)[-1]
                extrinsic_infer, intrinsic_infer = pose_encoding_to_extri_intri(
                    pose_enc, images_inference.shape[-2:]
                )
            
            with torch.amp.autocast('cuda', enabled=False):
                depth_map_infer, depth_conf_infer = self.model.depth_head(
                    aggregated_tokens_list, images_inference, ps_idx
                )
                point_map_infer, point_conf_infer = self.model.point_head(
                    aggregated_tokens_list, images_inference, ps_idx
                )
        
        # Step 3: Post-process to original size
        num_frames = point_map_infer.size(1)
        
        depth_map_infer_squeezed = depth_map_infer.squeeze(-1)
        depth_map_scaled = depth_map_infer_squeezed[:, :, pad_top:pad_top+H_scaled, pad_left:pad_left+W_scaled]
        depth_conf_scaled = depth_conf_infer[:, :, pad_top:pad_top+H_scaled, pad_left:pad_left+W_scaled]
        
        point_map_infer_reshaped = point_map_infer.squeeze(0).permute(0, 3, 1, 2)
        point_map_scaled = point_map_infer_reshaped[:, :, pad_top:pad_top+H_scaled, pad_left:pad_left+W_scaled]
        point_conf_scaled = point_conf_infer[:, :, pad_top:pad_top+H_scaled, pad_left:pad_left+W_scaled]
        
        depth_map_orig = F.interpolate(depth_map_scaled, size=(H_orig, W_orig), mode='bilinear', align_corners=False)
        depth_conf_orig = F.interpolate(depth_conf_scaled, size=(H_orig, W_orig), mode='bilinear', align_corners=False)
        point_map_orig = F.interpolate(point_map_scaled, size=(H_orig, W_orig), mode='bilinear', align_corners=False)
        point_conf_orig = F.interpolate(point_conf_scaled, size=(H_orig, W_orig), mode='bilinear', align_corners=False)
        
        point_map_orig = point_map_orig.permute(0, 2, 3, 1)
        
        # Step 4: Generate point cloud (in original-size space)
        if self.point_source == 'point_map':
            world_points_orig = point_map_orig  # (N, H_orig, W_orig, 3)
            conf_orig = point_conf_orig.squeeze(0)  # (N, H_orig, W_orig)
        elif self.point_source == 'backproject':
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
            raise ValueError(f"Unknown point_source: {self.point_source}")
        
        images_scaled = images_padded[:, :, pad_top:pad_top+H_scaled, pad_left:pad_left+W_scaled]
        masks_scaled = masks_padded[:, :, pad_top:pad_top+H_scaled, pad_left:pad_left+W_scaled]
        
        images_orig = F.interpolate(images_scaled, size=(H_orig, W_orig), mode='bilinear', align_corners=False)
        masks_orig = F.interpolate(masks_scaled, size=(H_orig, W_orig), mode='bilinear', align_corners=False)
        
        images_orig_cpu = images_orig.cpu().permute(0, 2, 3, 1)  # (N, H_orig, W_orig, 3)
        masks_orig_cpu = masks_orig.cpu()  # (N, 1, H_orig, W_orig)
        
        # Step 5: Separate foreground/background per frame
        background_points_all = []
        background_colors_all = []
        background_confs_all = []
        camera_data = {}
        foreground_frames = []
        
        for frame_idx in tqdm(range(num_frames), desc="Background", unit="frame"):
            frame_id = frame_ids[frame_idx]
            
            intrinsic_orig_frame = scale_intrinsics(
                intrinsic_infer[0][frame_idx].unsqueeze(0),
                old_size=(H_padded, W_padded),
                new_size=(H_orig, W_orig)
            )
            
            camera_data[frame_id] = {
                'extrinsic': extrinsic_infer[0][frame_idx].cpu().numpy().tolist(),
                'intrinsic': intrinsic_orig_frame.squeeze(0).cpu().numpy().tolist()
            }
            
            points_map = world_points_orig[frame_idx].cpu().numpy()  # (H_orig, W_orig, 3)
            colors_map = images_orig_cpu[frame_idx].numpy()  # (H_orig, W_orig, 3)
            conf_map = (conf_orig[frame_idx].cpu().squeeze(-1) if conf_orig[frame_idx].dim() > 2
                       else conf_orig[frame_idx].cpu()).numpy()  # (H_orig, W_orig)
            fg_mask_binary = (masks_orig_cpu[frame_idx].squeeze(0).numpy() > 0.5)  # (H_orig, W_orig)
            
            padded_mask = self._compute_padded_mask(fg_mask_binary)
            bg_mask_2d = ~padded_mask
            fg_mask_2d = fg_mask_binary
            
            fg_points = points_map[fg_mask_2d]
            fg_colors = colors_map[fg_mask_2d]
            bg_points = points_map[bg_mask_2d]
            bg_colors = colors_map[bg_mask_2d]
            bg_confs = conf_map[bg_mask_2d]
            
            background_points_all.append(bg_points)
            background_colors_all.append(bg_colors)
            background_confs_all.append(bg_confs)
            
            foreground_frames.append({
                'frame_id': frame_id,
                'foreground_pointmap': points_map,
                'foreground_mask': fg_mask_2d,
                'padded_mask': padded_mask,
                'color': colors_map,
                'confidence': conf_map,
                'foreground_points': fg_points,
                'foreground_colors': fg_colors,
                'intrinsic': intrinsic_orig_frame.squeeze(0).cpu().numpy(),
                'extrinsic': extrinsic_infer[0][frame_idx].cpu().numpy()
            })
        
        # Step 6: Return merged background data
        return {
            'background_points': background_points_all,
            'background_colors': background_colors_all,
            'background_confs': background_confs_all,
            'camera_data': camera_data,
            'image_size': (H_orig, W_orig)
        }, foreground_frames
    
    def generate_global_background(self, data_dir: str):
        """Generate global background point cloud."""
        frame_ids = find_frame_dirs(data_dir)
        
        if len(frame_ids) == 0:
            raise ValueError(f"No frames found: {data_dir}")
        
        image_paths = []
        mask_paths = []
        
        for frame_id in frame_ids:
            image_path = Path(data_dir) / frame_id / "images" / f"{frame_id}_original.png"
            mask_path = Path(data_dir) / frame_id / "masks" / f"{frame_id}_foreground_mask.png"
            
            if image_path.exists() and mask_path.exists():
                image_paths.append(str(image_path))
                mask_paths.append(str(mask_path))
        
        if self.model is None:
            self.load_model()
        
        bg_data, fg_frames = self.process_batch(image_paths, mask_paths, frame_ids)
        
        # Save foreground point clouds and mappings
        for fg_frame in fg_frames:
            frame_id = fg_frame['frame_id']
            pointcloud_dir = Path(data_dir) / frame_id / 'pointcloud'
            ensure_dir(pointcloud_dir)
            
            if len(fg_frame['foreground_points']) > 0:
                fg_pcd = o3d.geometry.PointCloud()
                fg_pcd.points = o3d.utility.Vector3dVector(fg_frame['foreground_points'])
                fg_pcd.colors = o3d.utility.Vector3dVector(fg_frame['foreground_colors'])
                o3d.io.write_point_cloud(str(pointcloud_dir / f"{frame_id}_foreground_1_view.ply"), fg_pcd)
            
            mapping_path = pointcloud_dir / f"{frame_id}_foreground_1_view.npz"
            np.savez(str(mapping_path),
                    foreground_pointmap=fg_frame['foreground_pointmap'],
                    foreground_mask=fg_frame['foreground_mask'],
                    padded_mask=fg_frame['padded_mask'],
                    color=fg_frame['color'],
                    confidence=fg_frame['confidence'],
                    pad_pixels=self.pad_pixels,
                    original_size=bg_data['image_size'],
                    intrinsic=fg_frame['intrinsic'],
                    extrinsic=fg_frame['extrinsic'])
        
        # Process global background point cloud
        global_bg_points = np.concatenate(bg_data['background_points'])
        global_bg_colors = np.concatenate(bg_data['background_colors'])
        global_bg_confs = np.concatenate(bg_data['background_confs'])
        
        if len(global_bg_points) > 0:
            # Confidence percentile filtering
            if self.confidence_threshold == 0.0:
                threshold_value = 0.0
            else:
                threshold_value = np.percentile(global_bg_confs, self.confidence_threshold)
            
            conf_mask = (global_bg_confs >= threshold_value) & (global_bg_confs > 1e-5)
            bg_points_conf = global_bg_points[conf_mask]
            bg_colors_conf = global_bg_colors[conf_mask]
            
            if len(bg_points_conf) == 0:
                print("Warning: no points remain after confidence filtering, skipping save.")
                return bg_data
            
            bg_pcd = o3d.geometry.PointCloud()
            bg_pcd.points = o3d.utility.Vector3dVector(bg_points_conf)
            bg_pcd.colors = o3d.utility.Vector3dVector(
                np.clip(bg_colors_conf, 0, 1) if bg_colors_conf.max() > 1.0 else bg_colors_conf
            )
            
            # Voxel downsampling
            if self.voxel_size:
                bg_pcd_downsampled = bg_pcd.voxel_down_sample(voxel_size=self.voxel_size)
            else:
                bg_pcd_downsampled = bg_pcd
            
            # Statistical outlier removal
            nb_neighbors = min(self.outlier_nb_neighbors, len(bg_pcd_downsampled.points))
            if nb_neighbors > 1:
                bg_pcd_final, _ = bg_pcd_downsampled.remove_statistical_outlier(
                    nb_neighbors=nb_neighbors, std_ratio=self.outlier_std_ratio
                )
            else:
                bg_pcd_final = bg_pcd_downsampled
            
            bg_ply_path = Path(data_dir) / "global_background.ply"
            o3d.io.write_point_cloud(str(bg_ply_path), bg_pcd_final)
            print(f"Background saved: {bg_ply_path} "
                  f"({len(global_bg_points):,} -> {len(bg_pcd_final.points):,} points)")
        else:
            print("Warning: background point cloud is empty.")
        
        # Save camera parameters
        camera_json_data = {
            "image_size": list(bg_data['image_size']),
            "coordinate_system": "y-down (original)"
        }
        camera_json_data.update(bg_data['camera_data'])
        
        camera_json_path = Path(data_dir) / "global_camera.json"
        with open(camera_json_path, "w") as f:
            json.dump(camera_json_data, f, indent=4)
        print(f"Camera params saved: {camera_json_path} ({len(bg_data['camera_data'])} frames)")
        
        return bg_data
    
    @classmethod
    def from_config(cls, config):
        """Create generator from config."""
        from utils.config import Config
        if not isinstance(config, Config):
            config = Config(config)
        
        return cls(
            device=config.get('common.device', 'cuda'),
            confidence_threshold=config.get('stage_1.background.confidence_threshold', 50.0),
            pad_pixels=config.get('stage_1.background.pad_pixels', 5),
            voxel_size=config.get('stage_1.background.voxel_size', 0.001),
            outlier_nb_neighbors=config.get('stage_1.background.outlier_nb_neighbors', 500),
            outlier_std_ratio=config.get('stage_1.background.outlier_std_ratio', 1.5),
            point_source=config.get('stage_1.background.point_source', 'backproject')
        )


if __name__ == '__main__':
    from utils.config import Config
    
    config = Config()
    
    fg_gen = ForegroundPointCloudGenerator.from_config(config)
    print(f"Foreground generator created: device={fg_gen.device}, long_edge={fg_gen.target_long_edge}")
    
    bg_gen = BackgroundPointCloudGenerator.from_config(config)
    print(f"Background generator created: device={bg_gen.device}, voxel={bg_gen.voxel_size}, source={bg_gen.point_source}")
