"""Point cloud alignment module with Kalman smoothing."""

import os
import cv2
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import trimesh
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm

from utils.file_io import find_frame_dirs, ensure_dir


class KalmanFilter1D:
    """1D Kalman filter with constant-velocity model. State: [position, velocity]."""
    
    def __init__(self, process_noise: float, measurement_noise: float):
        """Args:
            process_noise: Process noise covariance Q.
            measurement_noise: Measurement noise covariance R.
        """
        self.Q = np.array([[process_noise, 0], 
                          [0, process_noise]])
        self.R = measurement_noise
        self.x = np.zeros(2)  # [position, velocity]
        self.P = np.eye(2) * 1.0
    
    def predict(self, dt: float = 1.0):
        """Predict step: propagate state forward using motion model."""
        F = np.array([[1, dt], 
                     [0, 1]])
        
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self.Q
    
    def update(self, measurement: float) -> float:
        """Update step: correct prediction with measurement."""
        H = np.array([[1, 0]])
        
        y = measurement - H @ self.x
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T / S
        
        self.x = self.x + K.flatten() * y
        self.P = (np.eye(2) - np.outer(K, H)) @ self.P
        
        return self.x[0]


def kalman_smooth_1d(measurements: np.ndarray, 
                     process_noise: float, 
                     measurement_noise: float) -> np.ndarray:
    """Bidirectional Kalman smoothing for a 1D signal.
    
    Args:
        measurements: Raw measurement sequence (N,).
        process_noise: Process noise Q.
        measurement_noise: Measurement noise R.
    
    Returns:
        Smoothed sequence (N,).
    """
    n = len(measurements)
    
    # Forward pass
    kf_forward = KalmanFilter1D(process_noise, measurement_noise)
    kf_forward.x[0] = measurements[0]
    forward = []
    for m in measurements:
        kf_forward.predict()
        smoothed = kf_forward.update(m)
        forward.append(smoothed)
    
    # Backward pass
    kf_backward = KalmanFilter1D(process_noise, measurement_noise)
    kf_backward.x[0] = measurements[-1]
    backward = []
    for m in reversed(measurements):
        kf_backward.predict()
        smoothed = kf_backward.update(m)
        backward.append(smoothed)
    backward = list(reversed(backward))
    
    smooth = (np.array(forward) + np.array(backward)) / 2.0
    
    return smooth


def apply_transform_with_anisotropic_scale(points, scales, R, t):
    """Apply anisotropic scale transform: P' = R @ (S @ P) + t."""
    points_flat = points.reshape(-1, 3)
    transformed = (R @ (points_flat * scales).T).T + t
    return transformed.reshape(points.shape)


class PointCloudAligner:
    """Two-pass point cloud aligner: per-frame rigid alignment + Kalman temporal smoothing."""
    
    def __init__(self,
                 erosion_kernel_size: int = 11,
                 erosion_iterations: int = 1,
                 corr_conf_threshold: float = 50.0,
                 corr_outlier_nb_neighbors: int = 500,
                 corr_outlier_std_ratio: float = 0.5,
                 complete_conf_threshold: float = 30.0,
                 complete_outlier_nb_neighbors: int = 50,
                 complete_outlier_std_ratio: float = 1.5,
                 enable_smoothing: bool = True,
                 kalman_process_noise: float = 1e-7,
                 kalman_measurement_noise: float = 0.05,
                 kalman_z_penalty: float = 1.0,
                 min_points: int = 100,
                 save_debug: bool = False,
                 save_debug_plots: bool = False,
                 debug_show_all_views: bool = True):
        """Initialize aligner parameters.
        
        Args:
            erosion_kernel_size: Morphological erosion kernel size (odd, 5-21).
            erosion_iterations: Erosion iterations (1-3).
            kalman_z_penalty: Z-axis smoothing penalty (1.0=none, <0.5=strong).
            debug_show_all_views: Show 5-view fused cloud in debug (True) or view-0 only (False).
        """
        self.erosion_kernel_size = erosion_kernel_size
        self.erosion_iterations = erosion_iterations
        
        self.corr_conf_threshold = corr_conf_threshold
        self.corr_outlier_nb_neighbors = corr_outlier_nb_neighbors
        self.corr_outlier_std_ratio = corr_outlier_std_ratio
        
        self.complete_conf_threshold = complete_conf_threshold
        self.complete_outlier_nb_neighbors = complete_outlier_nb_neighbors
        self.complete_outlier_std_ratio = complete_outlier_std_ratio
        
        self.enable_smoothing = enable_smoothing
        self.kalman_process_noise = kalman_process_noise
        self.kalman_measurement_noise = kalman_measurement_noise
        self.kalman_z_penalty = kalman_z_penalty
        
        self.min_points = min_points
        self.save_debug = save_debug
        self.save_debug_plots = save_debug_plots
        self.debug_show_all_views = debug_show_all_views
    
    def _save_debug_clouds(self,
                          folder: str,
                          frame_id: str,
                          src_points_before_conf: np.ndarray,
                          tgt_points_before_conf: np.ndarray,
                          src_points_after_conf: np.ndarray,
                          tgt_points_after_conf: np.ndarray,
                          src_points: np.ndarray,
                          tgt_points: np.ndarray,
                          tgt_points_scaled: np.ndarray,
                          src_transformed: np.ndarray,
                          source_conf: Optional[np.ndarray] = None,
                          target_conf: Optional[np.ndarray] = None,
                          src_points_all_views: Optional[np.ndarray] = None,
                          src_colors_all_views: Optional[np.ndarray] = None,
                          src_transformed_all_views: Optional[np.ndarray] = None):
        """Save debug point clouds as PLY files."""
        output_dir = Path(folder) / frame_id / 'pointcloud'
        debug_dir = output_dir / 'debug'
        debug_dir.mkdir(exist_ok=True)
        
        SRC_COLOR = [100, 150, 255]       # blue
        SRC_COLOR_LIGHT = [100, 150, 255] # blue (light)
        TGT_COLOR = [255, 165, 0]         # orange
        TGT_COLOR_LIGHT = [255, 165, 0]   # orange (light)
        ALIGNED_COLOR = [100, 255, 100]   # green
        
        # Step 1: Before confidence filter
        combined_step1_before = np.vstack([src_points_before_conf, tgt_points_before_conf])
        colors_step1_before = np.vstack([
            np.tile(SRC_COLOR_LIGHT, (len(src_points_before_conf), 1)),
            np.tile(TGT_COLOR_LIGHT, (len(tgt_points_before_conf), 1))
        ])
        axis_points, axis_colors = create_coordinate_axes(combined_step1_before)
        combined_with_axes = np.vstack([combined_step1_before, axis_points])
        colors_with_axes = np.vstack([colors_step1_before, axis_colors])
        trimesh.PointCloud(combined_with_axes, colors=colors_with_axes).export(
            str(debug_dir / f"{frame_id}_step1_before_conf_filter.ply"))
        
        # Step 1: After confidence filter
        combined_step1_after = np.vstack([src_points_after_conf, tgt_points_after_conf])
        colors_step1_after = np.vstack([
            np.tile(SRC_COLOR, (len(src_points_after_conf), 1)),
            np.tile(TGT_COLOR, (len(tgt_points_after_conf), 1))
        ])
        axis_points, axis_colors = create_coordinate_axes(combined_step1_after)
        combined_with_axes = np.vstack([combined_step1_after, axis_points])
        colors_with_axes = np.vstack([colors_step1_after, axis_colors])
        trimesh.PointCloud(combined_with_axes, colors=colors_with_axes).export(
            str(debug_dir / f"{frame_id}_step1_after_conf_filter.ply"))
        
        # Step 2: Before outlier removal
        combined_step2_before = np.vstack([src_points_after_conf, tgt_points_after_conf])
        colors_step2_before = np.vstack([
            np.tile(SRC_COLOR, (len(src_points_after_conf), 1)),
            np.tile(TGT_COLOR, (len(tgt_points_after_conf), 1))
        ])
        axis_points, axis_colors = create_coordinate_axes(combined_step2_before)
        combined_with_axes = np.vstack([combined_step2_before, axis_points])
        colors_with_axes = np.vstack([colors_step2_before, axis_colors])
        trimesh.PointCloud(combined_with_axes, colors=colors_with_axes).export(
            str(debug_dir / f"{frame_id}_step2_before_outlier_removal.ply"))
        
        # Step 2: After outlier removal
        combined_step2_after = np.vstack([src_points, tgt_points])
        colors_step2_after = np.vstack([
            np.tile(SRC_COLOR, (len(src_points), 1)),
            np.tile(TGT_COLOR, (len(tgt_points), 1))
        ])
        axis_points, axis_colors = create_coordinate_axes(combined_step2_after)
        combined_with_axes = np.vstack([combined_step2_after, axis_points])
        colors_with_axes = np.vstack([colors_step2_after, axis_colors])
        trimesh.PointCloud(combined_with_axes, colors=colors_with_axes).export(
            str(debug_dir / f"{frame_id}_step2_after_outlier_removal.ply"))
        
        # Step 3: Before alignment (with bbox)
        combined_step3_before = np.vstack([src_points, tgt_points])
        colors_step3_before = np.vstack([
            np.tile(SRC_COLOR, (len(src_points), 1)),
            np.tile(TGT_COLOR, (len(tgt_points), 1))
        ])
        axis_points, axis_colors = create_coordinate_axes(combined_step3_before)
        src_bbox_points, src_bbox_colors = create_bbox(src_points, color=SRC_COLOR)
        tgt_bbox_points, tgt_bbox_colors = create_bbox(tgt_points, color=TGT_COLOR)
        
        combined_with_all = np.vstack([combined_step3_before, axis_points, src_bbox_points, tgt_bbox_points])
        colors_with_all = np.vstack([colors_step3_before, axis_colors, src_bbox_colors, tgt_bbox_colors])
        trimesh.PointCloud(combined_with_all, colors=colors_with_all).export(
            str(debug_dir / f"{frame_id}_step3_before_alignment.ply"))
        
        # Step 3-1: After target scaling
        combined_step3_1 = np.vstack([src_points, tgt_points_scaled])
        colors_step3_1 = np.vstack([
            np.tile(SRC_COLOR, (len(src_points), 1)),
            np.tile(TGT_COLOR, (len(tgt_points_scaled), 1))
        ])
        axis_points, axis_colors = create_coordinate_axes(combined_step3_1)
        src_bbox_points, src_bbox_colors = create_bbox(src_points, color=SRC_COLOR)
        tgt_bbox_points, tgt_bbox_colors = create_bbox(tgt_points_scaled, color=TGT_COLOR)
        
        combined_with_all = np.vstack([combined_step3_1, axis_points, src_bbox_points, tgt_bbox_points])
        colors_with_all = np.vstack([colors_step3_1, axis_colors, src_bbox_colors, tgt_bbox_colors])
        trimesh.PointCloud(combined_with_all, colors=colors_with_all).export(
            str(debug_dir / f"{frame_id}_step3_before_alignment_1.ply"))
        
        # Step 3: After alignment
        combined_step3_after = np.vstack([src_transformed, tgt_points_scaled])
        colors_step3_after = np.vstack([
            np.tile(ALIGNED_COLOR, (len(src_transformed), 1)),
            np.tile(TGT_COLOR, (len(tgt_points_scaled), 1))
        ])
        axis_points, axis_colors = create_coordinate_axes(combined_step3_after)
        combined_with_axes = np.vstack([combined_step3_after, axis_points])
        colors_with_axes = np.vstack([colors_step3_after, axis_colors])
        trimesh.PointCloud(combined_with_axes, colors=colors_with_axes).export(
            str(debug_dir / f"{frame_id}_step3_after_alignment.ply"))
        
        # Extra: Confidence visualization
        if source_conf is not None and target_conf is not None:
            from matplotlib.colors import Normalize
            import matplotlib
            
            conf_min = min(source_conf.min(), target_conf.min())
            conf_max = max(source_conf.max(), target_conf.max())
            
            try:
                cmap = matplotlib.colormaps['viridis']
            except (AttributeError, KeyError):
                from matplotlib.cm import get_cmap
                cmap = get_cmap('viridis')
            norm = Normalize(vmin=conf_min, vmax=conf_max)
            
            # Source confidence
            src_conf_normalized = norm(source_conf)
            src_conf_colors = (cmap(src_conf_normalized)[:, :3] * 255).astype(np.uint8)
            axis_points, axis_colors = create_coordinate_axes(src_points)
            src_with_axes = np.vstack([src_points, axis_points])
            src_colors_with_axes = np.vstack([src_conf_colors, axis_colors])
            trimesh.PointCloud(src_with_axes, colors=src_colors_with_axes).export(
                str(debug_dir / f"{frame_id}_extra_source_conf_viridis.ply"))
            
            # Target confidence
            tgt_conf_normalized = norm(target_conf)
            tgt_conf_colors = (cmap(tgt_conf_normalized)[:, :3] * 255).astype(np.uint8)
            axis_points, axis_colors = create_coordinate_axes(tgt_points)
            tgt_with_axes = np.vstack([tgt_points, axis_points])
            tgt_colors_with_axes = np.vstack([tgt_conf_colors, axis_colors])
            trimesh.PointCloud(tgt_with_axes, colors=tgt_colors_with_axes).export(
                str(debug_dir / f"{frame_id}_extra_target_conf_viridis.ply"))
            
            # Combined confidence visualization
            try:
                cmap_source = matplotlib.colormaps['Reds']
                cmap_target = matplotlib.colormaps['Blues']
            except (AttributeError, KeyError):
                from matplotlib.cm import get_cmap
                cmap_source = get_cmap('Reds')
                cmap_target = get_cmap('Blues')
            
            src_conf_colors_red = (cmap_source(src_conf_normalized)[:, :3] * 255).astype(np.uint8)
            tgt_conf_colors_blue = (cmap_target(tgt_conf_normalized)[:, :3] * 255).astype(np.uint8)
            
            combined_conf = np.vstack([src_points, tgt_points])
            combined_colors = np.vstack([src_conf_colors_red, tgt_conf_colors_blue])
            axis_points, axis_colors = create_coordinate_axes(combined_conf)
            combined_with_axes = np.vstack([combined_conf, axis_points])
            colors_with_axes = np.vstack([combined_colors, axis_colors])
            trimesh.PointCloud(combined_with_axes, colors=colors_with_axes).export(
                str(debug_dir / f"{frame_id}_extra_combined_conf_red_blue.ply"))
        
        # 5-view fused point cloud visualization (if provided)
        if src_points_all_views is not None and self.debug_show_all_views:
            if src_colors_all_views is not None:
                colors_5views = src_colors_all_views
            else:
                colors_5views = np.tile(SRC_COLOR, (len(src_points_all_views), 1))
            
            axis_points, axis_colors = create_coordinate_axes(src_points_all_views)
            combined_with_axes = np.vstack([src_points_all_views, axis_points])
            colors_with_axes = np.vstack([colors_5views, axis_colors])
            trimesh.PointCloud(combined_with_axes, colors=colors_with_axes).export(
                str(debug_dir / f"{frame_id}_extra_source_5views_original.ply"))
            
            # 5-view + target (before alignment comparison)
            combined_5views_tgt = np.vstack([src_points_all_views, tgt_points])
            colors_combined = np.vstack([
                colors_5views,
                np.tile(TGT_COLOR, (len(tgt_points), 1))
            ])
            axis_points, axis_colors = create_coordinate_axes(combined_5views_tgt)
            src_bbox_points, src_bbox_colors = create_bbox(src_points_all_views, color=SRC_COLOR)
            tgt_bbox_points, tgt_bbox_colors = create_bbox(tgt_points, color=TGT_COLOR)
            combined_with_all = np.vstack([combined_5views_tgt, axis_points, src_bbox_points, tgt_bbox_points])
            colors_with_all = np.vstack([colors_combined, axis_colors, src_bbox_colors, tgt_bbox_colors])
            trimesh.PointCloud(combined_with_all, colors=colors_with_all).export(
                str(debug_dir / f"{frame_id}_extra_5views_vs_target_before.ply"))
            
            # 5-view transformed (if provided)
            if src_transformed_all_views is not None:
                combined_aligned = np.vstack([src_transformed_all_views, tgt_points_scaled])
                colors_aligned = np.vstack([
                    np.tile(ALIGNED_COLOR, (len(src_transformed_all_views), 1)),
                    np.tile(TGT_COLOR, (len(tgt_points_scaled), 1))
                ])
                axis_points, axis_colors = create_coordinate_axes(combined_aligned)
                combined_with_axes = np.vstack([combined_aligned, axis_points])
                colors_with_axes = np.vstack([colors_aligned, axis_colors])
                trimesh.PointCloud(combined_with_axes, colors=colors_with_axes).export(
                    str(debug_dir / f"{frame_id}_extra_5views_vs_target_after.ply"))
    
    def _extract_correspondence_points(self, folder: str, frame_id: str) -> Tuple[Optional[Tuple], Optional[str]]:
        """Extract correspondence point pairs.
        
        Returns:
            Success: ((src_points, tgt_points, ...), None)
            Failure: (None, error_message)
        """
        v4_0_path = Path(folder) / frame_id / "pointcloud" / f"{frame_id}_foreground_5_views.npz"
        if not v4_0_path.exists():
            return None, "v4_0 mapping not found"
        
        v4_0_data = np.load(v4_0_path)
        v4_0_pointmaps = v4_0_data['foreground_pointmaps']
        v4_0_masks = v4_0_data['foreground_masks']
        v4_0_confs = v4_0_data['confidences']
        
        source_pointmap, source_mask = v4_0_pointmaps[0], v4_0_masks[0]
        
        single_view_path = Path(folder) / frame_id / "pointcloud" / f"{frame_id}_foreground_1_view.npz"
        if not single_view_path.exists():
            return None, "Single-view mapping not found"
        
        v4_1_data = np.load(single_view_path)
        target_pointmap = v4_1_data['foreground_pointmap']
        target_mask = v4_1_data['foreground_mask']
        
        if not np.array_equal(source_mask, target_mask):
            return None, "source_mask and target_mask mismatch"
        
        # Morphological erosion on common mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.erosion_kernel_size, self.erosion_kernel_size))
        common_mask = cv2.erode(source_mask.astype(np.uint8) * 255, kernel, iterations=self.erosion_iterations) > 0
        num_common = common_mask.sum()
        
        if num_common < self.min_points:
            return None, f"Insufficient common pixels ({num_common})"
        
        source_conf_full = v4_0_confs[0]
        if 'confidence' not in v4_1_data:
            return None, "Target missing confidence"
        target_conf_full = v4_1_data['confidence']
        
        # Confidence percentile filtering
        source_threshold = np.percentile(source_conf_full[common_mask], self.corr_conf_threshold) if self.corr_conf_threshold > 0 else 0.0
        target_threshold = np.percentile(target_conf_full[common_mask], self.corr_conf_threshold) if self.corr_conf_threshold > 0 else 0.0
        conf_mask = (source_conf_full >= source_threshold) & (target_conf_full >= target_threshold) & \
                    (source_conf_full > 1e-5) & (target_conf_full > 1e-5)
        final_mask = common_mask & conf_mask
        num_final = final_mask.sum()
        
        if num_final < self.min_points:
            return None, f"Insufficient points after confidence filter ({num_final})"
        
        src_points_before_conf = source_pointmap[common_mask]
        tgt_points_before_conf = target_pointmap[common_mask]
        
        src_points_after_conf = source_pointmap[final_mask]
        tgt_points_after_conf = target_pointmap[final_mask]
        source_conf = source_conf_full[final_mask]
        target_conf = target_conf_full[final_mask]
        
        # Outlier removal
        tgt_pcd = o3d.geometry.PointCloud()
        tgt_pcd.points = o3d.utility.Vector3dVector(tgt_points_after_conf)
        
        nb_neighbors = min(self.corr_outlier_nb_neighbors, len(tgt_points_after_conf))
        tgt_cleaned, inlier_indices = tgt_pcd.remove_statistical_outlier(
            nb_neighbors=nb_neighbors, std_ratio=self.corr_outlier_std_ratio
        )
        inlier_indices = np.array(inlier_indices)
        
        src_points = src_points_after_conf[inlier_indices]
        tgt_points = tgt_points_after_conf[inlier_indices]
        source_conf = source_conf[inlier_indices]
        target_conf = target_conf[inlier_indices]
        
        debug_data = {
            'src_points_before_conf': src_points_before_conf,
            'tgt_points_before_conf': tgt_points_before_conf,
            'src_points_after_conf': src_points_after_conf,
            'tgt_points_after_conf': tgt_points_after_conf,
        } if self.save_debug else None
        
        return (src_points, tgt_points, source_conf, target_conf, debug_data), None
    
    def align_single_frame(self,
                          folder: str,
                          frame_id: str) -> Optional[Dict]:
        """Align a single frame's point cloud.
        
        Returns:
            Alignment result dict or None on failure.
        """
        try:
            result, error = self._extract_correspondence_points(folder, frame_id)
            if result is None:
                return None
            
            src_points, tgt_points, source_conf, target_conf, debug_data = result
            
            v4_0_path = Path(folder) / frame_id / "pointcloud" / f"{frame_id}_foreground_5_views.npz"
            v4_0_data = np.load(v4_0_path)
            v4_0_pointmaps = v4_0_data['foreground_pointmaps']
            v4_0_masks = v4_0_data['foreground_masks']
            v4_0_confs = v4_0_data['confidences']
            v4_0_colors = v4_0_data['colors']
            
            # Compute source aspect ratios
            src_bbox_min_calc = np.min(src_points, axis=0)
            src_bbox_max_calc = np.max(src_points, axis=0)
            src_size_calc = src_bbox_max_calc - src_bbox_min_calc
            src_width_height_ratio = src_size_calc[2] / src_size_calc[1] if src_size_calc[1] > 0 else 1.0
            src_length_height_ratio = src_size_calc[0] / src_size_calc[1] if src_size_calc[1] > 0 else 1.0
            
            # Scale target width to match source width/height ratio
            tgt_bbox_min = np.min(tgt_points, axis=0)
            tgt_bbox_max = np.max(tgt_points, axis=0)
            tgt_size = tgt_bbox_max - tgt_bbox_min
            
            tgt_width_new = tgt_size[1] * src_width_height_ratio if tgt_size[1] > 0 else tgt_size[2]
            width_scale = tgt_width_new / tgt_size[2] if tgt_size[2] > 0 else 1.0
            
            z_fixed = tgt_bbox_min[2]
            tgt_points_scaled = tgt_points.copy()
            z_offset = tgt_points_scaled[:, 2] - z_fixed
            tgt_points_scaled[:, 2] = z_fixed + z_offset * width_scale
            
            tgt_bbox_min_calc = np.min(tgt_points_scaled, axis=0)
            tgt_bbox_max_calc = np.max(tgt_points_scaled, axis=0)
            tgt_size_calc = tgt_bbox_max_calc - tgt_bbox_min_calc
            tgt_length_height_ratio = tgt_size_calc[0] / tgt_size_calc[1] if tgt_size_calc[1] > 0 else 1.0
            
            # X-axis scaling
            x_scale = tgt_length_height_ratio / src_length_height_ratio if src_length_height_ratio > 0 else 1.0
            
            src_center_x = (src_bbox_min_calc[0] + src_bbox_max_calc[0]) / 2.0
            src_points_prescaled = src_points.copy()
            src_points_prescaled[:, 0] = src_center_x + (src_points[:, 0] - src_center_x) * x_scale
            
            # Isotropic scale
            src_size_prescaled = np.max(src_points_prescaled, axis=0) - np.min(src_points_prescaled, axis=0)
            scale = tgt_size_calc[1] / src_size_prescaled[1] if src_size_prescaled[1] > 0 else 1.0
            
            src_points_scaled = src_points_prescaled * scale
            
            # Translation
            src_bbox_scaled_min = np.min(src_points_scaled, axis=0)
            src_bbox_scaled_max = np.max(src_points_scaled, axis=0)
            
            t_y = tgt_bbox_max_calc[1] - src_bbox_scaled_max[1]
            src_center_xz = (src_bbox_scaled_min[[0, 2]] + src_bbox_scaled_max[[0, 2]]) / 2.0
            tgt_center_xz = (tgt_bbox_min_calc[[0, 2]] + tgt_bbox_max_calc[[0, 2]]) / 2.0
            t_x = tgt_center_xz[0] - src_center_xz[0]
            t_z = tgt_center_xz[1] - src_center_xz[1]
            
            T = np.array([t_x, t_y, t_z])
            
            # Reconstruct complete point cloud from all views
            all_points, all_colors, all_confs, all_view_indices = [], [], [], []
            
            for view_idx in range(5):
                view_mask = v4_0_masks[view_idx]
                if view_mask.sum() == 0:
                    continue
                all_points.append(v4_0_pointmaps[view_idx][view_mask])
                all_colors.append(v4_0_colors[view_idx][view_mask])
                all_confs.append(v4_0_confs[view_idx][view_mask])
                all_view_indices.append(np.full(view_mask.sum(), view_idx, dtype=np.int32))
            
            if not all_points:
                return None
            
            complete_points = np.concatenate(all_points)
            complete_colors = np.concatenate(all_colors)
            complete_confs = np.concatenate(all_confs)
            complete_view_indices = np.concatenate(all_view_indices)
            
            complete_points_original = complete_points.copy()
            complete_colors_original = complete_colors.copy()
            
            # Apply transform
            scales = np.array([scale, scale, scale])
            R = np.eye(3)
            
            # X-axis pre-scaling
            center_x = np.mean(complete_points[:, 0])
            complete_points_prescaled = complete_points.copy()
            complete_points_prescaled[:, 0] = center_x + (complete_points[:, 0] - center_x) * x_scale
            
            aligned_points = apply_transform_with_anisotropic_scale(complete_points_prescaled, scales, R, T)
            
            # Confidence percentile filtering
            if self.complete_conf_threshold == 0.0:
                threshold_value = 0.0
            else:
                threshold_value = np.percentile(complete_confs, self.complete_conf_threshold)
            
            conf_mask = (complete_confs >= threshold_value) & (complete_confs > 1e-5)
            aligned_points = aligned_points[conf_mask]
            complete_colors = complete_colors[conf_mask]
            complete_view_indices = complete_view_indices[conf_mask]
            
            if len(aligned_points) == 0:
                return None
            
            # Outlier removal
            aligned_pcd = o3d.geometry.PointCloud()
            aligned_pcd.points = o3d.utility.Vector3dVector(aligned_points)
            aligned_pcd.colors = o3d.utility.Vector3dVector(
                complete_colors / 255.0 if complete_colors.max() > 1 else complete_colors
            )
            
            nb_neighbors = min(self.complete_outlier_nb_neighbors, len(aligned_points))
            if nb_neighbors > 1:
                aligned_pcd_cleaned, inlier_indices = aligned_pcd.remove_statistical_outlier(
                    nb_neighbors=nb_neighbors, std_ratio=self.complete_outlier_std_ratio
                )
                
                inlier_indices = np.array(inlier_indices)
                aligned_points = np.asarray(aligned_pcd_cleaned.points)
                complete_colors = (np.asarray(aligned_pcd_cleaned.colors) * 255).astype(np.uint8)
                complete_view_indices = complete_view_indices[inlier_indices]
            
            # Debug save
            if self.save_debug and debug_data is not None:
                src_center_x = np.mean(src_points[:, 0])
                src_points_prescaled = src_points.copy()
                src_points_prescaled[:, 0] = src_center_x + (src_points[:, 0] - src_center_x) * x_scale
                src_transformed = src_points_prescaled * scale + T
                
                src_5views_transformed = None
                if self.debug_show_all_views:
                    center_x_5views = np.mean(complete_points_original[:, 0])
                    src_5views_prescaled = complete_points_original.copy()
                    src_5views_prescaled[:, 0] = center_x_5views + (complete_points_original[:, 0] - center_x_5views) * x_scale
                    src_5views_transformed = src_5views_prescaled * scale + T
                
                self._save_debug_clouds(
                    folder=folder,
                    frame_id=frame_id,
                    src_points_before_conf=debug_data['src_points_before_conf'],
                    tgt_points_before_conf=debug_data['tgt_points_before_conf'],
                    src_points_after_conf=debug_data['src_points_after_conf'],
                    tgt_points_after_conf=debug_data['tgt_points_after_conf'],
                    src_points=src_points,
                    tgt_points=tgt_points,
                    tgt_points_scaled=tgt_points_scaled,
                    src_transformed=src_transformed,
                    source_conf=source_conf,
                    target_conf=target_conf,
                    src_points_all_views=complete_points_original if self.debug_show_all_views else None,
                    src_colors_all_views=complete_colors_original if self.debug_show_all_views else None,
                    src_transformed_all_views=src_5views_transformed
                )
            
            # Save results
            output_dir = Path(folder) / frame_id / 'pointcloud'
            ensure_dir(output_dir)
            
            colors_uint8 = (complete_colors * 255).astype(np.uint8) if complete_colors.max() <= 1.0 else complete_colors.astype(np.uint8)
            
            aligned_ply = output_dir / f"{frame_id}_foreground_5_views_aligned.ply"
            trimesh.PointCloud(aligned_points, colors=colors_uint8).export(str(aligned_ply))
            
            aligned_npz = output_dir / f"{frame_id}_foreground_5_views_aligned.npz"
            np.savez(str(aligned_npz),
                    points=aligned_points,
                    colors=colors_uint8,
                    view_indices=complete_view_indices,
                    view_names=np.array(['foreground', '0', '1', '2', '3']),
                    x_scale=x_scale,
                    scale=scale,
                    translation=T)
            
            return {
                'frame_id': frame_id,
                'x_scale': x_scale,
                'scale': scale,
                'translation': T,
                'num_points': len(aligned_points)
            }
        
        except Exception as e:
            print(f"\nFrame {frame_id} alignment failed: {e}")
            return None
    
    def smooth_trajectory(self, folder: str, frame_ids: List[str]):
        """Pass 2: Kalman-smooth the alignment trajectory."""
        print("\n" + "="*80)
        print("Pass 2: Temporal Smoothing (Bidirectional Kalman Filter)")
        print("="*80)
        
        # Load all aligned point clouds
        aligned_data = []
        for frame_id in tqdm(frame_ids, desc="Loading", unit="frame"):
            npz_path = Path(folder) / frame_id / 'pointcloud' / f"{frame_id}_foreground_5_views_aligned.npz"
            if not npz_path.exists():
                continue
            
            data = np.load(npz_path)
            aligned_data.append({
                'frame_id': frame_id,
                'points': data['points'],
                'colors': data['colors'],
                'view_indices': data['view_indices'],
                'view_names': data['view_names'],
                'x_scale': float(data['x_scale']),
                'scale': float(data['scale']),
                'translation': data['translation']
            })
        
        if len(aligned_data) == 0:
            print("ERROR: No aligned point clouds found")
            return
        
        print(f"Loaded {len(aligned_data)} frames")
        
        # Smooth x_scale sequence
        x_scales_raw = np.array([d['x_scale'] for d in aligned_data])
        x_scales_smooth = kalman_smooth_1d(
            x_scales_raw,
            process_noise=self.kalman_process_noise,
            measurement_noise=self.kalman_measurement_noise
        )
        
        # Compute bbox centers and offsets
        bbox_centers = np.array([np.mean(d['points'], axis=0) for d in aligned_data])
        offsets = bbox_centers - bbox_centers[0]
        
        # Smooth bbox offsets
        offsets_smooth = np.zeros_like(offsets)
        for dim in range(3):
            if dim == 2:  # Z-axis penalty
                process_noise = self.kalman_process_noise * self.kalman_z_penalty
            else:
                process_noise = self.kalman_process_noise
            
            offsets_smooth[:, dim] = kalman_smooth_1d(
                offsets[:, dim],
                process_noise=process_noise,
                measurement_noise=self.kalman_measurement_noise
            )
        
        # Re-apply smoothed transforms and save
        for i, data in enumerate(tqdm(aligned_data, desc="Smoothing", unit="frame")):
            frame_id = data['frame_id']
            
            v4_0_path = Path(folder) / frame_id / "pointcloud" / f"{frame_id}_foreground_5_views.npz"
            if not v4_0_path.exists():
                continue
            
            v4_0_data = np.load(v4_0_path)
            v4_0_pointmaps = v4_0_data['foreground_pointmaps']
            v4_0_masks = v4_0_data['foreground_masks']
            v4_0_confs = v4_0_data['confidences']
            v4_0_colors = v4_0_data['colors']
            
            # Reconstruct complete point cloud
            all_points, all_colors, all_confs, all_view_indices = [], [], [], []
            for view_idx in range(5):
                view_mask = v4_0_masks[view_idx]
                if view_mask.sum() == 0:
                    continue
                all_points.append(v4_0_pointmaps[view_idx][view_mask])
                all_colors.append(v4_0_colors[view_idx][view_mask])
                all_confs.append(v4_0_confs[view_idx][view_mask])
                all_view_indices.append(np.full(view_mask.sum(), view_idx, dtype=np.int32))
            
            if not all_points:
                continue
            
            complete_points = np.concatenate(all_points)
            complete_colors = np.concatenate(all_colors)
            complete_confs = np.concatenate(all_confs)
            complete_view_indices = np.concatenate(all_view_indices)
            
            # Apply smoothed transform
            x_scale_smooth = x_scales_smooth[i]
            scale_original = data['scale']
            T_original = data['translation']
            
            scales = np.array([scale_original, scale_original, scale_original])
            R = np.eye(3)
            
            # x_scale pre-scaling
            center_x = np.mean(complete_points[:, 0])
            complete_points_prescaled = complete_points.copy()
            complete_points_prescaled[:, 0] = center_x + (complete_points[:, 0] - center_x) * x_scale_smooth
            
            aligned_points = apply_transform_with_anisotropic_scale(complete_points_prescaled, scales, R, T_original)
            
            # Confidence percentile filtering
            if self.complete_conf_threshold == 0.0:
                threshold_value = 0.0
            else:
                threshold_value = np.percentile(complete_confs, self.complete_conf_threshold)
            
            conf_mask = (complete_confs >= threshold_value) & (complete_confs > 1e-5)
            aligned_points = aligned_points[conf_mask]
            complete_colors = complete_colors[conf_mask]
            complete_view_indices = complete_view_indices[conf_mask]
            
            if len(aligned_points) == 0:
                continue
            
            # Outlier removal
            aligned_pcd = o3d.geometry.PointCloud()
            aligned_pcd.points = o3d.utility.Vector3dVector(aligned_points)
            aligned_pcd.colors = o3d.utility.Vector3dVector(
                complete_colors / 255.0 if complete_colors.max() > 1 else complete_colors
            )
            
            nb_neighbors = min(self.complete_outlier_nb_neighbors, len(aligned_points))
            if nb_neighbors > 1:
                aligned_pcd_cleaned, inlier_indices = aligned_pcd.remove_statistical_outlier(
                    nb_neighbors=nb_neighbors, std_ratio=self.complete_outlier_std_ratio
                )
                
                inlier_indices = np.array(inlier_indices)
                aligned_points = np.asarray(aligned_pcd_cleaned.points)
                complete_colors = (np.asarray(aligned_pcd_cleaned.colors) * 255).astype(np.uint8)
                complete_view_indices = complete_view_indices[inlier_indices]
            
            # Apply bbox offset adjustment
            bbox_adjustment = offsets_smooth[i] - offsets[i]
            points_adjusted = aligned_points + bbox_adjustment
            
            colors_uint8 = (complete_colors * 255).astype(np.uint8) if complete_colors.max() <= 1.0 else complete_colors.astype(np.uint8)
            
            # Save
            output_dir = Path(folder) / frame_id / 'pointcloud'
            npz_path = output_dir / f"{frame_id}_foreground_5_views_aligned_smooth.npz"
            ply_path = output_dir / f"{frame_id}_foreground_5_views_aligned_smooth.ply"
            
            np.savez(str(npz_path),
                    points=points_adjusted,
                    colors=colors_uint8,
                    view_indices=complete_view_indices,
                    view_names=np.array(['foreground', '0', '1', '2', '3']),
                    x_scale=x_scale_smooth,
                    scale=scale_original,
                    translation=T_original + bbox_adjustment)
            
            trimesh.PointCloud(points_adjusted, colors=colors_uint8).export(str(ply_path))
        
        if self.save_debug_plots:
            self._plot_smoothing_comparison(
                offsets,
                offsets_smooth,
                x_scales_raw,
                x_scales_smooth,
                [d['frame_id'] for d in aligned_data],
                folder
            )
        
        print("\n" + "="*80)
        print("Pass 2 complete.")
        print(f"  Output: *_foreground_5_views_aligned_smooth.{{ply,npz}}")
        print("="*80)
    
    def _plot_smoothing_comparison(self,
                                   offsets_raw: np.ndarray,
                                   offsets_smooth: np.ndarray,
                                   x_scales_raw: np.ndarray,
                                   x_scales_smooth: np.ndarray,
                                   frame_ids: List[str],
                                   output_folder: str):
        """Visualize Kalman smoothing: bbox trajectory and x_scale (2x3 subplot)."""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Temporal Smoothing: Bbox Trajectory & X-Scale (Kalman Filter)', 
                     fontsize=16, fontweight='bold')
        
        frames = np.arange(len(frame_ids))
        
        # Row 1: X/Y/Z offsets
        for idx, (axis_name, axis_idx) in enumerate([('X', 0), ('Y', 1), ('Z', 2)]):
            ax = axes[0, idx]
            ax.plot(frames, offsets_raw[:, axis_idx], 'r.-', alpha=0.6, label='Raw', markersize=4)
            ax.plot(frames, offsets_smooth[:, axis_idx], 'b-', linewidth=2, label='Smoothed', alpha=0.8)
            ax.set_xlabel('Frame Index')
            ax.set_ylabel(f'{axis_name} Offset')
            ax.set_title(f'{axis_name}-axis Offset (relative to frame 0)')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Row 2 left: XZ plane trajectory
        ax = axes[1, 0]
        ax.plot(offsets_raw[:, 0], offsets_raw[:, 2], 'r.-', alpha=0.6, label='Raw', markersize=4)
        ax.plot(offsets_smooth[:, 0], offsets_smooth[:, 2], 'b-', linewidth=2, label='Smoothed', alpha=0.8)
        ax.set_xlabel('X Offset')
        ax.set_ylabel('Z Offset')
        ax.set_title('XZ Plane Trajectory (Top View)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
        
        # Row 2 center: x_scale
        ax = axes[1, 1]
        ax.plot(frames, x_scales_raw, 'r.-', alpha=0.6, label='Raw', markersize=4)
        ax.plot(frames, x_scales_smooth, 'b-', linewidth=2, label='Smoothed', alpha=0.8)
        ax.set_xlabel('Frame Index')
        ax.set_ylabel('X-Scale Factor')
        ax.set_title('X-Scale Evolution (Length/Height Compensation)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Row 2 right: info
        ax = axes[1, 2]
        ax.axis('off')
        
        info_text = "Smoothing Strategy\n" + "="*35 + "\n\n"
        info_text += "Smoothed Parameters:\n"
        info_text += "  - x_scale (X-axis pre-scaling)\n"
        info_text += "  - bbox offset (position)\n\n"
        info_text += "Preserved Parameters:\n"
        info_text += "  - scale (per-frame size)\n"
        info_text += "    Reason: Target space is more\n"
        info_text += "    accurate, scale reflects true\n"
        info_text += "    source reconstruction quality\n\n"
        info_text += f"Total Frames: {len(frame_ids)}\n\n"
        info_text += "Color Code:\n"
        info_text += "  Red = Raw (original alignment)\n"
        info_text += "  Blue = Smoothed (Kalman filtered)"
        
        ax.text(0.05, 0.5, info_text, fontsize=10, family='monospace',
                verticalalignment='center', transform=ax.transAxes)
        
        plt.tight_layout()
        
        output_path = Path(output_folder) / 'bbox_trajectory_comparison.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {output_path}")
        plt.close()
    
    def align_all_frames(self,
                        folder: str,
                        num_frames: Optional[int] = None):
        """Align all frames (two-pass processing).
        
        Args:
            folder: Data directory.
            num_frames: Number of frames to process (None for all).
        """
        frame_ids = find_frame_dirs(folder, max_frames=num_frames)
        
        if len(frame_ids) == 0:
            raise ValueError(f"No frames found in: {folder}")
        
        print("="*80)
        print("Step 1.2: Align 5-view point clouds to background space (with Kalman smoothing)")
        print("="*80)
        print(f"  Directory: {folder}")
        print(f"  Method: Bbox-based rigid transformation")
        print(f"  Smoothing: {'Enabled' if self.enable_smoothing else 'Disabled'}")
        print(f"  Frames: {len(frame_ids)}")
        print("="*80)
        
        # Pass 1: Per-frame alignment
        print("\n" + "="*80)
        print("Pass 1: Rigid Alignment (bbox-based)")
        print("="*80)
        
        success_count = 0
        failed_frames = []
        
        for frame_id in tqdm(frame_ids, desc="Aligning", unit="frame"):
            result = self.align_single_frame(folder, frame_id)
            if result is not None:
                success_count += 1
            else:
                failed_frames.append(frame_id)
        
        print("\n" + "="*80)
        print(f"Pass 1 complete: {success_count}/{len(frame_ids)} frames succeeded")
        if failed_frames:
            print(f"  Failed: {len(failed_frames)} frames")
            for fid in failed_frames[:3]:
                print(f"    - {fid}")
            if len(failed_frames) > 3:
                print(f"    ... and {len(failed_frames) - 3} more")
        print("="*80)
        
        # Pass 2: Temporal Smoothing
        if self.enable_smoothing and success_count > 0:
            self.smooth_trajectory(folder, frame_ids)
        elif not self.enable_smoothing:
            print("\nKalman smoothing disabled (--no_smoothing)")
        
        print(f"\n{'='*80}")
        print(f"All processing complete.")
        print(f"{'='*80}\n")
    
    @classmethod
    def from_config(cls, config):
        """Create aligner from config."""
        from utils.config import Config
        if not isinstance(config, Config):
            config = Config(config)
        
        return cls(
            erosion_kernel_size=config.get('stage_1.alignment.erosion_kernel_size', 11),
            erosion_iterations=config.get('stage_1.alignment.erosion_iterations', 1),
            corr_conf_threshold=config.get('stage_1.alignment.corr_conf_threshold', 50.0),
            corr_outlier_nb_neighbors=config.get('stage_1.alignment.corr_outlier_nb_neighbors', 500),
            corr_outlier_std_ratio=config.get('stage_1.alignment.corr_outlier_std_ratio', 0.5),
            complete_conf_threshold=config.get('stage_1.alignment.complete_conf_threshold', 30.0),
            complete_outlier_nb_neighbors=config.get('stage_1.alignment.complete_outlier_nb_neighbors', 50),
            complete_outlier_std_ratio=config.get('stage_1.alignment.complete_outlier_std_ratio', 1.5),
            enable_smoothing=config.get('stage_1.alignment.enable_smoothing', True),
            kalman_process_noise=config.get('stage_1.alignment.kalman_process_noise', 1e-7),
            kalman_measurement_noise=config.get('stage_1.alignment.kalman_measurement_noise', 0.05),
            kalman_z_penalty=config.get('stage_1.alignment.kalman_z_penalty', 1.0),
            min_points=config.get('stage_1.alignment.min_points', 100),
            save_debug=config.get('stage_1.alignment.save_debug', False),
            save_debug_plots=config.get('stage_1.alignment.save_debug_plots', False),
            debug_show_all_views=config.get('stage_1.alignment.debug_show_all_views', True)
        )


def create_coordinate_axes(points, axis_scale=None, num_points_per_axis=50):
    """Create coordinate axes visualization as point cloud.
    
    Returns:
        axis_points (M, 3), axis_colors (M, 3) in RGB [0-255].
    """
    if axis_scale is None:
        if len(points) > 0:
            bbox_size = np.max(points, axis=0) - np.min(points, axis=0)
            axis_scale = np.max(bbox_size) * 0.1
        else:
            axis_scale = 0.5
    
    origin = np.array([0.0, 0.0, 0.0])
    axis_points = []
    axis_colors = []
    
    # X-axis: red
    for t in np.linspace(0, 1, num_points_per_axis):
        point = origin + t * np.array([axis_scale, 0, 0])
        axis_points.append(point)
        axis_colors.append([255, 0, 0])
    
    # Y-axis: green
    for t in np.linspace(0, 1, num_points_per_axis):
        point = origin + t * np.array([0, axis_scale, 0])
        axis_points.append(point)
        axis_colors.append([0, 255, 0])
    
    # Z-axis: blue
    for t in np.linspace(0, 1, num_points_per_axis):
        point = origin + t * np.array([0, 0, axis_scale])
        axis_points.append(point)
        axis_colors.append([0, 0, 255])
    
    # Origin: white
    axis_points.append(origin)
    axis_colors.append([255, 255, 255])
    
    return np.array(axis_points), np.array(axis_colors)


def create_bbox(points, color=[255, 255, 0], num_points_per_edge=50):
    """Create AABB bounding box visualization as point cloud.
    
    Returns:
        bbox_points (M, 3), bbox_colors (M, 3).
    """
    if len(points) == 0:
        return np.array([]).reshape(0, 3), np.array([]).reshape(0, 3)
    
    bbox_min = np.min(points, axis=0)
    bbox_max = np.max(points, axis=0)
    
    x_min, y_min, z_min = bbox_min
    x_max, y_max, z_max = bbox_max
    
    vertices = np.array([
        [x_min, y_min, z_min], [x_max, y_min, z_min],
        [x_max, y_max, z_min], [x_min, y_max, z_min],
        [x_min, y_min, z_max], [x_max, y_min, z_max],
        [x_max, y_max, z_max], [x_min, y_max, z_max],
    ])
    
    edges = [
        (0,1), (1,2), (2,3), (3,0),
        (4,5), (5,6), (6,7), (7,4),
        (0,4), (1,5), (2,6), (3,7),
    ]
    
    bbox_points = []
    for v1_idx, v2_idx in edges:
        v1, v2 = vertices[v1_idx], vertices[v2_idx]
        for t in np.linspace(0, 1, num_points_per_edge):
            point = v1 + t * (v2 - v1)
            bbox_points.append(point)
    
    bbox_points = np.array(bbox_points)
    bbox_colors = np.tile(color, (len(bbox_points), 1))
    
    return bbox_points, bbox_colors


if __name__ == '__main__':
    from utils.config import Config
    
    # Test Kalman smoothing
    print("Test 1: Kalman smoothing")
    noisy_signal = np.array([1.0, 1.2, 0.9, 1.1, 1.3, 0.95, 1.05])
    smoothed = kalman_smooth_1d(noisy_signal, 1e-5, 0.01)
    assert len(smoothed) == len(noisy_signal)
    print(f"  Raw:      {noisy_signal}")
    print(f"  Smoothed: {smoothed}")
    
    # Test class creation
    print("\nTest 2: Create aligner from config")
    config = Config()
    aligner = PointCloudAligner.from_config(config)
    print(f"  Corr conf threshold: {aligner.corr_conf_threshold}")
    print(f"  Complete conf threshold: {aligner.complete_conf_threshold}")
    print(f"  Smoothing enabled: {aligner.enable_smoothing}")
    print(f"  Kalman Q: {aligner.kalman_process_noise}")
    print(f"  Kalman R: {aligner.kalman_measurement_noise}")
