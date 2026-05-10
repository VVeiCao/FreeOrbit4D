"""Point cloud rendering module with PyTorch3D."""

import os
import json
import cv2
import shutil
import numpy as np
import torch
from pathlib import Path
from typing import Optional, Tuple, Dict, List
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from scipy.spatial.transform import Rotation

try:
    from pytorch3d.structures import Pointclouds
    from pytorch3d.renderer import (
        PointsRasterizationSettings,
        PointsRasterizer,
        PerspectiveCameras
    )
    HAS_PYTORCH3D = True
except ImportError:
    HAS_PYTORCH3D = False

try:
    import splines
    import splines.quaternion
    HAS_SPLINES = True
except ImportError:
    HAS_SPLINES = False

from utils.camera import (
    transform_to_z_up,
    transform_from_z_up,
    transform_rotation_to_z_up,
    transform_rotation_from_z_up,
    load_trajectory_json,
    get_image_size_from_intrinsic
)
from utils.pointcloud import load_pointcloud_ply, load_pointcloud_npz
from utils.file_io import find_frame_dirs, ensure_dir, save_json
from utils.trajectory import format_arc_trajectory_name
from utils.video import images_to_video


def compute_arc_parameters(data_dir: str, frame_id: str = "00000", arc_type: str = "yaw", use_1_view: bool = False) -> Optional[Tuple[np.ndarray, float, float, str]]:
    """Compute arc center, radius, and start angle from foreground point cloud."""
    points_original = None
    if use_1_view:
        fg_ply = os.path.join(data_dir, frame_id, "pointcloud", f"{frame_id}_foreground_1_view.ply")
        if os.path.exists(fg_ply):
            try:
                points_original, _ = load_pointcloud_ply(fg_ply, subsample=1)
            except Exception:
                pass
    if points_original is None:
        fg_npz_smooth = os.path.join(data_dir, frame_id, "pointcloud", f"{frame_id}_foreground_5_views_aligned_smooth.npz")
        fg_npz_aligned = os.path.join(data_dir, frame_id, "pointcloud", f"{frame_id}_foreground_5_views_aligned.npz")
        for npz_path in [fg_npz_smooth, fg_npz_aligned]:
            if os.path.exists(npz_path):
                try:
                    data = np.load(npz_path)
                    points_original = data['points']
                    break
                except Exception:
                    continue
    if points_original is None:
        print(f"  Warning: foreground point cloud not found (1_view PLY or 5_views NPZ)")
        return None
    
    try:
        points_zup = transform_to_z_up(points_original)
        
        if len(points_zup) == 0:
            print(f"  Warning: point cloud is empty")
            return None
        
        depth_min = np.min(points_zup[:, 1])
        
        if arc_type == "yaw":
            # XY plane: horizontal orbit around Z axis
            arc_center = np.array([0.0, depth_min, 0.0])
            arc_radius = depth_min
            start_point = np.array([0.0, 0.0, 0.0])
            vec_to_start = start_point - arc_center
            arc_start_angle = np.arctan2(vec_to_start[1], vec_to_start[0])
        
        elif arc_type == "pitch":
            # YZ plane: vertical pitch around X axis
            arc_center = np.array([0.0, depth_min, 0.0])
            arc_radius = depth_min
            start_point = np.array([0.0, 0.0, 0.0])
            vec_to_start = start_point - arc_center
            arc_start_angle = np.arctan2(vec_to_start[2], vec_to_start[1])
        
        elif arc_type == "roll":
            # XZ plane: lateral roll around Y axis
            arc_center = np.array([0.0, depth_min, 0.0])
            arc_radius = depth_min
            start_point = np.array([0.0, 0.0, 0.0])
            vec_to_start = start_point - arc_center
            arc_start_angle = np.arctan2(vec_to_start[2], vec_to_start[0])
        
        else:
            raise ValueError(f"Unsupported arc_type: {arc_type}. Options: yaw, pitch, roll")
        
        print(f"  Arc: type={arc_type}, center={arc_center}, "
              f"radius={arc_radius:.3f}, start_angle={np.rad2deg(arc_start_angle):.1f} deg")
        
        return arc_center, arc_radius, arc_start_angle, arc_type
    
    except Exception as e:
        print(f"  Error computing arc parameters: {e}")
        import traceback
        traceback.print_exc()
        return None


def generate_arc_keyframes(arc_center: np.ndarray,
                           arc_radius: float,
                           arc_start_angle: float,
                           sweep_angle_deg: float,
                           arc_type: str = "yaw",
                           num_keyframes: int = 8,
                           reference_R: Optional[np.ndarray] = None) -> List[Dict]:
    """Generate arc keyframe data in Z-up coordinate system."""
    keyframes_data = []
    
    sweep_angle = np.deg2rad(sweep_angle_deg)
    angles = np.linspace(arc_start_angle, arc_start_angle + sweep_angle, num_keyframes)
    
    for i, angle in enumerate(angles):
        if arc_type == "yaw":
            # XY plane: rotate around Z axis
            x = arc_center[0] + arc_radius * np.cos(angle)
            y = arc_center[1] + arc_radius * np.sin(angle)
            z = 0.0
            
        elif arc_type == "pitch":
            # YZ plane: rotate around X axis
            x = 0.0
            y = arc_center[1] + arc_radius * np.cos(angle)
            z = arc_center[2] + arc_radius * np.sin(angle)
            
        elif arc_type == "roll":
            # XZ plane: rotate around Y axis
            x = arc_center[0] + arc_radius * np.cos(angle)
            y = arc_center[1]
            z = arc_center[2] + arc_radius * np.sin(angle)
            
        else:
            raise ValueError(f"Unsupported arc_type: {arc_type}. Options: yaw, pitch, roll")
        
        position_zup = np.array([x, y, z])
        
        if reference_R is not None and i == 0:
            R_world_cam_zup = reference_R
        else:
            rotation_angle = angle - arc_start_angle
            cos_a, sin_a = np.cos(rotation_angle), np.sin(rotation_angle)
            
            if arc_type == "yaw":
                R_rotation = np.array([
                    [cos_a, -sin_a, 0],
                    [sin_a, cos_a, 0],
                    [0, 0, 1]
                ])
                
            elif arc_type == "pitch":
                R_rotation = np.array([
                    [1, 0, 0],
                    [0, cos_a, -sin_a],
                    [0, sin_a, cos_a]
                ])
                
            elif arc_type == "roll":
                R_rotation = np.array([
                    [cos_a, 0, sin_a],
                    [0, 1, 0],
                    [-sin_a, 0, cos_a]
                ])
            
            if reference_R is not None:
                R_world_cam_zup = R_rotation @ reference_R
            else:
                R_world_cam_zup = R_rotation
        
        wxyz_zup = Rotation.from_matrix(R_world_cam_zup).as_quat()[[3, 0, 1, 2]]
        
        keyframes_data.append({
            'position_zup': position_zup,
            'wxyz_zup': wxyz_zup,
            'timestep': i
        })
    
    return keyframes_data


def interpolate_arc_trajectory(keyframes_data: List[Dict],
                               num_frames: int,
                               intrinsics: np.ndarray,
                               image_size: Tuple[int, int]) -> Dict:
    """Interpolate arc keyframes into a full camera trajectory using splines."""
    if not HAS_SPLINES:
        raise ImportError("splines library required: pip install splines")
    
    positions_zup = np.array([kf['position_zup'] for kf in keyframes_data])
    wxyzs_zup = np.array([kf['wxyz_zup'] for kf in keyframes_data])
    
    tension = 0.0
    spline_position = splines.KochanekBartels(
        positions_zup.tolist(), tcb=(tension, 0.0, 0.0), endconditions="natural"
    )
    
    quaternions = [splines.quaternion.UnitQuaternion.from_unit_xyzw(np.roll(wxyz, -1)) for wxyz in wxyzs_zup]
    spline_orientation = splines.quaternion.KochanekBartels(
        quaternions, tcb=(tension, 0.0, 0.0), endconditions="natural"
    )
    
    camera_path = []
    t_values = np.linspace(0, len(keyframes_data) - 1, num_frames)
    
    for timestep, t in enumerate(t_values):
        position_zup = spline_position.evaluate(t)
        quat = spline_orientation.evaluate(t)
        wxyz_zup = np.array([quat.scalar, *quat.vector])
        
        # Convert back to original coordinate system (y-down)
        position_original = transform_from_z_up(position_zup.reshape(1, 3)).flatten()
        R_world_cam_zup = Rotation.from_quat(wxyz_zup[[1, 2, 3, 0]]).as_matrix()
        R_world_cam_original = transform_rotation_from_z_up(R_world_cam_zup)
        
        # Build W2C extrinsic matrix
        T_world_cam = np.eye(4)
        T_world_cam[:3, :3] = R_world_cam_original
        T_world_cam[:3, 3] = position_original
        T_cam_world = np.linalg.inv(T_world_cam)
        extrinsic_3x4 = T_cam_world[:3, :]
        
        camera_path.append({
            "timestep": timestep,
            "extrinsic": extrinsic_3x4.tolist(),
            "intrinsic": intrinsics.tolist()
        })
    
    return {
        "camera_path": camera_path,
        "image_size": list(image_size)
    }


def load_foreground_pointcloud_safe(data_dir: str, frame_id: str, subsample: int = 1, use_1_view: bool = False):
    """Load foreground point cloud safely. Returns (points, colors, view_indices) or None."""
    if use_1_view:
        ply_1view = os.path.join(data_dir, frame_id, "pointcloud", f"{frame_id}_foreground_1_view.ply")
        if os.path.exists(ply_1view):
            try:
                points, colors = load_pointcloud_ply(ply_1view, subsample=subsample)
                view_indices = np.zeros(len(points), dtype=np.int32)
                return (points, colors, view_indices)
            except Exception:
                pass
        return None
    
    npz_smooth = os.path.join(data_dir, frame_id, "pointcloud", f"{frame_id}_foreground_5_views_aligned_smooth.npz")
    npz_aligned = os.path.join(data_dir, frame_id, "pointcloud", f"{frame_id}_foreground_5_views_aligned.npz")
    
    for npz_path in [npz_smooth, npz_aligned]:
        if os.path.exists(npz_path):
            try:
                data = np.load(npz_path)
                points = data['points'][::subsample]
                colors = data['colors'][::subsample]
                view_indices = data['view_indices'][::subsample] if 'view_indices' in data else np.zeros(len(points), dtype=np.int32)
                return (points, colors, view_indices)
            except Exception:
                continue
    
    return None


def render_pointcloud_with_multiview_mask_and_depth(points_3d: np.ndarray,
                                                     colors: np.ndarray,
                                                     view_indices: np.ndarray,
                                                     bg_count: int,
                                                     extrinsics: np.ndarray,
                                                     intrinsics: np.ndarray,
                                                     image_shape: Tuple[int, int],
                                                     device: str,
                                                     point_radius_px: float = 1.0,
                                                     points_per_pixel: int = 1) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Render point cloud via PyTorch3D. Returns (rgb_image, mask, multiview_mask, depth_map)."""
    if not HAS_PYTORCH3D:
        raise ImportError("PyTorch3D is required")
    
    H, W = image_shape
    
    def to_tensor(x):
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x).to(device).float()
        return x.to(device).float()
    
    points_3d_tensor = to_tensor(points_3d)
    colors_tensor = to_tensor(colors) / 255.0
    extrinsics = to_tensor(extrinsics)
    intrinsics = to_tensor(intrinsics)
    view_indices_tensor = torch.from_numpy(view_indices).to(device).long()
    
    # OpenCV -> PyTorch3D coordinate conversion
    R_cv, T_cv = extrinsics[:3, :3], extrinsics[:3, 3]
    flip_mat = torch.tensor([[-1, 0, 0], [0, -1, 0], [0, 0, 1]], device=device, dtype=torch.float32)
    R_p3d, T_p3d = flip_mat @ R_cv, flip_mat @ T_cv
    
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    focal_length = torch.stack([fx, fy], dim=0).unsqueeze(0)
    principal_point = torch.stack([cx, cy], dim=0).unsqueeze(0)
    
    cameras = PerspectiveCameras(
        device=device,
        R=R_p3d.T.unsqueeze(0),
        T=T_p3d.unsqueeze(0),
        focal_length=focal_length,
        principal_point=principal_point,
        in_ndc=False,
        image_size=[(H, W)]
    )
    
    pcd = Pointclouds(points=[points_3d_tensor], features=[colors_tensor])
    
    raster_settings = PointsRasterizationSettings(
        image_size=(H, W),
        radius=point_radius_px / min(H, W) * 2.0,
        points_per_pixel=points_per_pixel,
        bin_size=0
    )
    
    rasterizer = PointsRasterizer(cameras=cameras, raster_settings=raster_settings)
    fragments = rasterizer(pcd)
    
    visible_idx = fragments.idx[0, :, :, 0]
    depth_zbuf = fragments.zbuf[0, :, :, 0]
    
    rgb_image = np.zeros((H, W, 3), dtype=np.float32)
    valid_mask = (visible_idx != -1)
    
    if valid_mask.any():
        valid_pixels = torch.where(valid_mask)
        point_indices = visible_idx[valid_pixels].long()
        point_colors = colors_tensor[point_indices]
        rgb_image[valid_pixels[0].cpu().numpy(), valid_pixels[1].cpu().numpy()] = \
            point_colors.detach().cpu().numpy()
    
    rgb_image = (rgb_image * 255).clip(0, 255).astype(np.uint8)
    mask = (np.sum(rgb_image, axis=2) > 5).astype(np.uint8)
    
    visible_idx_cpu = visible_idx.cpu()
    multiview_mask = np.zeros((H, W), dtype=np.uint8)
    
    has_point = (visible_idx_cpu != -1)
    is_foreground = (visible_idx_cpu >= bg_count)
    fg_visible = has_point & is_foreground
    
    if fg_visible.any():
        fg_pixels = torch.where(fg_visible)
        fg_indices = visible_idx_cpu[fg_pixels] - bg_count
        point_views = view_indices_tensor[fg_indices.long()].cpu()
        is_multiview = (point_views >= 0)
        
        valid_rows = fg_pixels[0][is_multiview].numpy()
        valid_cols = fg_pixels[1][is_multiview].numpy()
        multiview_mask[valid_rows, valid_cols] = 1
    
    depth_map = depth_zbuf.cpu().numpy()
    
    return rgb_image, mask, multiview_mask, depth_map


def generate_videos_from_subdirs(subdirs: dict, fps: int, output_root: str):
    """Generate videos from rendered image sequences."""
    import subprocess
    
    videos_dir = os.path.join(output_root, "videos")
    os.makedirs(videos_dir, exist_ok=True)
    
    for name, image_dir in subdirs.items():
        image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
        if not image_files:
            continue
        
        video_path = os.path.join(videos_dir, f"{name}.mp4")
        
        if name == 'original_images':
            input_pattern = os.path.join(image_dir, '%05d_original.png')
        else:
            input_pattern = os.path.join(image_dir, '%05d.png')
        
        cmd = [
            'ffmpeg', '-y',
            '-framerate', str(fps),
            '-i', input_pattern,
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-crf', '18',
            '-preset', 'medium',
            video_path
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            if result.returncode != 0:
                print(f"  Error: failed to generate {name}.mp4")
        except FileNotFoundError:
            print(f"  Error: ffmpeg not found, install with: apt install ffmpeg")
            break
        except Exception as e:
            print(f"  Error generating {name}: {e}")


def organize_inference_structure(output_root: str, subdirs: dict):
    """Organize inference folder structure."""
    inference_dir = os.path.join(output_root, "inference")
    os.makedirs(inference_dir, exist_ok=True)
    
    videos_dir = os.path.join(output_root, "videos")
    
    inference_files = [
        (os.path.join(subdirs['original_images'], "00000_original.png"), 
         os.path.join(inference_dir, "reference_image.png")),
        (os.path.join(videos_dir, "rendered_depths.mp4"), 
         os.path.join(inference_dir, "rendered_depths.mp4")),
        (os.path.join(videos_dir, "original_images.mp4"), 
         os.path.join(inference_dir, "original_images.mp4"))
    ]
    
    for src, dst in inference_files:
        if os.path.exists(src):
            shutil.copy2(src, dst)
        else:
            print(f"  Warning: source file not found: {src}")


class PointCloudRenderer:
    """Point cloud renderer using PyTorch3D."""
    
    def __init__(self,
                 point_radius_px: float = 1.0,
                 image_height: int = 480,
                 image_width: int = 832,
                 depth_percentile_min: float = 1.0,
                 depth_percentile_max: float = 99.0,
                 background_color: int = 127,
                 fps: int = 10,
                 device: str = "cuda",
                 points_per_pixel: int = 1,
                 subsample: int = 1,
                 prefetch_size: int = 8,
                 num_workers: int = 4,
                 output_rendering_base: Optional[str] = None):
        self.point_radius_px = point_radius_px
        self.image_height = image_height
        self.image_width = image_width
        self.depth_percentile_min = depth_percentile_min
        self.depth_percentile_max = depth_percentile_max
        self.background_color = background_color
        self.fps = fps
        self.device = device
        self.points_per_pixel = points_per_pixel
        self.subsample = subsample
        self.prefetch_size = prefetch_size
        self.num_workers = num_workers
        self.output_rendering_base = output_rendering_base
    
    def generate_arc_trajectory(self,
                               data_dir: str,
                               arc_type: str = "yaw",
                               arc_angle: float = 90.0,
                               num_frames: Optional[int] = None,
                               save_json_path: Optional[str] = None,
                               reference_frame_id: str = "00000",
                               fixed_video_time: Optional[int] = None,
                               arc_radius_scale: Optional[float] = None,
                               arc_radius: Optional[float] = None,
                               foreground_1_view: bool = False) -> str:
        """Generate arc trajectory JSON file and return its path."""
        if num_frames is None:
            frame_ids = find_frame_dirs(data_dir)
            num_frames = len(frame_ids)
        
        arc_params = compute_arc_parameters(data_dir, reference_frame_id, arc_type, use_1_view=foreground_1_view)
        if arc_params is None:
            raise ValueError(f"Failed to compute arc parameters (arc_type={arc_type})")
        
        arc_center, computed_arc_radius, arc_start_angle, _ = arc_params
        if arc_radius is not None:
            final_arc_radius = float(arc_radius)
        else:
            scale = 1.0 if arc_radius_scale is None else float(arc_radius_scale)
            final_arc_radius = float(computed_arc_radius) * scale
        
        # Load reference camera parameters from global_camera.json
        camera_json_path = Path(data_dir) / "global_camera.json"
        if not camera_json_path.exists():
            print(f"Warning: global_camera.json not found, using default intrinsics")
            intrinsics = np.array([[640, 0, 320], [0, 640, 240], [0, 0, 1]], dtype=np.float32)
            image_size = (self.image_height, self.image_width)
            reference_R = None
        else:
            with open(camera_json_path, 'r') as f:
                camera_data = json.load(f)
            
            if "image_size" in camera_data:
                image_size = tuple(camera_data["image_size"])
            else:
                image_size = (self.image_height, self.image_width)
            
            if reference_frame_id in camera_data:
                intrinsics = np.array(camera_data[reference_frame_id]['intrinsic'], dtype=np.float32)
                
                T_cam_world = np.vstack([np.array(camera_data[reference_frame_id]['extrinsic']), [0, 0, 0, 1]])
                T_world_cam = np.linalg.inv(T_cam_world)
                R_original = T_world_cam[:3, :3]
                reference_R = transform_rotation_to_z_up(R_original)
            else:
                print(f"Warning: reference frame {reference_frame_id} not found, using defaults")
                intrinsics = np.array([[640, 0, 320], [0, 640, 240], [0, 0, 1]], dtype=np.float32)
                reference_R = None
        
        num_keyframes = min(8, num_frames)
        keyframes_data = generate_arc_keyframes(
            arc_center=arc_center,
            arc_radius=final_arc_radius,
            arc_start_angle=arc_start_angle,
            sweep_angle_deg=arc_angle,
            arc_type=arc_type,
            num_keyframes=num_keyframes,
            reference_R=reference_R
        )
        
        trajectory_data = interpolate_arc_trajectory(
            keyframes_data=keyframes_data,
            num_frames=num_frames,
            intrinsics=intrinsics,
            image_size=image_size
        )
        
        # Orbit mode: all views use the same frame's point cloud
        if fixed_video_time is not None:
            camera_path = trajectory_data["camera_path"]
            for i, cam in enumerate(camera_path):
                cam["video_time"] = fixed_video_time
                cam["output_frame"] = i
        
        if save_json_path:
            output_path = Path(data_dir) / save_json_path
        else:
            trajectory_name = format_arc_trajectory_name(arc_type, arc_angle, arc_radius_scale)
            output_path = Path(data_dir) / f"{trajectory_name}.json"
        
        save_json(trajectory_data, str(output_path))
        
        return str(output_path)
    
    def render_trajectory(self,
                         data_dir: str,
                         trajectory_json: Optional[str] = None,
                         arc_mode: bool = False,
                         arc_type: Optional[str] = None,
                         arc_angle: Optional[float] = None,
                         num_frames: Optional[int] = None,
                         output_dir: Optional[str] = None,
                         fixed_video_time: Optional[int] = None,
                         output_gif: bool = False,
                         foreground_only: bool = False,
                         output_rgba: bool = False,
                         arc_radius_scale: Optional[float] = None,
                         arc_radius: Optional[float] = None,
                         foreground_1_view: bool = False,
                         overwrite: bool = True):
        """Render a full camera trajectory over point cloud frames."""
        if not HAS_PYTORCH3D:
            raise ImportError("PyTorch3D is required: pip install pytorch3d")
        
        trajectory_data = None
        if trajectory_json is None or arc_mode:
            if arc_angle is None:
                arc_angle = 90.0
            if arc_type is None:
                arc_type = "yaw"
            trajectory_json = self.generate_arc_trajectory(
                data_dir, arc_type, arc_angle, num_frames,
                fixed_video_time=fixed_video_time,
                arc_radius_scale=arc_radius_scale,
                arc_radius=arc_radius,
                foreground_1_view=foreground_1_view
            )
            trajectory_data = load_trajectory_json(trajectory_json)
        elif trajectory_json:
            traj_path = Path(trajectory_json)
            
            if not traj_path.is_absolute() and not traj_path.exists():
                traj_path = Path(data_dir) / trajectory_json
            
            trajectory_data = load_trajectory_json(str(traj_path))
        else:
            raise ValueError("Must provide trajectory_json or enable arc_mode")
        
        camera_path = trajectory_data.get("camera_path", [])
        if not camera_path:
            raise ValueError("No camera path in trajectory data")
        
        if "image_size" in trajectory_data:
            H, W = trajectory_data["image_size"]
        else:
            first_intrinsic = np.array(camera_path[0]["intrinsic"], dtype=np.float32)
            H, W = get_image_size_from_intrinsic(first_intrinsic)
        
        if num_frames:
            camera_path = camera_path[:num_frames]
        
        print(f"  Rendering {len(camera_path)} frames at {H}x{W}")
        
        # Set up output directory
        if output_dir:
            output_root = output_dir
        else:
            if self.output_rendering_base:
                if trajectory_json:
                    trajectory_name = os.path.splitext(os.path.basename(trajectory_json))[0]
                    output_root = os.path.join(self.output_rendering_base, trajectory_name)
                else:
                    if arc_type and arc_angle is not None:
                        trajectory_name = format_arc_trajectory_name(arc_type, arc_angle, arc_radius_scale)
                        output_root = os.path.join(self.output_rendering_base, trajectory_name)
                    else:
                        output_root = os.path.join(self.output_rendering_base, "arc_trajectory")
            else:
                dataset_name = os.path.basename(os.path.normpath(data_dir))
                if trajectory_json:
                    trajectory_name = os.path.splitext(os.path.basename(trajectory_json))[0]
                    output_root = os.path.join("outputs", "rendering", dataset_name, trajectory_name)
                else:
                    output_root = os.path.join("outputs", "rendering", dataset_name, "arc_trajectory")
        
        if os.path.exists(output_root):
            shutil.rmtree(output_root)
        
        subdirs = {
            'original_images': os.path.join(output_root, "raw_images", "original_images"),
            'rendered_images': os.path.join(output_root, "raw_images", "rendered_images"),
            'rendered_depths': os.path.join(output_root, "raw_images", "rendered_depths")
        }
        
        for subdir in subdirs.values():
            os.makedirs(subdir, exist_ok=True)
        
        # Load background point cloud
        if foreground_only:
            bg_points, bg_colors, bg_count = None, None, 0
        else:
            bg_ply_path = os.path.join(data_dir, "global_background.ply")
            try:
                bg_points, bg_colors = load_pointcloud_ply(bg_ply_path, subsample=self.subsample)
                bg_count = len(bg_points)
            except FileNotFoundError:
                bg_points, bg_colors, bg_count = None, None, 0
        
        has_bullet_time = trajectory_data.get("metadata", {}).get("has_bullet_time", False)
        
        # Pass 1: render and collect depth values
        rendered_data = []
        all_valid_depths = []
        
        pointcloud_cache = {}
        
        unique_video_times = []
        for cam_info in camera_path:
            video_time = cam_info.get('video_time', cam_info.get('timestep', 0))
            if video_time not in unique_video_times:
                unique_video_times.append(video_time)
        
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            future_to_video_time = {}
            for vt in unique_video_times[:self.prefetch_size]:
                frame_id = f"{int(vt):05d}"
                future = executor.submit(load_foreground_pointcloud_safe, data_dir, frame_id, self.subsample, foreground_1_view)
                future_to_video_time[future] = vt
            
            next_to_load_idx = self.prefetch_size
            
            pbar = tqdm(total=len(camera_path), desc="Rendering")
            
            for idx, cam_info in enumerate(camera_path):
                video_time = cam_info.get('video_time', cam_info.get('timestep', idx))
                output_frame = cam_info.get('output_frame', idx)
                is_bullet_time = cam_info.get('is_bullet_time', False)
                
                frame_id = f"{int(video_time):05d}"
                output_frame_id = f"{int(output_frame):05d}"
                
                extrinsic = np.array(cam_info["extrinsic"], dtype=np.float32)
                intrinsic = np.array(cam_info["intrinsic"], dtype=np.float32)
                
                if video_time in pointcloud_cache:
                    fg_data = pointcloud_cache[video_time]
                else:
                    current_future = None
                    for future, vt in list(future_to_video_time.items()):
                        if vt == video_time:
                            current_future = future
                            break
                    
                    if current_future is not None:
                        fg_data = current_future.result()
                        del future_to_video_time[current_future]
                    else:
                        fg_data = load_foreground_pointcloud_safe(data_dir, frame_id, self.subsample, foreground_1_view)
                    
                    pointcloud_cache[video_time] = fg_data
                
                if next_to_load_idx < len(unique_video_times):
                    next_vt = unique_video_times[next_to_load_idx]
                    if next_vt not in pointcloud_cache:
                        next_frame_id = f"{int(next_vt):05d}"
                        future = executor.submit(load_foreground_pointcloud_safe, data_dir, next_frame_id, self.subsample, foreground_1_view)
                        future_to_video_time[future] = next_vt
                    next_to_load_idx += 1
                
                try:
                    if fg_data is None:
                        if bg_points is not None and len(bg_points) > 0:
                            all_points, all_colors = bg_points, bg_colors
                            view_indices = np.zeros(len(bg_points), dtype=np.int32)
                        else:
                            rendered_data.append(None)
                            pbar.update(1)
                            continue
                    else:
                        fg_points, fg_colors, view_indices = fg_data
                        if bg_points is not None:
                            all_points = np.vstack([bg_points, fg_points])
                            all_colors = np.vstack([bg_colors, fg_colors])
                        else:
                            all_points, all_colors = fg_points, fg_colors
                    
                    rgb_image, mask, multiview_mask, depth_map = render_pointcloud_with_multiview_mask_and_depth(
                        all_points, all_colors, view_indices, bg_count, extrinsic, intrinsic,
                        (H, W), self.device, self.point_radius_px, self.points_per_pixel
                    )
                    
                    rendered_data.append((rgb_image, mask, depth_map, output_frame_id, frame_id, is_bullet_time))
                    
                    if mask.any():
                        valid_depths = depth_map[mask == 1]
                        valid_depths = valid_depths[valid_depths > 0]
                        if len(valid_depths) > 0:
                            all_valid_depths.append(valid_depths)
                except Exception as e:
                    rendered_data.append(None)
                
                pbar.update(1)
            
            pbar.close()
        
        pointcloud_cache.clear()
        
        if not all_valid_depths:
            print(f"Error: no valid depth values found")
            return
        
        all_valid_depths_concat = np.concatenate(all_valid_depths)
        depth_min_global = np.percentile(all_valid_depths_concat, self.depth_percentile_min)
        depth_max_global = np.percentile(all_valid_depths_concat, self.depth_percentile_max)
        
        # Pass 2: save all outputs
        success_count = 0
        bullet_time_count = 0
        for data in tqdm(rendered_data, desc="Saving"):
            if data is None:
                continue
            
            if len(data) == 6:
                rgb_image, mask, depth_map, output_frame_id, video_frame_id, is_bullet_time = data
            else:
                rgb_image, mask, depth_map, output_frame_id = data
                video_frame_id = output_frame_id
                is_bullet_time = False
            
            mask_binary = mask.astype(bool)
            
            if is_bullet_time:
                bullet_time_count += 1
            
            original_image_path = os.path.join(data_dir, video_frame_id, "images", f"{video_frame_id}_original.png")
            if os.path.exists(original_image_path):
                shutil.copy(original_image_path, os.path.join(subdirs['original_images'], f"{output_frame_id}_original.png"))
            
            rgb_filled = rgb_image.copy()
            rgb_filled[~mask_binary] = self.background_color
            
            if output_rgba:
                rgba = np.zeros((H, W, 4), dtype=np.uint8)
                rgba[:, :, :3] = rgb_image
                rgba[mask_binary, 3] = 255
                rgba[~mask_binary, 3] = 0
                rgba[~mask_binary, :3] = 0
                bgra = np.dstack([rgba[:, :, 2], rgba[:, :, 1], rgba[:, :, 0], rgba[:, :, 3]])
                cv2.imwrite(os.path.join(subdirs['rendered_images'], f"{output_frame_id}.png"), bgra)
            else:
                cv2.imwrite(os.path.join(subdirs['rendered_images'], f"{output_frame_id}.png"), 
                           cv2.cvtColor(rgb_filled, cv2.COLOR_RGB2BGR))
            
            depth_img = np.full((H, W), self.background_color, dtype=np.uint8)
            if mask_binary.any() and depth_max_global > depth_min_global:
                depth_norm = (depth_map - depth_min_global) / (depth_max_global - depth_min_global)
                depth_uint8 = (254 - (depth_norm * 252 + 1)).clip(1, 253).astype(np.uint8)
                depth_img[mask_binary] = depth_uint8[mask_binary]
            cv2.imwrite(os.path.join(subdirs['rendered_depths'], f"{output_frame_id}.png"), 
                       np.stack([depth_img]*3, axis=2))
            
            success_count += 1
        
        print(f"Done: {success_count}/{len(camera_path)} frames rendered")
        print(f"Output: {output_root}")
        
        if success_count > 0:
            generate_videos_from_subdirs(subdirs, self.fps, output_root)
            if not foreground_only:
                organize_inference_structure(output_root, subdirs)
            if output_gif:
                from utils.video import images_to_gif
                rendered_dir = subdirs["rendered_images"]
                gif_path = os.path.join(output_root, "orbit.gif")
                if not images_to_gif(rendered_dir, gif_path, fps=self.fps, rgba=output_rgba):
                    print(f"  Warning: GIF generation failed")
        
        return output_root
    
    @classmethod
    def from_config(cls, config):
        """Create renderer from config."""
        from utils.config import Config
        if not isinstance(config, Config):
            config = Config(config)
        
        return cls(
            point_radius_px=config.get('stage_1.rendering.point_radius_px', 1.0),
            image_height=config.get('stage_1.rendering.image_height', 480),
            image_width=config.get('stage_1.rendering.image_width', 832),
            depth_percentile_min=config.get('stage_1.rendering.depth_percentile_min', 1.0),
            depth_percentile_max=config.get('stage_1.rendering.depth_percentile_max', 99.0),
            background_color=config.get('stage_1.rendering.background_color', 127),
            fps=config.get('stage_1.rendering.fps', 10),
            device=config.get('common.device', 'cuda'),
            output_rendering_base=config.get('project.output_rendering_base')
        )


if __name__ == '__main__':
    from utils.config import Config
    
    config = Config()
    renderer = PointCloudRenderer.from_config(config)
    print(f"Renderer created: {renderer.image_width}x{renderer.image_height}, "
          f"radius={renderer.point_radius_px}px, fps={renderer.fps}")
