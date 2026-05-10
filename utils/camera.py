"""Camera utilities: intrinsic scaling, coordinate transforms, and trajectory I/O."""

import json
import numpy as np
import torch
from typing import Tuple, Dict, Optional
from pathlib import Path


def scale_intrinsics(intrinsics, old_size: Tuple[int, int], new_size: Tuple[int, int]):
    """Scale camera intrinsics from old_size (H, W) to new_size (H, W)."""
    scale_x = new_size[1] / old_size[1]  # W_new / W_old
    scale_y = new_size[0] / old_size[0]  # H_new / H_old

    is_torch = isinstance(intrinsics, torch.Tensor)

    if is_torch:
        intrinsics_scaled = intrinsics.clone()
    else:
        intrinsics_scaled = intrinsics.copy()

    intrinsics_scaled[..., 0, 0] *= scale_x  # fx
    intrinsics_scaled[..., 1, 1] *= scale_y  # fy
    intrinsics_scaled[..., 0, 2] *= scale_x  # cx
    intrinsics_scaled[..., 1, 2] *= scale_y  # cy

    return intrinsics_scaled


# ============================================================================
# Coordinate system transforms (y-down <-> z-up)
# ============================================================================

def transform_to_z_up(points: np.ndarray) -> np.ndarray:
    """Convert points from y-down to z-up: (x, y, z) -> (x, z, -y)."""
    points_flat = points.reshape(-1, 3)
    result = np.column_stack([
        points_flat[:, 0],
        points_flat[:, 2],
        -points_flat[:, 1]
    ])
    return result.reshape(points.shape)


def transform_rotation_to_z_up(R: np.ndarray) -> np.ndarray:
    """Convert a rotation matrix from y-down to z-up coordinate system."""
    transform_matrix = np.array([
        [1,  0,  0],
        [0,  0,  1],
        [0, -1,  0]
    ], dtype=np.float64)

    return transform_matrix @ R


def transform_from_z_up(points: np.ndarray) -> np.ndarray:
    """Convert points from z-up to y-down: (x, y, z) -> (x, -z, y)."""
    points_flat = points.reshape(-1, 3)
    result = np.column_stack([
        points_flat[:, 0],
        -points_flat[:, 2],
        points_flat[:, 1]
    ])
    return result.reshape(points.shape)


def transform_rotation_from_z_up(R: np.ndarray) -> np.ndarray:
    """Convert a rotation matrix from z-up to y-down coordinate system."""
    transform_matrix = np.array([
        [1,  0,  0],
        [0,  0, -1],
        [0,  1,  0]
    ], dtype=np.float64)

    return transform_matrix @ R


# ============================================================================
# Trajectory file handling
# ============================================================================

def load_trajectory_json(json_path: str) -> Dict:
    """Load a camera trajectory JSON file, auto-detecting and converting format.

    Supports camera_path format (standard) and global_camera format (from step 1.1,
    auto-converted to camera_path format).
    """
    json_path = Path(json_path)

    if not json_path.exists():
        raise FileNotFoundError(f"Trajectory file not found: {json_path}")

    with open(json_path, 'r', encoding='utf-8') as f:
        trajectory_data = json.load(f)

    image_size = None
    if "image_size" in trajectory_data:
        image_size = trajectory_data["image_size"]
    elif "metadata" in trajectory_data and "image_size" in trajectory_data["metadata"]:
        image_size = trajectory_data["metadata"]["image_size"]

    if "camera_path" in trajectory_data:
        if image_size is not None:
            trajectory_data["image_size"] = image_size
        return trajectory_data

    if isinstance(trajectory_data, dict):
        first_frame_key = None
        for key in trajectory_data.keys():
            if key.isdigit():
                first_frame_key = key
                break

        if first_frame_key is not None:
            first_value = trajectory_data[first_frame_key]
            if isinstance(first_value, dict) and "extrinsic" in first_value and "intrinsic" in first_value:
                camera_path = []
                for frame_id_str in sorted([k for k in trajectory_data.keys() if k.isdigit()]):
                    frame_data = trajectory_data[frame_id_str]
                    timestep = int(frame_id_str)
                    camera_path.append({
                        "timestep": timestep,
                        "extrinsic": frame_data["extrinsic"],
                        "intrinsic": frame_data["intrinsic"]
                    })

                converted_data = {"camera_path": camera_path}
                if image_size is not None:
                    converted_data["image_size"] = image_size
                return converted_data

    return trajectory_data


def get_image_size_from_intrinsic(intrinsic: np.ndarray) -> Tuple[int, int]:
    """Infer (height, width) from intrinsic matrix, assuming principal point is at image center."""
    cx = intrinsic[0, 2]
    cy = intrinsic[1, 2]

    width = int(2 * cx)
    height = int(2 * cy)

    return height, width


# ============================================================================
# Camera pose utilities
# ============================================================================

def extrinsic_to_pose(extrinsic_3x4: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Extract camera position and rotation (world frame) from a [R|t] extrinsic matrix."""
    T_cam_world = np.vstack([extrinsic_3x4, [0, 0, 0, 1]])
    T_world_cam = np.linalg.inv(T_cam_world)

    position = T_world_cam[:3, 3]
    rotation = T_world_cam[:3, :3]

    return position, rotation


def pose_to_extrinsic(position: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    """Build a [R|t] extrinsic matrix from camera position and rotation (world frame)."""
    T_world_cam = np.eye(4)
    T_world_cam[:3, :3] = rotation
    T_world_cam[:3, 3] = position

    T_cam_world = np.linalg.inv(T_world_cam)
    extrinsic_3x4 = T_cam_world[:3, :]

    return extrinsic_3x4
