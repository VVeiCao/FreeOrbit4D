"""Point cloud loading, saving, filtering, transformation, and Open3D integration."""

import os
import numpy as np
import trimesh
import open3d as o3d
from typing import Tuple, List, Optional, Dict
from pathlib import Path


# ============================================================================
# Loading and saving
# ============================================================================

def load_pointcloud_ply(ply_path: str, subsample: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """Load a PLY point cloud file and return (points, colors)."""
    ply_path = Path(ply_path)

    if not ply_path.exists():
        raise FileNotFoundError(f"Point cloud file not found: {ply_path}")

    mesh = trimesh.load(str(ply_path))
    points = np.array(mesh.vertices)

    if hasattr(mesh, 'visual') and hasattr(mesh.visual, 'vertex_colors'):
        colors = np.array(mesh.visual.vertex_colors)[:, :3]
    else:
        colors = np.ones((len(points), 3), dtype=np.uint8) * 128

    if subsample > 1:
        indices = np.arange(0, len(points), subsample)
        points = points[indices]
        colors = colors[indices]

    return points, colors


def load_pointcloud_npz(npz_path: str, subsample: int = 1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load an NPZ point cloud file and return (points, colors, view_indices)."""
    npz_path = Path(npz_path)

    if not npz_path.exists():
        raise FileNotFoundError(f"Point cloud file not found: {npz_path}")

    data = np.load(str(npz_path))
    points = data['points']
    colors = data['colors']
    view_indices = data['view_indices']

    if subsample > 1:
        indices = np.arange(0, len(points), subsample)
        points = points[indices]
        colors = colors[indices]
        view_indices = view_indices[indices]

    return points, colors, view_indices


def save_pointcloud_ply(output_path: str,
                        points: np.ndarray,
                        colors: np.ndarray):
    """Save a point cloud as a PLY file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if colors.dtype != np.uint8:
        colors = np.clip(colors, 0, 255).astype(np.uint8)

    mesh = trimesh.Trimesh(vertices=points, vertex_colors=colors)
    mesh.export(str(output_path))


def save_pointcloud_npz(output_path: str,
                        points: np.ndarray,
                        colors: np.ndarray,
                        **kwargs):
    """Save a point cloud as an NPZ file with optional extra fields."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    save_dict = {
        'points': points,
        'colors': colors,
        **kwargs
    }

    np.savez(str(output_path), **save_dict)


# ============================================================================
# Filtering
# ============================================================================

def filter_by_confidence(points: np.ndarray,
                        confidences: np.ndarray,
                        threshold: float) -> Tuple[np.ndarray, np.ndarray]:
    """Filter points by confidence threshold, returning (filtered_points, mask)."""
    mask = confidences >= threshold
    filtered_points = points[mask]
    return filtered_points, mask


def remove_outliers_statistical(pcd: o3d.geometry.PointCloud,
                                nb_neighbors: int = 20,
                                std_ratio: float = 2.0) -> o3d.geometry.PointCloud:
    """Remove statistical outliers from an Open3D point cloud."""
    pcd_filtered, _ = pcd.remove_statistical_outlier(
        nb_neighbors=nb_neighbors,
        std_ratio=std_ratio
    )
    return pcd_filtered


# ============================================================================
# Transformation
# ============================================================================

def transform_points(points: np.ndarray,
                    scale: float = 1.0,
                    translation: np.ndarray = None,
                    rotation: np.ndarray = None) -> np.ndarray:
    """Apply scale, rotation, and translation to points."""
    result = points.copy()

    if scale != 1.0:
        result = result * scale

    if rotation is not None:
        result = result @ rotation.T

    if translation is not None:
        result = result + translation

    return result


# ============================================================================
# Merging
# ============================================================================

def merge_pointclouds(pointcloud_list: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
    """Merge a list of point cloud dicts (each with 'points' and 'colors')."""
    all_points = []
    all_colors = []

    for pc in pointcloud_list:
        if 'points' in pc and len(pc['points']) > 0:
            all_points.append(pc['points'])
            all_colors.append(pc['colors'])

    if len(all_points) == 0:
        return np.array([]).reshape(0, 3), np.array([]).reshape(0, 3)

    all_points = np.concatenate(all_points, axis=0)
    all_colors = np.concatenate(all_colors, axis=0)

    return all_points, all_colors


# ============================================================================
# Open3D conversion
# ============================================================================

def numpy_to_o3d_pointcloud(points: np.ndarray,
                           colors: np.ndarray = None) -> o3d.geometry.PointCloud:
    """Convert numpy arrays to an Open3D PointCloud."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    if colors is not None:
        if colors.dtype == np.uint8:
            colors = colors.astype(np.float64) / 255.0
        pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd


def o3d_to_numpy_pointcloud(pcd: o3d.geometry.PointCloud) -> Tuple[np.ndarray, np.ndarray]:
    """Convert an Open3D PointCloud to numpy arrays (points, colors)."""
    points = np.asarray(pcd.points)

    if pcd.has_colors():
        colors = (np.asarray(pcd.colors) * 255).astype(np.uint8)
    else:
        colors = np.ones((len(points), 3), dtype=np.uint8) * 128

    return points, colors
