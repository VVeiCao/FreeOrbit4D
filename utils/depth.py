"""Depth map backprojection, normalization, and visualization utilities."""

import torch
import numpy as np
from typing import Tuple


def backproject_depth_to_points_batch(depths, intrinsics, extrinsics_3x4):
    """Backproject depth maps to world-space 3D points.

    Args:
        depths: (B, H, W) depth tensor.
        intrinsics: (B, 3, 3) camera intrinsics tensor.
        extrinsics_3x4: (B, 3, 4) World2Cam extrinsics tensor.

    Returns:
        (B, H*W, 3) world-space points tensor.
    """
    B, H, W = depths.shape
    device = depths.device
    dtype = depths.dtype

    # Pixel grid (x, y)
    y, x = torch.meshgrid(
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing='ij'
    )

    # Homogeneous pixel coordinates (x, y, 1)
    pixels = torch.stack([x, y, torch.ones_like(x)], dim=-1)  # (H, W, 3)
    pixels = pixels.reshape(1, H*W, 3).expand(B, -1, -1).float()  # (B, H*W, 3)

    # Pixel coords -> camera coords: pixels @ K^{-T}
    intrinsics_inv = torch.inverse(intrinsics)  # (B, 3, 3)
    cam_coords = torch.bmm(pixels, intrinsics_inv.transpose(1, 2))  # (B, H*W, 3)

    # Scale by depth
    cam_points = cam_coords * depths.reshape(B, -1, 1)  # (B, H*W, 3)

    # Homogeneous camera points
    cam_points_h = torch.cat([
        cam_points,
        torch.ones((B, cam_points.shape[1], 1), device=device, dtype=dtype)
    ], dim=-1)  # (B, H*W, 4)

    # Camera coords -> world coords
    bottom = torch.tensor([[0, 0, 0, 1]], device=device, dtype=dtype).expand(B, 1, 4)
    T_cam_world = torch.cat([extrinsics_3x4, bottom], dim=1)  # (B, 4, 4)

    T_world_cam = torch.inverse(T_cam_world)  # (B, 4, 4)

    world_points_h = torch.bmm(cam_points_h, T_world_cam.transpose(1, 2))  # (B, H*W, 4)
    world_points = world_points_h[:, :, :3]  # (B, H*W, 3)

    return world_points


def normalize_depth(depth_map: np.ndarray,
                    min_percentile: float = 1.0,
                    max_percentile: float = 99.0) -> np.ndarray:
    """Normalize a depth map to [0, 255] uint8 using percentile clipping."""
    valid_mask = depth_map > 0
    if not valid_mask.any():
        return np.zeros_like(depth_map, dtype=np.uint8)

    valid_depths = depth_map[valid_mask]

    depth_min = np.percentile(valid_depths, min_percentile)
    depth_max = np.percentile(valid_depths, max_percentile)

    depth_normalized = np.clip(depth_map, depth_min, depth_max)
    depth_normalized = (depth_normalized - depth_min) / (depth_max - depth_min + 1e-8)
    depth_normalized = (depth_normalized * 255).astype(np.uint8)

    return depth_normalized
