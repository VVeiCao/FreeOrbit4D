import torch
import math
import numpy as np
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree

def is_not_nan(*args):
    return all(not isinstance(x, float) or not math.isnan(x) for x in args)

class metrics_log:
    def __init__(self):
        # pose metrics
        self.ATE_list = []
        self.RTE_list = []
        self.RRE_list = []

        self.static_depth = []
        self.dynamic_depth = []
        self.whole_depth = []

        self.static_point = []
        self.dynamic_point = []
        self.whole_point = []
        
        self.static_tracking = []
        self.dynamic_tracking = []
        self.whole_tracking = []


    def log_metrics(self, ATE, RTE, RRE, static_depth, dynamic_depth, whole_depth, static_point, dynamic_point, whole_point, static_tracking, dynamic_tracking, whole_tracking):
        metrics = [ATE, RTE, RRE, static_depth, dynamic_depth, whole_depth,
                static_point, dynamic_point, whole_point, static_tracking, dynamic_tracking, whole_tracking]

        if is_not_nan(*metrics):
            self.ATE_list.append(ATE)
            self.RTE_list.append(RTE)
            self.RRE_list.append(RRE)
            self.static_depth.append(static_depth)
            self.dynamic_depth.append(dynamic_depth)
            self.whole_depth.append(whole_depth)

            self.static_point.append(static_point)
            self.dynamic_point.append(dynamic_point)
            self.whole_point.append(whole_point)
            self.static_tracking.append(static_tracking)
            self.dynamic_tracking.append(dynamic_tracking)
            self.whole_tracking.append(whole_tracking)

    def report_metrics(self):
        print('=============Metrics============')
        print('ATE: ', np.mean(self.ATE_list), 'RTE: ', np.mean(self.RTE_list), 'RRE: ', np.mean(self.RRE_list))
        print('static_depth: ', np.mean(self.static_depth), 'dynamic_depth: ', np.mean(self.dynamic_depth), 'whole_depth: ', np.mean(self.whole_depth))
        print('static_point: ', np.mean(self.static_point), 'dynamic_point: ', np.mean(self.dynamic_point), 'whole_point: ', np.mean(self.whole_point))
        print('static_tracking: ', np.mean(self.static_tracking), 'dynamic_tracking: ', np.mean(self.dynamic_tracking), 'whole_tracking: ', np.mean(self.whole_tracking))

    def reset_metrics(self):
        self.ATE_list = []
        self.RTE_list = []
        self.RRE_list = []
        self.static_depth = []
        self.dynamic_depth = []
        self.whole_depth = []
        self.static_point = []
        self.dynamic_point = []
        self.whole_point = []
        self.static_tracking = []
        self.dynamic_tracking = []
        self.whole_tracking = []

# ----------------------------- Pose metrics -----------------------------
def compute_pose_metrics(pred_extrinsics, gt_extrinsics):
    """
    pred_extrinsics, gt_extrinsics: (B, 2, 3, 4)  # [B, (frame0, frame1), 3, 4]
    Returns:
        ATE_list: Absolute Trajectory Error (m)
        RTE_list: Relative Translation Error (m)
        RRE_list: Relative Rotation Error (deg)
        rra_list: rotation angular error (deg)
        rta_list: translation direction error (deg)
        auc30: Area under curve of (rra + rta) < 30 deg
    """
    def compute_rotation_error(R1, R2):
        R_err = R1 @ R2.T
        cos_theta = (torch.trace(R_err) - 1) / 2
        cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
        return torch.rad2deg(torch.acos(cos_theta))

    def compute_translation_direction_error(t1, t2):
        t1 = t1 / t1.norm()
        t2 = t2 / t2.norm()
        cos_theta = torch.dot(t1.squeeze(), t2.squeeze())
        cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
        return torch.rad2deg(torch.acos(cos_theta))

    ATE_list, RTE_list, RRE_list = [], [], []
    rra_list, rta_list = [], []
    B = pred_extrinsics.shape[0]
    for b in range(B):
        def to_4x4(E):
            T = torch.eye(4, device=E.device)
            T[:3, :] = E
            return T
        T0_pred = to_4x4(pred_extrinsics[b, 0])
        T1_pred = to_4x4(pred_extrinsics[b, 1])
        T0_gt   = to_4x4(gt_extrinsics[b, 0])
        T1_gt   = to_4x4(gt_extrinsics[b, 1])
        # Absolute translation error (ATE)
        t0_pred = T0_pred[:3, 3]
        t0_gt   = T0_gt[:3, 3]
        t1_pred = T1_pred[:3, 3]
        t1_gt   = T1_gt[:3, 3]
        ATE = (t1_pred - t1_gt).norm()
        ATE_list.append(ATE.item())
        # Relative pose
        T_rel_pred = T1_pred @ torch.linalg.inv(T0_pred)
        T_rel_gt   = T1_gt   @ torch.linalg.inv(T0_gt)
        R_rel_pred = T_rel_pred[:3, :3]
        t_rel_pred = T_rel_pred[:3, 3]
        R_rel_gt   = T_rel_gt[:3, :3]
        t_rel_gt   = T_rel_gt[:3, 3]
        # RRE
        RRE = compute_rotation_error(R_rel_pred, R_rel_gt)
        RRE_list.append(RRE.item())
        # RTE
        RTE = (t_rel_pred - t_rel_gt).norm()
        RTE_list.append(RTE.item())
        # Angular metrics
        rra = RRE  # same as RRE
        rta = compute_translation_direction_error(t_rel_pred, t_rel_gt)
        rra_list.append(rra)
        rta_list.append(rta.item())
    # AUC@30 based on angular sum
    total_errors = torch.tensor(rra_list) + torch.tensor(rta_list)
    sorted_errors = torch.sort(total_errors)[0]
    thresholds = torch.linspace(0, 30, 100)
    cdf = torch.tensor([(sorted_errors <= t).float().mean().item() for t in thresholds])
    auc30 = torch.trapz(cdf, thresholds) / 30.0
    return ATE_list, RTE_list, RRE_list, rra_list, rta_list, auc30.item()

def compute_dtu_metrics(pred_points, gt_points, threshold=None):
    """
    pred_points: (N1, 3) numpy or torch tensor
    gt_points:   (N2, 3) numpy or torch tensor
    threshold: optional float to mask out outliers (> threshold distance)
    Returns: accuracy, completeness, overall (all in same unit as coordinates)
    """
    if isinstance(pred_points, torch.Tensor):
        pred_points = pred_points.cpu().numpy()
    if isinstance(gt_points, torch.Tensor):
        gt_points = gt_points.cpu().numpy()
    # Nearest neighbor distances
    pred_tree = cKDTree(pred_points)
    gt_tree = cKDTree(gt_points)
    dists_pred_to_gt, _ = gt_tree.query(pred_points)
    dists_gt_to_pred, _ = pred_tree.query(gt_points)
    if threshold is not None:
        dists_pred_to_gt = dists_pred_to_gt[dists_pred_to_gt < threshold]
        dists_gt_to_pred = dists_gt_to_pred[dists_gt_to_pred < threshold]
    accuracy = np.mean(dists_pred_to_gt) if len(dists_pred_to_gt) > 0 else float('nan')
    completeness = np.mean(dists_gt_to_pred) if len(dists_gt_to_pred) > 0 else float('nan')
    overall = np.nanmean([accuracy, completeness])
    return accuracy, completeness, overall

# ----------------------------- Depth metrics -----------------------------

def backproject_depth_to_points(depth, intrinsics, mask):
    B, H, W = depth.shape
    device = depth.device
    y, x = torch.meshgrid(
        torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
    ones = torch.ones_like(x)
    pix_coords = torch.stack([x, y, ones], dim=-1).float()  # (H, W, 3)
    points_list = []
    for b in range(B):
        mask_b = mask[b]
        depth_b = depth[b][mask_b]
        if depth_b.numel() == 0:
            points_list.append(torch.zeros((0, 3), device=device))
            continue
        pix = pix_coords[mask_b]
        pix = pix @ torch.inverse(intrinsics[b]).T
        pts = pix * depth_b.unsqueeze(-1)
        points_list.append(pts)
    return points_list  # list of (N_i, 3)

def scale_align(pred_points, gt_points):
    s_num = (pred_points * gt_points).sum()
    s_den = (pred_points ** 2).sum()
    s = s_num / (s_den + 1e-8)
    return pred_points * s

def downsample_points(points, max_points=5000):
    if points.shape[0] <= max_points:
        return points
    idx = torch.randperm(points.shape[0], device=points.device)[:max_points]
    return points[idx]

def pairwise_min_distance_chunked(A, B, chunk_size=1024):
    device = A.device
    N1, N2 = A.shape[0], B.shape[0]
    min_dists_A = []
    for i in range(0, N1, chunk_size):
        dists = torch.cdist(A[i:i+chunk_size], B, p=2)
        min_dists_A.append(dists.min(dim=1)[0])
    acc = torch.cat(min_dists_A).mean()
    min_dists_B = []
    for i in range(0, N2, chunk_size):
        dists = torch.cdist(B[i:i+chunk_size], A, p=2)
        min_dists_B.append(dists.min(dim=1)[0])
    comp = torch.cat(min_dists_B).mean()
    return acc, comp

def chamfer_metrics_with_mask(pred_depth, gt_depth, intrinsics, pred_mask, gt_mask, max_points=10000):
    B = pred_depth.shape[0]
    acc_list, comp_list = [], []
    pred_points_list = backproject_depth_to_points(pred_depth, intrinsics, pred_mask)
    gt_points_list   = backproject_depth_to_points(gt_depth, intrinsics, gt_mask)
    for b in range(B):
        P = pred_points_list[b]
        G = gt_points_list[b]
        if P.shape[0] == 0 or G.shape[0] == 0:
            continue
        P = scale_align(P, G)
        P = downsample_points(P, max_points)
        G = downsample_points(G, max_points)
        acc, comp = pairwise_min_distance_chunked(P, G)
        acc_list.append(acc)
        comp_list.append(comp)
    acc = torch.stack(acc_list).mean()
    comp = torch.stack(comp_list).mean()
    overall = (acc + comp) / 2
    return acc.item(), comp.item(), overall.item()

# ----------------------------- Point metrics -----------------------------

def chamfer_from_pointmaps(pred_map, gt_map, pred_mask, gt_mask, max_points=10000):
    B = pred_map.shape[0]
    acc_list, comp_list = [], []
    for b in range(B):
        pred_pts = pred_map[b][pred_mask[b]]
        gt_pts = gt_map[b][gt_mask[b]]
        if pred_pts.shape[0] == 0 or gt_pts.shape[0] == 0:
            continue
        pred_pts = scale_align(pred_pts, gt_pts)
        pred_pts = downsample_points(pred_pts, max_points)
        gt_pts = downsample_points(gt_pts, max_points)
        acc, comp = pairwise_min_distance_chunked(pred_pts, gt_pts)
        acc_list.append(acc)
        comp_list.append(comp)
    acc = torch.stack(acc_list).mean()
    comp = torch.stack(comp_list).mean()
    overall = (acc + comp) / 2
    return acc.item(), comp.item(), overall.item()

# ----------------------------- Tracking metrics -----------------------------
def eval_2d_point_tracking(pred_tracks, gt_tracks, thresholds=[1, 3, 5, 10]):
    """
    Evaluate 2D point tracking accuracy over trajectories.
    Args:
        pred_tracks: Tensor (B, T, N, 2), predicted 2D trajectories.
        gt_tracks:   Tensor (B, T, N, 2), ground truth 2D trajectories.
        thresholds:  List of distance thresholds for PCK metric.
    Returns:
        epe: float, average endpoint error
        pck_dict: dict, PCK@threshold
        auc: float, normalized AUC
    """
    assert pred_tracks.shape == gt_tracks.shape, "Shape mismatch"
    B, T, N, _ = pred_tracks.shape
    # Compute Euclidean distance per point per frame
    dists = torch.norm(pred_tracks - gt_tracks, dim=-1)  # (B, T, N)
    # Average EPE across all frames and points
    epe = dists.mean().item()
    # Compute PCK
    pck_dict = {}
    for th in thresholds:
        correct = (dists <= th).float().mean().item()
        pck_dict[f'PCK@{th}'] = correct
    # AUC over normalized thresholds
    norm_th = torch.tensor(thresholds, dtype=torch.float32)
    pck_values = torch.tensor(list(pck_dict.values()))
    auc = torch.trapz(pck_values, norm_th / norm_th[-1]).item()
    return epe, auc

def get_dynamic_point_mask(points, dynamic_mask):
    """
    Check which points fall in dynamic regions.
    Args:
        points: (N, 2), float or int, pixel coordinates (x, y)
        dynamic_mask: (H, W), bool tensor
    Returns:
        point_mask: (N,) bool tensor — True if the point is in dynamic region
    """
    H, W = dynamic_mask.shape
    points_int = points.round().long()
    # Clamp to image bounds
    u = points_int[:, 0].clamp(0, W - 1)
    v = points_int[:, 1].clamp(0, H - 1)
    return dynamic_mask[v, u]  # (N,)

def compute_depth_error_map(depth_map, depth_gt, mask=None):
    """
    Compute depth error map after scale alignment (batched).
    Inputs:
        depth_map: (B, H, W) unscaled predicted depth
        depth_gt:  (B, H, W) ground truth depth
        mask:      optional (B, H, W) boolean mask of valid pixels
    Returns:
        error_map: (B, H, W) absolute error after scale alignment
        scales:    list of floats, scale per sample
    """
    B = depth_map.shape[0]
    error_maps = []
    scales = []
    for b in range(B):
        d_pred = depth_map[b]
        d_gt = depth_gt[b]
        m = mask[b] if mask is not None else (d_gt > 0) & torch.isfinite(d_gt) & torch.isfinite(d_pred)
        pred = d_pred[m]
        gt = d_gt[m]
        if pred.numel() == 0:
            error_maps.append(torch.full_like(d_pred, float('nan')))
            scales.append(float('nan'))
            continue
        scale = (pred * gt).sum() / (pred ** 2).sum().clamp(min=1e-6)
        aligned_pred = d_pred * scale
        error = torch.abs(aligned_pred - d_gt)
        error_maps.append(error)
        scales.append(scale.item())
    return torch.stack(error_maps), scales

def project_pointcloud_batch(point_map, intrinsics):
    """
    Vectorized projection of batched 3D point maps to depth maps.
    Inputs:
        point_map: (B, H, W, 3) - predicted 3D points in camera coordinates
        intrinsics: (B, 3, 3)   - batched intrinsics
    Returns:
        depth_proj: (B, H, W)
        valid_mask: (B, H, W)
    """
    B, H, W, _ = point_map.shape
    device = point_map.device
    # Flatten for easier batch matrix operations
    points = point_map.view(B, -1, 3)  # (B, HW, 3)
    xyz = points.permute(0, 2, 1)      # (B, 3, HW)
    # Project with intrinsics: [u, v, 1]^T = K @ [x, y, z]^T → (B, 3, HW)
    pixels_homo = torch.bmm(intrinsics, xyz)  # (B, 3, HW)
    x = pixels_homo[:, 0] / pixels_homo[:, 2]
    y = pixels_homo[:, 1] / pixels_homo[:, 2]
    z = points[:, :, 2]  # (B, HW)
    # Round to nearest pixel coordinates
    u = x.round().long().clamp(min=0, max=W - 1)  # (B, HW)
    v = y.round().long().clamp(min=0, max=H - 1)
    # Create batch indices
    batch_idx = torch.arange(B, device=device).view(B, 1).expand(B, H * W)  # (B, HW)
    # Flatten all indices
    flat_indices = (batch_idx.flatten(), v.flatten(), u.flatten())
    # Depth buffer initialization
    depth_proj = torch.zeros(B, H, W, device=device)
    valid_mask = torch.zeros(B, H, W, dtype=torch.bool, device=device)
    # Only keep positive and finite depths
    valid = (z > 0) & torch.isfinite(z)
    z_valid = z[valid]
    idx_valid = (
        batch_idx[valid],
        v[valid],
        u[valid])
    # z-buffer: for simplicity, latest wins (or you can do z-buffer min)
    depth_proj[idx_valid] = z_valid
    valid_mask[idx_valid] = True
    return depth_proj, valid_mask