#!/usr/bin/env python3
"""Step 1.3: Camera trajectory editor (Z-up coordinate system) with Viser GUI."""

import os
import glob
import time
import argparse
import json
import colorsys
import sys
import warnings
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass
import numpy as np
import trimesh
import viser
from scipy.spatial.transform import Rotation

try:
    import splines
    import splines.quaternion
    HAS_SPLINES = True
except ImportError:
    print("Warning: splines library not found. Install: pip install splines")
    HAS_SPLINES = False


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FRAME_VERSIONS = ("single_view", "aligned", "smooth")
DISPLAY_VERSION_ORDER = ("smooth", "aligned", "single_view")
DEFAULT_WXYZ = np.array([1.0, 0.0, 0.0, 0.0])


@dataclass
class Keyframe:
    timestep: int
    position: np.ndarray  # (3,) Z-up
    wxyz: np.ndarray      # (4,) quaternion [w,x,y,z]
    fov: float
    aspect: float


@dataclass
class ArcSettings:
    center: np.ndarray
    radius: float
    num_keyframes: int
    yaw_angle_deg: float
    pitch_angle_deg: float
    roll_angle_deg: float


def transform_to_z_up(points: np.ndarray) -> np.ndarray:
    """y-down -> z-up: (x,y,z) -> (x,z,-y)"""
    points_flat = points.reshape(-1, 3)
    return np.column_stack([points_flat[:, 0], points_flat[:, 2], -points_flat[:, 1]]).reshape(points.shape)

def transform_rotation_to_z_up(R: np.ndarray) -> np.ndarray:
    return np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float64) @ R

def transform_from_z_up(points: np.ndarray) -> np.ndarray:
    """z-up -> y-down: (x,y,z) -> (x,-z,y)"""
    points_flat = points.reshape(-1, 3)
    return np.column_stack([points_flat[:, 0], -points_flat[:, 2], points_flat[:, 1]]).reshape(points.shape)

def transform_rotation_from_z_up(R: np.ndarray) -> np.ndarray:
    return np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64) @ R


def safe_float(value, default: float, min_value: float = None) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(value):
        return default
    if min_value is not None:
        value = max(value, min_value)
    return value


def safe_int(value, default: int, min_value: int = None) -> int:
    try:
        value = int(value)
    except (TypeError, ValueError):
        return default
    if min_value is not None:
        value = max(value, min_value)
    return value


def rotation_matrix_to_wxyz(R: np.ndarray, fallback: np.ndarray = None) -> np.ndarray:
    if fallback is None:
        fallback = DEFAULT_WXYZ

    R = np.asarray(R, dtype=np.float64)
    if R.shape != (3, 3) or not np.isfinite(R).all():
        return fallback

    try:
        U, _, Vt = np.linalg.svd(R)
        R_orth = U @ Vt
        if np.linalg.det(R_orth) < 0:
            U[:, -1] *= -1
            R_orth = U @ Vt
        return Rotation.from_matrix(R_orth).as_quat()[[3, 0, 1, 2]]
    except (ValueError, np.linalg.LinAlgError):
        return fallback


def remove_scene_handles(handles: List, clear: bool = True):
    for handle in list(handles):
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Attempted to remove already removed node.*")
                handle.remove()
        except Exception:
            pass
    if clear:
        handles.clear()


def set_handles_visible(handles: List, visible: bool):
    for handle in handles:
        try:
            handle.visible = visible
        except Exception:
            pass


def show_modal(client: Optional[viser.ClientHandle], title: str, markdown: str):
    if client is None:
        return
    with client.gui.add_modal(title) as modal:
        client.gui.add_markdown(markdown)
        close_button = client.gui.add_button("OK")

        @close_button.on_click
        def _(_) -> None:
            modal.close()


def normalized_json_filename(filename: str) -> str:
    return filename if filename.endswith(".json") else f"{filename}.json"


def load_ply_points(path: str, default_color, subsample: int = 1):
    mesh = trimesh.load(path)
    points = np.asarray(mesh.vertices)
    if hasattr(mesh, 'visual') and hasattr(mesh.visual, 'vertex_colors'):
        colors = np.asarray(mesh.visual.vertex_colors)[:, :3]
    else:
        colors = np.ones((len(points), 3), dtype=np.uint8) * np.asarray(default_color, dtype=np.uint8)

    if len(colors) != len(points):
        colors = np.ones((len(points), 3), dtype=np.uint8) * np.asarray(default_color, dtype=np.uint8)

    if subsample > 1:
        indices = np.arange(0, len(points), subsample)
        points, colors = points[indices], colors[indices]

    return transform_to_z_up(points), colors


def parse_args():
    parser = argparse.ArgumentParser(description="Step 1.3: Camera trajectory editor")
    parser.add_argument("--config", type=str, default=None, help="Scene config; uses project.output_prepared as data_dir")
    parser.add_argument("--data_dir", type=str, default=None, help="Data directory")
    parser.add_argument("--port", type=int, default=8080, help="Viser port")
    parser.add_argument("--point_size", type=float, default=0.002, help="Background point size")
    parser.add_argument("--foreground_point_size", type=float, default=0.001, help="Foreground point size")
    parser.add_argument("--fps", type=float, default=5.0, help="Playback FPS")
    parser.add_argument("--subsample", type=int, default=2, help="Subsample rate")
    parser.add_argument("--num_frames", type=int, default=None, help="Number of frames to load")
    parser.add_argument("--show_bbox", action="store_true", help="Show AABB bounding box")
    args = parser.parse_args()

    if args.config and args.data_dir is None:
        if str(PROJECT_ROOT) not in sys.path:
            sys.path.insert(0, str(PROJECT_ROOT))
        from utils.config import Config
        args.data_dir = Config(args.config).get("project.output_prepared")

    if args.data_dir is None:
        args.data_dir = "outputs/prepared/camel"

    return args


class CameraTrajectoryEditor:
    """Camera trajectory editor with Viser GUI (Z-up coordinate system)."""

    def __init__(self, data_dir: str, port: int = 8080, point_size: float = 0.003,
                 foreground_point_size: float = None,
                 fps: float = 5.0, subsample: int = 1, num_frames: int = None, show_bbox: bool = False):
        self.data_dir = data_dir
        self.port = port
        self.point_size = point_size
        self.foreground_point_size = foreground_point_size if foreground_point_size is not None else point_size
        self.fps = fps
        self.subsample = subsample
        self.num_frames_limit = num_frames
        self.show_bbox = show_bbox
        self.image_width, self.image_height = 640, 480

        self.background_data = self._load_global_background()
        self.camera_data = self._load_camera_parameters()
        self.intrinsics_frame0 = self._extract_frame0_intrinsics()
        self.foreground_frames = self._load_aligned_foreground_frames()
        self.num_frames = max([len(v) for v in self.foreground_frames.values()]) if self.foreground_frames else 0

        if self.num_frames == 0 and self.background_data is None:
            raise ValueError("No valid point cloud files found")

        self.keyframes: Dict[int, Keyframe] = {}
        self.keyframe_counter = 0
        self.keyframe_handles: Dict[int, viser.CameraFrustumHandle] = {}
        self.trajectory_spline_position = None
        self.trajectory_spline_orientation = None
        self.trajectory_spline_timestep = None
        self.trajectory_handles: List[viser.SceneNodeHandle] = []
        self.trajectory_camera_handle: Optional[viser.CameraFrustumHandle] = None
        self.scene_center = None

        self.arc_preview_handles: List[viser.SceneNodeHandle] = []
        self.arc_params = {
            axis: {'center': None, 'base_radius': None, 'start_angle': None}
            for axis in ("yaw", "pitch", "roll")
        }

        self.available_trajectories: Dict[str, Dict] = {}
        self.trajectory_colors = [
            (255, 100, 100), (100, 255, 100), (100, 100, 255), (255, 255, 100),
            (76, 114, 176), (85, 168, 104), (255, 165, 0), (128, 0, 128),
        ]
        self.trajectory_color_order: List[str] = []

        self._scan_trajectory_files()

        self.server = self._setup_viser()
        self._add_initial_keyframe()
        self._compute_arc_parameters()

    def _load_global_background(self) -> Optional[Dict]:
        bg_path = os.path.join(self.data_dir, "global_background.ply")
        if not os.path.exists(bg_path):
            return None
        try:
            points, colors = load_ply_points(bg_path, default_color=[150, 150, 150], subsample=self.subsample)
            return {'points': points, 'colors': colors}
        except Exception as e:
            print(f"Failed to load background: {e}")
            return None

    def _load_camera_parameters(self) -> Optional[Dict]:
        camera_json_path = os.path.join(self.data_dir, "global_camera.json")
        if not os.path.exists(camera_json_path):
            return None
        try:
            with open(camera_json_path, 'r') as f:
                camera_data = json.load(f)
            if "image_size" in camera_data:
                self.image_height, self.image_width = camera_data["image_size"]
            camera_data_processed = {}
            for frame_id, cam_info in camera_data.items():
                if not isinstance(cam_info, dict) or "extrinsic" not in cam_info:
                    continue
                T_cam_world = np.vstack([np.array(cam_info['extrinsic']), [0, 0, 0, 1]])
                T_world_cam = np.linalg.inv(T_cam_world)
                position_old, R_old = T_world_cam[:3, 3], T_world_cam[:3, :3]
                position_new = transform_to_z_up(position_old.reshape(1, 3)).flatten()
                R_new = transform_rotation_to_z_up(R_old)
                T_world_cam_new = np.eye(4)
                T_world_cam_new[:3, :3] = R_new
                T_world_cam_new[:3, 3] = position_new
                camera_data_processed[frame_id] = {
                    'extrinsic': np.linalg.inv(T_world_cam_new)[:3, :],
                    'intrinsic': np.array(cam_info['intrinsic']),
                    'position': position_new,
                    'R_world_cam': R_new
                }
            return camera_data_processed
        except Exception as e:
            print(f"Failed to load camera parameters: {e}")
            return None

    def _extract_frame0_intrinsics(self) -> Optional[np.ndarray]:
        if not self.camera_data or '00000' not in self.camera_data:
            self.unified_fov = np.deg2rad(60.0)
            self.unified_aspect = self.image_width / self.image_height
            return None
        intrinsics = self.camera_data['00000']['intrinsic']
        self.unified_fov = 2 * np.arctan(self.image_height / (2 * intrinsics[1, 1]))
        self.unified_aspect = self.image_width / self.image_height
        return intrinsics

    def _load_aligned_foreground_frames(self) -> Dict[str, List[Dict]]:
        subdirs = sorted([d for d in os.listdir(self.data_dir)
                         if os.path.isdir(os.path.join(self.data_dir, d)) and d.isdigit()])
        if self.num_frames_limit:
            subdirs = subdirs[:self.num_frames_limit]

        foreground_dict = {version: [] for version in FRAME_VERSIONS}

        self.solid_colors = {
            'single_view': np.array([255, 165, 0], dtype=np.uint8),
            'aligned': np.array([100, 255, 100], dtype=np.uint8),
            'smooth': np.array([100, 150, 255], dtype=np.uint8)
        }

        patterns = {
            'single_view': '{frame_id}_foreground_1_view.ply',
            'aligned': '{frame_id}_foreground_5_views_aligned.ply',
            'smooth': '{frame_id}_foreground_5_views_aligned_smooth.ply'
        }

        counts = {version: 0 for version in FRAME_VERSIONS}

        for frame_id in subdirs:
            for version, pattern in patterns.items():
                fg_path = os.path.join(self.data_dir, frame_id, 'pointcloud',
                                      pattern.format(frame_id=frame_id))
                if os.path.exists(fg_path):
                    try:
                        points, colors = load_ply_points(
                            fg_path,
                            default_color=[150, 150, 150],
                            subsample=self.subsample,
                        )
                        solid_colors = np.tile(self.solid_colors[version], (len(points), 1))
                        foreground_dict[version].append({
                            'frame_id': frame_id,
                            'points': points,
                            'colors': colors,
                            'solid_colors': solid_colors
                        })
                        counts[version] += 1
                    except Exception:
                        pass

        self._num_single_view_loaded = counts['single_view']
        self._num_aligned_loaded = counts['aligned']
        self._num_smooth_loaded = counts['smooth']

        return foreground_dict

    def _scan_trajectory_files(self):
        json_files = glob.glob(os.path.join(self.data_dir, "*.json"))
        if not json_files:
            return
        for traj_file in sorted(json_files):
            filename = os.path.basename(traj_file)
            try:
                with open(traj_file, 'r') as f:
                    traj_data = json.load(f)
                if filename == "global_camera.json":
                    converted_data = self._convert_global_camera_to_trajectory(traj_data)
                    if converted_data is None:
                        continue
                    traj_data = converted_data
                    num_frames = len(traj_data['camera_path'])
                else:
                    if 'camera_path' not in traj_data and 'keyframes' not in traj_data:
                        continue
                    if 'camera_path' in traj_data:
                        num_frames = len(traj_data['camera_path'])
                    elif 'keyframes' in traj_data:
                        num_frames = len(traj_data['keyframes'])
                    else:
                        num_frames = 0
                self.available_trajectories[filename] = {
                    'data': traj_data, 'handles': [], 'checkbox': None, 'num_frames': num_frames
                }
            except Exception:
                pass

    def _convert_global_camera_to_trajectory(self, camera_data: Dict) -> Optional[Dict]:
        try:
            camera_path = []
            image_size = camera_data.get("image_size", [self.image_height, self.image_width])
            frame_ids = sorted([k for k in camera_data.keys() if k.isdigit() or k.replace('_', '').isdigit()])
            for idx, frame_id in enumerate(frame_ids):
                frame_data = camera_data[frame_id]
                if 'extrinsic' in frame_data and 'intrinsic' in frame_data:
                    camera_path.append({
                        'timestep': idx, 'extrinsic': frame_data['extrinsic'],
                        'intrinsic': frame_data['intrinsic']
                    })
            if len(camera_path) == 0:
                return None
            return {'camera_path': camera_path, 'image_size': image_size}
        except Exception:
            return None

    def _setup_viser(self) -> viser.ViserServer:
        server = viser.ViserServer(host="0.0.0.0", port=self.port)
        server.gui.configure_theme(titlebar_content=None, control_layout="collapsible")
        self._create_gui(server)
        self._create_scene(server)
        self._setup_initial_camera(server)
        return server

    def _setup_initial_camera(self, server: viser.ViserServer):
        all_points = []
        if self.background_data:
            all_points.append(self.background_data['points'])
        for version in FRAME_VERSIONS:
            for fg_frame in self.foreground_frames[version]:
                all_points.append(fg_frame['points'])
        if not all_points:
            return
        all_points_combined = np.concatenate(all_points)
        bbox_min, bbox_max = np.min(all_points_combined, axis=0), np.max(all_points_combined, axis=0)
        self.scene_center = (bbox_min + bbox_max) / 2.0
        max_extent = np.max(bbox_max - bbox_min)
        initial_position = (self.camera_data['00000']['position'] if self.camera_data and '00000' in self.camera_data
                           else self.scene_center + np.array([0, -max_extent*1.5, max_extent*0.5]))

        @server.on_client_connect
        def _(client: viser.ClientHandle) -> None:
            client.camera.position = initial_position
            client.camera.look_at = self.scene_center
            client.camera.up_direction = np.array([0.0, 0.0, 1.0])

    def _add_initial_keyframe(self):
        if not self.camera_data or '00000' not in self.camera_data:
            return
        try:
            cam_info = self.camera_data['00000']
            keyframe = self._make_keyframe(
                timestep=0,
                position=cam_info['position'],
                wxyz=rotation_matrix_to_wxyz(cam_info['R_world_cam']),
            )
            self._add_keyframe(keyframe)
            self._update_stats()
        except Exception:
            pass

    def _camera_fov_aspect(self):
        fov = self.unified_fov if self.unified_fov is not None else np.deg2rad(60.0)
        aspect = self.unified_aspect if self.unified_aspect is not None else (self.image_width / self.image_height)
        return fov, aspect

    def _make_keyframe(self, timestep: int, position: np.ndarray, wxyz: np.ndarray) -> Keyframe:
        fov, aspect = self._camera_fov_aspect()
        return Keyframe(timestep=timestep, position=position, wxyz=wxyz, fov=fov, aspect=aspect)

    def _add_keyframe(self, keyframe: Keyframe) -> int:
        keyframe_id = self.keyframe_counter
        self.keyframes[keyframe_id] = keyframe
        self.keyframe_counter += 1
        self._visualize_keyframe(keyframe_id, keyframe)
        return keyframe_id

    def _create_gui(self, server: viser.ViserServer):
        with server.gui.add_folder("Time Control"):
            self.gui_timestep = server.gui.add_slider(
                "Frame", min=0, max=max(self.num_frames - 1, 0), step=1, initial_value=0,
            )
            first_frame_id = "N/A"
            for version in DISPLAY_VERSION_ORDER:
                if len(self.foreground_frames[version]) > 0:
                    first_frame_id = self.foreground_frames[version][0]['frame_id']
                    break
            self.gui_frame_id_label = server.gui.add_text("Frame ID", initial_value=first_frame_id, disabled=True)
            self.gui_play_button = server.gui.add_button("Play", icon=viser.Icon.PLAYER_PLAY)
            self.gui_pause_button = server.gui.add_button("Pause", icon=viser.Icon.PLAYER_PAUSE, visible=False)
            self.gui_framerate = server.gui.add_slider("FPS", min=1, max=30, step=0.5, initial_value=self.fps)
            self.gui_view_mode = server.gui.add_button_group(
                "Playback View", ("First Person", "Third Person"),
                hint="First Person: follow camera | Third Person: observe camera"
            )
            self.gui_view_mode.value = "First Person"
            self.is_playing = False

        self.gui_reset_camera = server.gui.add_button("Reset View", icon=viser.Icon.VIEWFINDER)

        with server.gui.add_folder("Arc Trajectory Generator"):
            self.gui_arc_radius_scale = server.gui.add_slider(
                "Radius Scale", min=0.3, max=3.0, step=0.1, initial_value=1.0,
                hint="1.0=default, >1 camera moves back, <1 camera moves closer"
            )
            self.gui_arc_yaw_enabled = server.gui.add_checkbox("Yaw (Horizontal Orbit)", initial_value=True)
            self.gui_arc_yaw_angle = server.gui.add_slider(
                "Yaw Angle", min=-180, max=180, step=5.0, initial_value=-90,
                hint="Positive=CCW, Negative=CW"
            )
            self.gui_arc_pitch_enabled = server.gui.add_checkbox("Pitch (Vertical Tilt)", initial_value=False)
            self.gui_arc_pitch_angle = server.gui.add_slider(
                "Pitch Angle", min=-180, max=180, step=5.0, initial_value=-10,
                hint="Positive=up, Negative=down"
            )
            self.gui_arc_roll_enabled = server.gui.add_checkbox("Roll (Lateral Roll)", initial_value=False)
            self.gui_arc_roll_angle = server.gui.add_slider(
                "Roll Angle", min=-180, max=180, step=5.0, initial_value=0,
                hint="Positive=right tilt, Negative=left tilt"
            )
            self.gui_arc_num_keyframes = server.gui.add_slider(
                "Keyframes", min=4, max=24, step=1.0, initial_value=8
            )
            self.gui_arc_generate = server.gui.add_button("Generate Trajectory", color="green", icon=viser.Icon.CIRCLES)

        with server.gui.add_folder("Save Trajectory"):
            self.gui_trajectory_filename = server.gui.add_text("Filename", initial_value="my_trajectory.json")
            self.gui_save_trajectory = server.gui.add_button("Save", color="blue", icon=viser.Icon.FILE_EXPORT)

        with server.gui.add_folder("Display Control"):
            self.gui_show_background = server.gui.add_checkbox("Background", True)
            self.gui_show_camera = server.gui.add_checkbox("Original Camera", True)
            self.gui_show_axes = server.gui.add_checkbox("Axes", True)
            if len(self.available_trajectories) > 0:
                server.gui.add_markdown("---\n**Saved Trajectories**")
                for idx, (filename, traj_info) in enumerate(sorted(self.available_trajectories.items())):
                    checkbox = server.gui.add_checkbox(
                        f"{filename}", initial_value=False,
                        hint=f"{traj_info['num_frames']} frames"
                    )
                    traj_info['checkbox'] = checkbox
                    traj_info['color'] = None
                    @checkbox.on_update
                    def _(event, fn=filename):
                        self._update_trajectory_display(fn)
                self.gui_clear_all_trajectories = server.gui.add_button(
                    "Clear All Trajectories", icon=viser.Icon.EYE_OFF
                )
        self.gui_show_original_trajectory = type('obj', (object,), {'value': False})()
        self.gui_rainbow_trajectories = type('obj', (object,), {'value': False})()
        self.gui_stats = None
        self.server = server

    def _create_scene(self, server: viser.ViserServer):
        self.background_handle = None
        if self.background_data:
            self.background_handle = server.scene.add_point_cloud(
                name="/pointcloud/global_background",
                points=self.background_data['points'],
                colors=self.background_data['colors'],
                point_size=self.point_size, point_shape="circle",
            )

        self.foreground_handles = {version: [] for version in FRAME_VERSIONS}

        for version in FRAME_VERSIONS:
            for pc_data in self.foreground_frames[version]:
                frame_id = pc_data['frame_id']
                handle = server.scene.add_point_cloud(
                    name=f"/pointcloud/foreground_{version}_{frame_id}",
                    points=pc_data['points'], colors=pc_data['colors'],
                    point_size=self.foreground_point_size, point_shape="circle",
                )
                self.foreground_handles[version].append(handle)

        self.camera_handles = []
        self.camera_points_handle = None
        if self.camera_data:
            self._create_cameras(server)

        self.axes_handle = None
        self._create_coordinate_axes(server)

        self.bbox_handle = None
        if self.show_bbox:
            for version in DISPLAY_VERSION_ORDER:
                if len(self.foreground_frames[version]) > 0:
                    self._create_bbox(server, self.foreground_frames[version][0]['points'])
                    break

        self._update_display()
        self._bind_events()

    def _create_bbox(self, server: viser.ViserServer, points: np.ndarray):
        bbox_min, bbox_max = np.min(points, axis=0), np.max(points, axis=0)
        vertices = np.array([[bbox_min[0], bbox_min[1], bbox_min[2]], [bbox_max[0], bbox_min[1], bbox_min[2]],
                            [bbox_max[0], bbox_max[1], bbox_min[2]], [bbox_min[0], bbox_max[1], bbox_min[2]],
                            [bbox_min[0], bbox_min[1], bbox_max[2]], [bbox_max[0], bbox_min[1], bbox_max[2]],
                            [bbox_max[0], bbox_max[1], bbox_max[2]], [bbox_min[0], bbox_max[1], bbox_max[2]]])
        edges = [(0,1), (1,2), (2,3), (3,0), (4,5), (5,6), (6,7), (7,4), (0,4), (1,5), (2,6), (3,7)]
        line_points = []
        for v1_idx, v2_idx in edges:
            for t in np.linspace(0, 1, 50):
                line_points.append(vertices[v1_idx] + t * (vertices[v2_idx] - vertices[v1_idx]))
        bbox_points = np.array(line_points)
        bbox_colors = np.tile([255, 255, 0], (len(bbox_points), 1))
        self.bbox_handle = server.scene.add_point_cloud(
            "/reference/frame_0_bbox", points=bbox_points, colors=bbox_colors,
            point_size=self.point_size * 5.0, point_shape="circle")

    def _create_coordinate_axes(self, server: viser.ViserServer):
        all_points = []
        if self.background_data:
            all_points.append(self.background_data['points'])
        for version in FRAME_VERSIONS:
            all_points.extend([fg['points'] for fg in self.foreground_frames[version]])
        axis_scale = 0.5 if not all_points else np.max(np.max(np.concatenate(all_points), axis=0) -
                                                       np.min(np.concatenate(all_points), axis=0)) * 0.1
        origin = np.array([0.0, 0.0, 0.0])
        axis_points, axis_colors = [], []
        for axis_vec, color in [([axis_scale, 0, 0], [255, 0, 0]),
                                ([0, axis_scale, 0], [0, 255, 0]),
                                ([0, 0, axis_scale], [0, 0, 255])]:
            for t in np.linspace(0, 1, 50):
                axis_points.append(origin + t * np.array(axis_vec))
                axis_colors.append(color)
        axis_points.append(origin)
        axis_colors.append([255, 255, 255])
        self.axes_handle = server.scene.add_point_cloud(
            "/reference/coordinate_axes", points=np.array(axis_points), colors=np.array(axis_colors),
            point_size=self.point_size * 3.0, point_shape="circle")

    def _create_cameras(self, server: viser.ViserServer):
        if not self.camera_data:
            return
        camera_color = (196, 78, 82)
        frame_ids_sorted = sorted(self.camera_data.keys())
        camera_positions = []
        for i, frame_id in enumerate(frame_ids_sorted):
            try:
                cam_info = self.camera_data[frame_id]
                wxyz = rotation_matrix_to_wxyz(cam_info['R_world_cam'])
                fov_y = 2 * np.arctan(self.image_height / (2 * cam_info['intrinsic'][1, 1]))
                aspect = self.image_width / self.image_height
                handle = server.scene.add_camera_frustum(
                    f"/camera/original_frame_{frame_id}", fov=fov_y, aspect=aspect, scale=0.018,
                    wxyz=wxyz, position=cam_info['position'], color=camera_color)
                self.camera_handles.append({'handle': handle, 'frame_id': frame_id})
                camera_positions.append(cam_info['position'])
            except Exception:
                pass
        if len(camera_positions) > 0:
            camera_positions = np.array(camera_positions)
            camera_colors = np.tile(camera_color, (len(camera_positions), 1))
            self.camera_points_handle = server.scene.add_point_cloud(
                "/camera/original_positions", points=camera_positions,
                colors=camera_colors, point_size=0.018, point_shape="circle"
            )
            self.original_trajectory_handle = None
            if len(camera_positions) >= 2:
                try:
                    self.original_trajectory_handle = server.scene.add_spline_catmull_rom(
                        "/camera/original_trajectory", positions=camera_positions,
                        color=camera_color, line_width=2.0, segments=len(camera_positions) * 2
                    )
                    self.original_trajectory_handle.visible = False
                except Exception:
                    pass

    def _bind_events(self):
        @self.gui_timestep.on_update
        def _(_) -> None:
            self._update_display()

        @self.gui_play_button.on_click
        def _(_) -> None:
            self.is_playing = True
            self.gui_play_button.visible = False
            self.gui_pause_button.visible = True
            self.gui_timestep.disabled = True

        @self.gui_pause_button.on_click
        def _(_) -> None:
            self.is_playing = False
            self.gui_play_button.visible = True
            self.gui_pause_button.visible = False
            self.gui_timestep.disabled = False

        @self.gui_show_background.on_update
        def _(_) -> None:
            self._update_display()

        @self.gui_show_camera.on_update
        def _(_) -> None:
            self._update_display()

        @self.gui_show_axes.on_update
        def _(_) -> None:
            self._update_display()

        @self.gui_reset_camera.on_click
        def _(event: viser.GuiEvent) -> None:
            self._reset_camera_callback(event)

        @self.gui_view_mode.on_click
        def _(_) -> None:
            if self.is_playing and len(self.keyframes) >= 2:
                self._update_camera_along_trajectory(self.gui_timestep.value)

        @self.gui_save_trajectory.on_click
        def _(event: viser.GuiEvent) -> None:
            self._save_trajectory_callback(event)

        for gui_elem in [self.gui_arc_radius_scale, self.gui_arc_yaw_enabled, self.gui_arc_yaw_angle,
                         self.gui_arc_pitch_enabled, self.gui_arc_pitch_angle,
                         self.gui_arc_roll_enabled, self.gui_arc_roll_angle]:
            @gui_elem.on_update
            def _(_) -> None:
                self._update_arc_preview()

        @self.gui_arc_num_keyframes.on_update
        def _(_) -> None:
            self._update_arc_preview()

        @self.gui_arc_generate.on_click
        def _(event: viser.GuiEvent) -> None:
            self._generate_arc_trajectory_callback(event)

        if len(self.available_trajectories) > 0:
            @self.gui_clear_all_trajectories.on_click
            def _(_) -> None:
                self._clear_all_trajectory_displays()

    def _update_display(self):
        """Update display: show only smooth foreground for current frame."""
        current_timestep = self.gui_timestep.value
        show_background = self.gui_show_background.value
        show_camera = self.gui_show_camera.value
        show_axes = self.gui_show_axes.value

        current_frame_id = None
        if current_timestep < len(self.foreground_frames['smooth']):
            current_frame_id = self.foreground_frames['smooth'][current_timestep]['frame_id']

        if current_frame_id:
            self.gui_frame_id_label.value = current_frame_id

        with self.server.atomic():
            if self.background_handle:
                self.background_handle.visible = show_background
            for handle in self.foreground_handles['single_view']:
                handle.visible = False
            for handle in self.foreground_handles['aligned']:
                handle.visible = False
            for i, handle in enumerate(self.foreground_handles['smooth']):
                handle.visible = (i == current_timestep)
            if self.camera_handles and current_frame_id:
                for cam_info in self.camera_handles:
                    if not show_camera:
                        cam_info['handle'].visible = False
                    else:
                        cam_info['handle'].visible = (cam_info['frame_id'] == current_frame_id)
            if self.camera_points_handle:
                self.camera_points_handle.visible = show_camera
            if self.axes_handle:
                self.axes_handle.visible = show_axes

    def _visualize_keyframe(self, keyframe_id: int, keyframe: Keyframe):
        handle = self.server.scene.add_camera_frustum(
            name=f"/keyframe/keyframe_{keyframe_id}",
            fov=keyframe.fov, aspect=keyframe.aspect, scale=0.05,
            wxyz=keyframe.wxyz, position=keyframe.position, color=(200, 10, 30),
        )
        self.keyframe_handles[keyframe_id] = handle

    def _reset_camera_callback(self, event: viser.GuiEvent):
        if not event.client or self.scene_center is None:
            return
        event.client.camera.look_at = self.scene_center
        event.client.camera.up_direction = np.array([0.0, 0.0, 1.0])

    def _clear_all_keyframes(self, preserve_frame0: bool = False):
        frame0_keyframe = next((kf for kf in self.keyframes.values() if kf.timestep == 0), None)
        remove_scene_handles(list(self.keyframe_handles.values()), clear=False)
        self.keyframes.clear()
        self.keyframe_handles.clear()
        if preserve_frame0 and frame0_keyframe:
            self.keyframe_counter = 0
            self._add_keyframe(frame0_keyframe)
        else:
            self.keyframe_counter = 0
        self._update_stats()
        self._clear_trajectory()

    def _update_stats(self):
        pass

    # ========================================================================
    # Trajectory interpolation and visualization
    # ========================================================================

    def _update_trajectory(self):
        if not HAS_SPLINES:
            return
        if len(self.keyframes) < 2:
            self._clear_trajectory()
            return
        keyframes_sorted = sorted(self.keyframes.values(), key=lambda x: x.timestep)
        positions = np.array([kf.position for kf in keyframes_sorted])
        wxyzs = np.array([kf.wxyz for kf in keyframes_sorted])
        timesteps = np.array([kf.timestep for kf in keyframes_sorted])
        tension = 0.0
        try:
            self.trajectory_spline_position = splines.KochanekBartels(
                positions.tolist(), tcb=(tension, 0.0, 0.0), endconditions="natural")
            quaternions = [splines.quaternion.UnitQuaternion.from_unit_xyzw(np.roll(wxyz, -1)) for wxyz in wxyzs]
            self.trajectory_spline_orientation = splines.quaternion.KochanekBartels(
                quaternions, tcb=(tension, 0.0, 0.0), endconditions="natural")
            self.trajectory_spline_timestep = splines.KochanekBartels(
                timesteps.tolist(), tcb=(tension, 0.0, 0.0), endconditions="natural")
            self._visualize_trajectory()
        except Exception:
            pass

    def _visualize_trajectory(self):
        if not self.trajectory_spline_position:
            return
        self._clear_trajectory_visualization()
        num_samples = 100
        t_values = np.linspace(0, len(self.keyframes) - 1, num_samples)
        trajectory_positions = self.trajectory_spline_position.evaluate(t_values)
        colors_array = np.array([colorsys.hls_to_rgb(h, 0.5, 1.0)
                                for h in np.linspace(0.0, 1.0, len(trajectory_positions))])
        try:
            self.trajectory_handles.append(self.server.scene.add_spline_catmull_rom(
                "/trajectory/path", positions=trajectory_positions,
                color=(220, 220, 220), line_width=0.5, segments=num_samples + 1))
        except Exception:
            pass
        self.trajectory_handles.append(self.server.scene.add_point_cloud(
            "/trajectory/points", points=trajectory_positions, colors=colors_array,
            point_size=0.02, point_shape="circle"))

    def _clear_trajectory(self):
        self.trajectory_spline_position = None
        self.trajectory_spline_orientation = None
        self.trajectory_spline_timestep = None
        self._clear_trajectory_visualization()
        if self.trajectory_camera_handle:
            remove_scene_handles([self.trajectory_camera_handle], clear=False)
            self.trajectory_camera_handle = None

    def _clear_trajectory_visualization(self):
        remove_scene_handles(self.trajectory_handles)

    def _save_trajectory_callback(self, event: viser.GuiEvent):
        if len(self.keyframes) < 2:
            show_modal(event.client, "Error", "Need at least 2 keyframes to save trajectory")
            return
        if self.trajectory_spline_position is None or self.trajectory_spline_timestep is None:
            show_modal(event.client, "Error", "Trajectory not generated, please add keyframes first")
            return
        try:
            filename = normalized_json_filename(self.gui_trajectory_filename.value)
            output_path = os.path.join(self.data_dir, filename)
            file_exists = os.path.exists(output_path)
            action = "overwritten" if file_exists else "created"
            self._save_trajectory_to_json()
            show_modal(
                event.client,
                "Success",
                f"Trajectory {action}\n\n"
                f"**File**: `{filename}`\n"
                f"**Path**: `{self.data_dir}`",
            )
        except Exception as e:
            print(f"Failed to save trajectory: {e}")
            import traceback
            traceback.print_exc()
            show_modal(event.client, "Error", f"Save failed:\n`{str(e)}`")

    def _save_trajectory_to_json(self):
        """Save camera trajectory to JSON (transform back to y-down coordinate system)."""
        keyframes_sorted = sorted(self.keyframes.values(), key=lambda x: x.timestep)
        min_timestep = keyframes_sorted[0].timestep
        max_timestep = keyframes_sorted[-1].timestep

        if self.intrinsics_frame0 is None:
            raise ValueError("Frame 0 intrinsics not found")

        intrinsics_original = self.intrinsics_frame0
        original_frames = int(max_timestep - min_timestep + 1)

        trajectory_data = {
            "keyframes": [],
            "camera_path": [],
            "metadata": {
                "image_size": [self.image_height, self.image_width],
                "num_keyframes": len(self.keyframes),
                "timestep_range": [int(min_timestep), int(max_timestep)],
                "coordinate_system": "y-down (original)",
                "total_output_frames": original_frames,
            }
        }

        for keyframe_id, keyframe in sorted(self.keyframes.items(), key=lambda x: x[1].timestep):
            position_original = transform_from_z_up(keyframe.position.reshape(1, 3)).flatten()
            R_world_cam_zup = Rotation.from_quat(keyframe.wxyz[[1, 2, 3, 0]]).as_matrix()
            R_world_cam_original = transform_rotation_from_z_up(R_world_cam_zup)
            T_world_cam = np.eye(4)
            T_world_cam[:3, :3] = R_world_cam_original
            T_world_cam[:3, 3] = position_original
            T_cam_world = np.linalg.inv(T_world_cam)
            extrinsic_3x4 = T_cam_world[:3, :]
            trajectory_data["keyframes"].append({
                "timestep": int(keyframe.timestep),
                "position": position_original.tolist(),
                "wxyz": keyframe.wxyz.tolist(),
                "extrinsic": extrinsic_3x4.tolist(),
                "intrinsic": intrinsics_original.tolist(),
                "fov": float(keyframe.fov),
                "aspect": float(keyframe.aspect)
            })

        def zup_to_extrinsic(pos_zup, wxyz_zup):
            pos_original = transform_from_z_up(np.array(pos_zup).reshape(1, 3)).flatten()
            R_zup = Rotation.from_quat(np.array(wxyz_zup)[[1, 2, 3, 0]]).as_matrix()
            R_original = transform_rotation_from_z_up(R_zup)
            T_world_cam = np.eye(4)
            T_world_cam[:3, :3] = R_original
            T_world_cam[:3, 3] = pos_original
            T_cam_world = np.linalg.inv(T_world_cam)
            return T_cam_world[:3, :], pos_original

        num_samples = 100
        t_values = np.linspace(0, len(self.keyframes) - 1, num_samples)
        timestep_samples = self.trajectory_spline_timestep.evaluate(t_values)

        for output_frame_idx, video_time in enumerate(range(int(min_timestep), int(max_timestep) + 1)):
            idx = np.argmin(np.abs(timestep_samples - video_time))
            t = t_values[idx]
            position_zup = self.trajectory_spline_position.evaluate(t)
            quat = self.trajectory_spline_orientation.evaluate(t)
            wxyz_zup = np.array([quat.scalar, *quat.vector])
            extrinsic_3x4, position_original = zup_to_extrinsic(position_zup, wxyz_zup)
            trajectory_data["camera_path"].append({
                "output_frame": output_frame_idx,
                "video_time": video_time,
                "extrinsic": extrinsic_3x4.tolist(),
                "intrinsic": intrinsics_original.tolist(),
                "position": position_original.tolist(),
                "wxyz": wxyz_zup.tolist()
            })

        filename = normalized_json_filename(self.gui_trajectory_filename.value)
        output_path = os.path.join(self.data_dir, filename)
        os.makedirs(self.data_dir, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(trajectory_data, f, indent=2)
        print(f"Trajectory saved: {output_path} ({len(trajectory_data['camera_path'])} frames)")

    def _update_camera_along_trajectory(self, current_timestep: int):
        """Update camera position along interpolated trajectory."""
        if self.trajectory_spline_position is None or self.trajectory_spline_timestep is None:
            return
        if len(self.keyframes) < 2:
            return
        try:
            keyframes_sorted = sorted(self.keyframes.values(), key=lambda x: x.timestep)
            min_timestep = keyframes_sorted[0].timestep
            max_timestep = keyframes_sorted[-1].timestep

            if min_timestep <= current_timestep <= max_timestep:
                num_samples = 100
                t_values = np.linspace(0, len(self.keyframes) - 1, num_samples)
                timestep_samples = self.trajectory_spline_timestep.evaluate(t_values)
                idx = np.argmin(np.abs(timestep_samples - current_timestep))
                t = t_values[idx]
                position = self.trajectory_spline_position.evaluate(t)
                quat = self.trajectory_spline_orientation.evaluate(t)
                wxyz = np.array([quat.scalar, *quat.vector])
                view_mode = self.gui_view_mode.value

                if view_mode == "First Person":
                    fov = self.unified_fov if self.unified_fov is not None else np.deg2rad(60.0)
                    for client in self.server.get_clients().values():
                        with client.atomic():
                            client.camera.position = position
                            client.camera.wxyz = wxyz
                            client.camera.fov = fov
                    if self.trajectory_camera_handle is not None:
                        self.trajectory_camera_handle.visible = False
                else:
                    if self.trajectory_camera_handle is None:
                        fov = self.unified_fov if self.unified_fov is not None else np.deg2rad(60.0)
                        aspect = self.unified_aspect if self.unified_aspect is not None else (self.image_width / self.image_height)
                        self.trajectory_camera_handle = self.server.scene.add_camera_frustum(
                            name="/trajectory/current_camera",
                            fov=fov, aspect=aspect, scale=0.05,
                            wxyz=wxyz, position=position, color=(10, 200, 30), visible=True
                        )
                    else:
                        self.trajectory_camera_handle.position = position
                        self.trajectory_camera_handle.wxyz = wxyz
                        self.trajectory_camera_handle.visible = True
        except Exception:
            pass

    def run(self):
        print(f"Camera Trajectory Editor")
        print(f"  URL: http://localhost:{self.port}")
        print(f"  Data: {self.data_dir}")
        if self.background_data:
            print(f"  Background: {len(self.background_data['points']):,} points")
        print(f"  Foreground: {self.num_frames} frames")
        print(f"  Coordinate system: Z-up (X=right/red, Y=forward/green, Z=up/blue)")

        prev_timestep = self.gui_timestep.value
        while True:
            if self.is_playing and self.num_frames > 0:
                next_timestep = (self.gui_timestep.value + 1) % self.num_frames
                self.gui_timestep.value = next_timestep
                if len(self.keyframes) >= 2 and self.trajectory_spline_position:
                    self._update_camera_along_trajectory(next_timestep)
            if self.gui_timestep.value != prev_timestep:
                self._update_display()
                prev_timestep = self.gui_timestep.value
            time.sleep(1.0 / self.gui_framerate.value)

    # ========================================================================
    # Arc trajectory generator
    # ========================================================================

    def _compute_arc_parameters(self):
        """Compute arc parameters for all rotation types: Yaw, Pitch, Roll."""
        first_frame = None
        for version in DISPLAY_VERSION_ORDER:
            if len(self.foreground_frames[version]) > 0:
                first_frame = self.foreground_frames[version][0]
                break
        if first_frame is None:
            return

        points = first_frame['points']
        points = points[np.isfinite(points).all(axis=1)]
        if len(points) == 0:
            return

        depth_min = np.min(points[:, 1])
        if not np.isfinite(depth_min):
            return
        start_point = np.array([0.0, 0.0, 0.0])

        for arc_type in ("yaw", "pitch", "roll"):
            center = np.array([0.0, depth_min, 0.0])
            radius = max(abs(depth_min), 1.0e-8)
            vec_to_start = start_point - center
            if arc_type == 'yaw':
                start_angle = np.arctan2(vec_to_start[1], vec_to_start[0])
            elif arc_type == 'pitch':
                start_angle = np.arctan2(vec_to_start[2], vec_to_start[1])
            else:
                start_angle = np.arctan2(vec_to_start[2], vec_to_start[0])
            self.arc_params[arc_type] = {
                'center': center, 'base_radius': radius, 'start_angle': start_angle
            }

    def _has_arc_parameters(self) -> bool:
        return 'yaw' in self.arc_params and self.arc_params['yaw']['center'] is not None

    def _active_arc_settings(self) -> Optional[ArcSettings]:
        if not self._has_arc_parameters():
            return None
        arc_center = self.arc_params['yaw']['center']
        arc_base_radius = safe_float(self.arc_params['yaw']['base_radius'], 1.0, min_value=1.0e-8)
        radius_scale = safe_float(self.gui_arc_radius_scale.value, 1.0, min_value=1.0e-6)
        return ArcSettings(
            center=arc_center,
            radius=arc_base_radius * radius_scale,
            num_keyframes=safe_int(self.gui_arc_num_keyframes.value, 8, min_value=2),
            yaw_angle_deg=safe_float(self.gui_arc_yaw_angle.value, 0.0) if self.gui_arc_yaw_enabled.value else 0.0,
            pitch_angle_deg=safe_float(self.gui_arc_pitch_angle.value, 0.0) if self.gui_arc_pitch_enabled.value else 0.0,
            roll_angle_deg=safe_float(self.gui_arc_roll_angle.value, 0.0) if self.gui_arc_roll_enabled.value else 0.0,
        )

    def _generate_arc_trajectory_callback(self, event: viser.GuiEvent):
        settings = self._active_arc_settings()
        if settings is None:
            show_modal(event.client, "Error", "**Arc parameters not computed**\n\nMake sure foreground point cloud is loaded")
            return

        if not (self.gui_arc_yaw_enabled.value or self.gui_arc_pitch_enabled.value or self.gui_arc_roll_enabled.value):
            show_modal(event.client, "Info", "**Please enable at least one rotation axis**\n\nCheck Yaw / Pitch / Roll")
            return

        keyframes_data = self._generate_composite_arc_keyframes(
            settings.center,
            settings.radius,
            settings.yaw_angle_deg,
            settings.pitch_angle_deg,
            settings.roll_angle_deg,
            settings.num_keyframes,
        )
        if not keyframes_data:
            return

        self._clear_all_keyframes()

        for kf_data in keyframes_data:
            self._add_keyframe(self._make_keyframe(
                timestep=kf_data['timestep'],
                position=kf_data['position'],
                wxyz=kf_data['wxyz'],
            ))

        self._update_stats()
        self._update_trajectory()
        self._clear_arc_preview()

        enabled = []
        if self.gui_arc_yaw_enabled.value:
            enabled.append(f"Yaw:{settings.yaw_angle_deg:.0f}")
        if self.gui_arc_pitch_enabled.value:
            enabled.append(f"Pitch:{settings.pitch_angle_deg:.0f}")
        if self.gui_arc_roll_enabled.value:
            enabled.append(f"Roll:{settings.roll_angle_deg:.0f}")
        show_modal(
            event.client,
            "Success",
            f"Trajectory generated\n\n**Type**: {' + '.join(enabled)}\n**Keyframes**: {settings.num_keyframes}",
        )

    def _generate_composite_arc_keyframes(self, arc_center: np.ndarray, arc_radius: float,
                                          yaw_angle_deg: float, pitch_angle_deg: float, roll_angle_deg: float,
                                          num_keyframes: int) -> List[Dict]:
        """Generate composite rotation arc keyframes.

        Radius scaling is achieved by adjusting center position; start point stays at origin.
        """
        keyframes_data = []

        reference_R = None
        if self.camera_data and '00000' in self.camera_data:
            reference_R = self.camera_data['00000']['R_world_cam']

        start_point = np.array([0.0, 0.0, 0.0])
        arc_center = np.asarray(arc_center, dtype=np.float64)
        center_norm = np.linalg.norm(arc_center)
        if arc_center.shape != (3,) or not np.isfinite(arc_center).all() or center_norm < 1.0e-8:
            arc_center = np.array([0.0, 1.0, 0.0])
            center_norm = 1.0
        arc_radius = safe_float(arc_radius, center_norm, min_value=1.0e-8)
        yaw_angle_deg = safe_float(yaw_angle_deg, 0.0)
        pitch_angle_deg = safe_float(pitch_angle_deg, 0.0)
        roll_angle_deg = safe_float(roll_angle_deg, 0.0)

        base_direction = arc_center / center_norm
        adjusted_center = base_direction * arc_radius
        initial_vec = start_point - adjusted_center
        fallback_wxyz = rotation_matrix_to_wxyz(reference_R) if reference_R is not None else None

        for i in range(num_keyframes):
            t = i / max(num_keyframes - 1, 1)

            yaw_current = np.deg2rad(yaw_angle_deg * t)
            pitch_current = np.deg2rad(pitch_angle_deg * t)
            roll_current = np.deg2rad(roll_angle_deg * t)

            # Yaw: rotate around Z-axis (XY plane)
            cos_yaw, sin_yaw = np.cos(yaw_current), np.sin(yaw_current)
            R_yaw = np.array([[cos_yaw, -sin_yaw, 0], [sin_yaw, cos_yaw, 0], [0, 0, 1]])

            # Pitch: rotate around X-axis (YZ plane)
            cos_pitch, sin_pitch = np.cos(pitch_current), np.sin(pitch_current)
            R_pitch = np.array([[1, 0, 0], [0, cos_pitch, -sin_pitch], [0, sin_pitch, cos_pitch]])

            # Roll: rotate around Y-axis (XZ plane)
            cos_roll, sin_roll = np.cos(roll_current), np.sin(roll_current)
            R_roll = np.array([[cos_roll, 0, sin_roll], [0, 1, 0], [-sin_roll, 0, cos_roll]])

            R_composite = R_yaw @ R_pitch @ R_roll
            rotated_vec = R_composite @ initial_vec
            position = adjusted_center + rotated_vec

            if reference_R is not None:
                R_world_cam = R_composite @ reference_R
            else:
                forward = adjusted_center - position
                forward = forward / np.linalg.norm(forward)
                world_up = np.array([0.0, 0.0, 1.0])
                right = np.cross(forward, world_up)
                if np.linalg.norm(right) < 0.01:
                    right = np.cross(forward, np.array([0.0, 1.0, 0.0]))
                right = right / np.linalg.norm(right)
                up = np.cross(right, forward)
                up = up / np.linalg.norm(up)
                R_world_cam = np.column_stack([right, up, -forward])

            wxyz = rotation_matrix_to_wxyz(R_world_cam, fallback=fallback_wxyz)

            if self.num_frames > 0:
                timestep = int(i * (self.num_frames - 1) / max(num_keyframes - 1, 1))
            else:
                timestep = i

            keyframes_data.append({'position': position, 'wxyz': wxyz, 'timestep': timestep})

        return keyframes_data

    def _update_arc_preview(self):
        self._clear_arc_preview()
        settings = self._active_arc_settings()
        if settings is None:
            return

        keyframes_data = self._generate_composite_arc_keyframes(
            settings.center,
            settings.radius,
            settings.yaw_angle_deg,
            settings.pitch_angle_deg,
            settings.roll_angle_deg,
            settings.num_keyframes,
        )
        if not keyframes_data:
            return

        preview_data = self._generate_composite_arc_keyframes(
            settings.center,
            settings.radius,
            settings.yaw_angle_deg,
            settings.pitch_angle_deg,
            settings.roll_angle_deg,
            100,
        )
        preview_points = np.array([kf['position'] for kf in preview_data])

        preview_colors = np.tile([255, 255, 0], (len(preview_points), 1))
        self.arc_preview_handles.append(
            self.server.scene.add_point_cloud(
                "/arc_preview/path", points=preview_points, colors=preview_colors,
                point_size=0.015, point_shape="circle"
            )
        )

        keyframe_positions = np.array([kf['position'] for kf in keyframes_data])
        keyframe_colors = np.tile([255, 165, 0], (len(keyframe_positions), 1))
        self.arc_preview_handles.append(
            self.server.scene.add_point_cloud(
                "/arc_preview/keyframes", points=keyframe_positions, colors=keyframe_colors,
                point_size=0.03, point_shape="circle"
            )
        )

        fov, aspect = self._camera_fov_aspect()
        for idx, kf_data in enumerate(keyframes_data):
            camera_handle = self.server.scene.add_camera_frustum(
                name=f"/arc_preview/camera_{idx}", fov=fov, aspect=aspect, scale=0.04,
                wxyz=kf_data['wxyz'], position=kf_data['position'], color=(100, 200, 100),
            )
            self.arc_preview_handles.append(camera_handle)

        self.arc_preview_handles.append(
            self.server.scene.add_point_cloud(
                "/arc_preview/center", points=settings.center.reshape(1, 3),
                colors=np.array([[255, 0, 255]]), point_size=0.04, point_shape="circle"
            )
        )

        origin = np.array([[0.0, 0.0, 0.0]])
        self.arc_preview_handles.append(
            self.server.scene.add_point_cloud(
                "/arc_preview/origin", points=origin,
                colors=np.array([[0, 255, 255]]), point_size=0.04, point_shape="circle"
            )
        )

    def _clear_arc_preview(self):
        remove_scene_handles(self.arc_preview_handles)

    # ========================================================================
    # Multi-trajectory display
    # ========================================================================

    def _update_trajectory_display(self, filename: str):
        if filename not in self.available_trajectories:
            return
        traj_info = self.available_trajectories[filename]
        is_visible = traj_info['checkbox'].value
        if is_visible:
            if traj_info['color'] is None:
                color_idx = len(self.trajectory_color_order) % len(self.trajectory_colors)
                traj_info['color'] = self.trajectory_colors[color_idx]
                self.trajectory_color_order.append(filename)
            if len(traj_info['handles']) == 0:
                self._visualize_saved_trajectory(filename)
            else:
                set_handles_visible(traj_info['handles'], True)
        else:
            set_handles_visible(traj_info['handles'], False)

    def _visualize_saved_trajectory(self, filename: str):
        if filename not in self.available_trajectories:
            return
        traj_info = self.available_trajectories[filename]
        traj_data = traj_info['data']
        color = traj_info['color']

        try:
            remove_scene_handles(traj_info['handles'])

            if 'camera_path' in traj_data and len(traj_data['camera_path']) > 0:
                camera_path = traj_data['camera_path']
            elif 'keyframes' in traj_data and len(traj_data['keyframes']) > 0:
                camera_path = traj_data['keyframes']
            else:
                return

            positions_original = []
            rotations_data = []
            for frame in camera_path:
                pos = None
                wxyz = None
                if 'position' in frame:
                    pos = np.array(frame['position'])
                if 'extrinsic' in frame:
                    extrinsic = np.array(frame['extrinsic'])
                    T_cam_world = np.vstack([extrinsic, [0, 0, 0, 1]])
                    T_world_cam = np.linalg.inv(T_cam_world)
                    R_world_cam = T_world_cam[:3, :3]
                    if pos is None:
                        pos = T_world_cam[:3, 3]
                    R_world_cam_zup = transform_rotation_to_z_up(R_world_cam)
                    wxyz = rotation_matrix_to_wxyz(R_world_cam_zup)
                elif 'wxyz' in frame:
                    wxyz = np.array(frame['wxyz'])
                if pos is not None:
                    positions_original.append(pos)
                    rotations_data.append(wxyz)

            if len(positions_original) == 0:
                return

            positions_original = np.array(positions_original)
            positions_zup = transform_to_z_up(positions_original)

            try:
                use_rainbow = hasattr(self, 'gui_rainbow_trajectories') and self.gui_rainbow_trajectories.value
                line_color = (200, 200, 200) if use_rainbow else color
                handle = self.server.scene.add_spline_catmull_rom(
                    f"/saved_trajectory/{filename}/path", positions=positions_zup,
                    color=line_color, line_width=2.0, segments=len(positions_zup) * 2
                )
                traj_info['handles'].append(handle)
            except Exception:
                pass

            positions_zup_skip_first = positions_zup[1:] if len(positions_zup) > 1 else positions_zup
            use_rainbow = hasattr(self, 'gui_rainbow_trajectories') and self.gui_rainbow_trajectories.value
            if use_rainbow:
                point_colors = np.array([colorsys.hls_to_rgb(h, 0.5, 1.0)
                                        for h in np.linspace(0.0, 1.0, len(positions_zup_skip_first))])
            else:
                point_colors = np.tile(color, (len(positions_zup_skip_first), 1))

            handle = self.server.scene.add_point_cloud(
                f"/saved_trajectory/{filename}/points", points=positions_zup_skip_first,
                colors=point_colors, point_size=0.01, point_shape="circle"
            )
            traj_info['handles'].append(handle)

            fov = self.unified_fov if self.unified_fov is not None else np.deg2rad(60.0)
            aspect = self.unified_aspect if self.unified_aspect is not None else (self.image_width / self.image_height)

            n_frames = len(positions_zup)
            if n_frames >= 3:
                display_indices = [0, n_frames // 2, n_frames - 1]
            elif n_frames == 2:
                display_indices = [0, 1]
            else:
                display_indices = [0] if n_frames > 0 else []

            for i, idx in enumerate(display_indices):
                pos = positions_zup[idx]
                wxyz = rotations_data[idx] if idx < len(rotations_data) and rotations_data[idx] is not None else None
                if wxyz is not None:
                    use_rainbow = hasattr(self, 'gui_rainbow_trajectories') and self.gui_rainbow_trajectories.value
                    if use_rainbow:
                        hue = idx / max(n_frames - 1, 1)
                        rgb = colorsys.hls_to_rgb(hue, 0.5, 1.0)
                        camera_color = tuple(int(c * 255) for c in rgb)
                    else:
                        camera_color = color
                    camera_handle = self.server.scene.add_camera_frustum(
                        name=f"/saved_trajectory/{filename}/camera_{i}",
                        fov=fov, aspect=aspect, scale=0.035,
                        wxyz=wxyz, position=pos, color=camera_color,
                    )
                    traj_info['handles'].append(camera_handle)

        except Exception as e:
            print(f"Failed to load trajectory {filename}: {e}")
            import traceback
            traceback.print_exc()

    def _clear_all_trajectory_displays(self):
        for filename, traj_info in self.available_trajectories.items():
            if traj_info['checkbox'] is not None:
                traj_info['checkbox'].value = False
            traj_info['color'] = None
            set_handles_visible(traj_info['handles'], False)
        self.trajectory_color_order.clear()

    def _refresh_all_trajectory_displays(self):
        for filename, traj_info in self.available_trajectories.items():
            if traj_info['checkbox'] is not None and traj_info['checkbox'].value:
                remove_scene_handles(traj_info['handles'])
                self._visualize_saved_trajectory(filename)


def main():
    args = parse_args()
    if not os.path.exists(args.data_dir):
        print(f"Error: Data directory not found: {args.data_dir}")
        return 1
    if not HAS_SPLINES:
        print(f"Error: splines library not installed. Run: pip install splines")
        return 1
    try:
        viewer = CameraTrajectoryEditor(
            data_dir=args.data_dir, port=args.port, point_size=args.point_size,
            foreground_point_size=args.foreground_point_size,
            fps=args.fps, subsample=args.subsample, num_frames=args.num_frames, show_bbox=args.show_bbox
        )
        viewer.run()
        return 0
    except ValueError as e:
        print(f"Error: {e}")
        return 1
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        return 0
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
