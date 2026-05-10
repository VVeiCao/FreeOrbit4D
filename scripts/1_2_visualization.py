#!/usr/bin/env python3
"""Smoothed point cloud visualization with Viser (global background + foreground)."""

import os
import sys
import time
import argparse
import json
from typing import List, Dict, Optional
import numpy as np
import trimesh
import viser
from scipy.spatial.transform import Rotation
import matplotlib


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def transform_to_z_up(points: np.ndarray) -> np.ndarray:
    """y-down -> z-up: (x, y, z) -> (x, z, -y)."""
    points_flat = points.reshape(-1, 3)
    return np.column_stack([points_flat[:, 0], points_flat[:, 2], -points_flat[:, 1]]).reshape(points.shape)


def transform_rotation_to_z_up(R: np.ndarray) -> np.ndarray:
    return np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float64) @ R


def rotation_matrix_to_wxyz(R: np.ndarray) -> np.ndarray:
    R = np.asarray(R, dtype=np.float64)
    if R.shape != (3, 3) or not np.isfinite(R).all():
        return np.array([1.0, 0.0, 0.0, 0.0])
    try:
        U, _, Vt = np.linalg.svd(R)
        R_orth = U @ Vt
        if np.linalg.det(R_orth) < 0:
            U[:, -1] *= -1
            R_orth = U @ Vt
        return Rotation.from_matrix(R_orth).as_quat()[[3, 0, 1, 2]]
    except (ValueError, np.linalg.LinAlgError):
        return np.array([1.0, 0.0, 0.0, 0.0])


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
    parser = argparse.ArgumentParser(description="Step 1.2: Smoothed point cloud visualization")
    parser.add_argument("--config", type=str, default=None, help="Scene config; uses project.output_prepared as data_dir")
    parser.add_argument("--data_dir", type=str, default=None, help="Data directory")
    parser.add_argument("--port", type=int, default=8080, help="Viser port")
    parser.add_argument("--point_size", type=float, default=0.001, help="Point size")
    parser.add_argument("--fps", type=float, default=5.0, help="Playback frame rate")
    parser.add_argument("--subsample", type=int, default=4, help="Subsample rate")
    parser.add_argument("--num_frames", type=int, default=None, help="Number of frames to load")
    parser.add_argument("--show_bbox", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.config and args.data_dir is None:
        if PROJECT_ROOT not in sys.path:
            sys.path.insert(0, PROJECT_ROOT)
        from utils.config import Config
        args.data_dir = Config(args.config).get("project.output_prepared")

    if args.data_dir is None:
        args.data_dir = "outputs/prepared/camel"

    return args


class SmoothedPointcloudViewer:
    """Viewer for smoothed 5-view foreground point clouds in background space."""

    def __init__(
        self,
        data_dir: str,
        port: int = 8080,
        point_size: float = 0.003,
        fps: float = 5.0,
        subsample: int = 1,
        num_frames: int = None,
        show_bbox: bool = False
    ):
        self.data_dir = data_dir
        self.port = port
        self.point_size = point_size
        self.fps = fps
        self.subsample = subsample
        self.num_frames_limit = num_frames
        self.image_width, self.image_height = 640, 480
        self.scene_center = None
        self.initial_camera_position = None
        self.is_playing = False

        self.background_data = self._load_global_background()
        self.camera_data = self._load_camera_parameters()
        self.foreground_frames = self._load_smoothed_foreground_frames()
        self.num_frames = len(self.foreground_frames)

        if self.num_frames == 0 and self.background_data is None:
            raise ValueError("No valid point cloud files found")

        self.server = self._setup_viser()

    def _load_global_background(self) -> Optional[Dict]:
        """Load global background point cloud."""
        bg_path = os.path.join(self.data_dir, "global_background.ply")
        if not os.path.exists(bg_path):
            return None

        try:
            points, colors = load_ply_points(bg_path, default_color=[150, 150, 150], subsample=self.subsample)
            return {'points': points, 'colors': colors}
        except Exception as e:
            print(f"Error loading background: {e}")
            return None

    def _load_camera_parameters(self) -> Optional[Dict]:
        """Load global camera parameters."""
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
                    'R_world_cam': R_new,
                }
            return camera_data_processed
        except Exception as e:
            print(f"Error loading camera parameters: {e}")
            return None

    def _load_smoothed_foreground_frames(self) -> List[Dict]:
        """Load per-frame smoothed foreground point clouds."""
        subdirs = sorted([d for d in os.listdir(self.data_dir)
                         if os.path.isdir(os.path.join(self.data_dir, d)) and d.isdigit()])

        if self.num_frames_limit:
            subdirs = subdirs[:self.num_frames_limit]

        foreground_list = []
        num_with_smooth = 0

        for frame_id in subdirs:
            fg_path = os.path.join(self.data_dir, frame_id, 'pointcloud', f"{frame_id}_foreground_5_views_aligned_smooth.ply")
            if not os.path.exists(fg_path):
                continue

            try:
                points, colors = load_ply_points(fg_path, default_color=[0, 255, 0], subsample=self.subsample)
                frame_data = {
                    'frame_id': frame_id,
                    'points': points,
                    'colors': colors,
                }
                num_with_smooth += 1

                foreground_list.append(frame_data)

            except Exception as e:
                print(f"Error loading frame {frame_id}: {e}")

        print(f"Loaded {len(foreground_list)} frames, {num_with_smooth} with smoothed foreground")
        self.has_smooth_data = num_with_smooth > 0

        return foreground_list

    def _get_display_foreground(self, pc_data: Dict):
        """Return the foreground point cloud used by the viewer."""
        return pc_data['points'], pc_data['colors'], "Smoothed"

    def _setup_viser(self) -> viser.ViserServer:
        """Set up Viser server."""
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
        for fg_frame in self.foreground_frames:
            display_points, _, _ = self._get_display_foreground(fg_frame)
            all_points.append(display_points)

        if not all_points:
            return

        all_points_combined = np.concatenate(all_points)
        bbox_min, bbox_max = np.min(all_points_combined, axis=0), np.max(all_points_combined, axis=0)
        self.scene_center = (bbox_min + bbox_max) / 2.0
        max_extent = np.max(bbox_max - bbox_min)
        initial_position = (
            self.camera_data['00000']['position']
            if self.camera_data and '00000' in self.camera_data
            else self.scene_center + np.array([0, -max_extent * 1.5, max_extent * 0.5])
        )
        self.initial_camera_position = initial_position

        @server.on_client_connect
        def _(client: viser.ClientHandle) -> None:
            client.camera.position = initial_position
            client.camera.look_at = self.scene_center
            client.camera.up_direction = np.array([0.0, 0.0, 1.0])

    def _create_gui(self, server: viser.ViserServer):
        """Create GUI controls."""
        with server.gui.add_folder("Time Control"):
            self.gui_timestep = server.gui.add_slider(
                "Frame",
                min=0,
                max=max(self.num_frames - 1, 0),
                step=1,
                initial_value=0,
            )

            self.gui_frame_id_label = server.gui.add_text(
                "Frame ID",
                initial_value=self.foreground_frames[0]['frame_id'] if self.foreground_frames else "N/A",
                disabled=True,
            )

            self.gui_play_button = server.gui.add_button("Play", icon=viser.Icon.PLAYER_PLAY)
            self.gui_pause_button = server.gui.add_button("Pause", icon=viser.Icon.PLAYER_PAUSE, visible=False)
            self.gui_framerate = server.gui.add_slider("FPS", min=1, max=30, step=0.5, initial_value=self.fps)
            self.gui_show_mode = server.gui.add_button_group(
                "Display Mode",
                ("Current Frame", "All Frames"),
                hint="Current Frame: show one timestep | All Frames: show the whole sequence",
            )
            self.gui_show_mode.value = "Current Frame"

        self.gui_reset_camera = server.gui.add_button("Reset View", icon=viser.Icon.VIEWFINDER)

        with server.gui.add_folder("Display Control"):
            self.gui_show_background = server.gui.add_checkbox("Background", True)
            self.gui_show_foreground = server.gui.add_checkbox("Foreground", True)
            self.gui_show_camera = server.gui.add_checkbox("Original Camera", True)
            self.gui_show_axes = server.gui.add_checkbox("Axes", True)

            server.gui.add_text("Foreground Source", initial_value="Smoothed", disabled=True)

            server.gui.add_text(
                "Coordinate System",
                initial_value="Z-up: X=right/red, Y=forward/green, Z=up/blue",
                disabled=True,
            )

            self.gui_point_size = server.gui.add_slider(
                "Point Size",
                min=0.001,
                max=0.01,
                step=0.0001,
                initial_value=self.point_size
            )

        with server.gui.add_folder("Statistics"):
            bg_points = len(self.background_data['points']) if self.background_data else 0
            fg_points = sum(len(self._get_display_foreground(pc)[0]) for pc in self.foreground_frames)
            num_with_smooth = len(self.foreground_frames)

            info_text = f"Script: Step 1.2 (smoothed foreground in background space)\n"
            info_text += f"Background points: {bg_points:,}\n"
            info_text += f"Foreground frames: {self.num_frames}\n"
            if self.has_smooth_data:
                info_text += f"Smoothed frames: {num_with_smooth}\n"
            info_text += f"Foreground total points: {fg_points:,}\n"
            info_text += f"Total points: {bg_points + fg_points:,}"

            self.gui_stats = server.gui.add_text(
                "Stats",
                initial_value=info_text,
                disabled=True,
            )

        self.server = server

    def _create_scene(self, server: viser.ViserServer):
        """Create 3D scene."""
        self.background_handle = None
        if self.background_data:
            self.background_handle = server.scene.add_point_cloud(
                name="/pointcloud/global_background",
                points=self.background_data['points'],
                colors=self.background_data['colors'],
                point_size=self.point_size,
                point_shape="circle",
            )

        self.foreground_handles = []

        for pc_data in self.foreground_frames:
            frame_id = pc_data['frame_id']
            points, colors, source = self._get_display_foreground(pc_data)

            handle = server.scene.add_point_cloud(
                name=f"/pointcloud/foreground_{source.lower().replace(' ', '_')}_{frame_id}",
                points=points,
                colors=colors,
                point_size=self.point_size,
                point_shape="circle",
            )
            self.foreground_handles.append(handle)

        self.camera_handles = []
        if self.camera_data:
            self._create_cameras(server)

        self.axes_handle = None
        self._create_coordinate_axes(server)

        self._update_display()
        self._bind_events()

    def _create_coordinate_axes(self, server: viser.ViserServer):
        """Create Z-up coordinate axes visualization."""
        all_points = []
        if self.background_data:
            all_points.append(self.background_data['points'])
        for fg_frame in self.foreground_frames:
            display_points, _, _ = self._get_display_foreground(fg_frame)
            all_points.append(display_points)

        if len(all_points) == 0:
            axis_scale = 0.5
        else:
            all_points_combined = np.concatenate(all_points, axis=0)
            bbox_min = np.min(all_points_combined, axis=0)
            bbox_max = np.max(all_points_combined, axis=0)
            bbox_size = bbox_max - bbox_min
            max_size = np.max(bbox_size)
            axis_scale = max_size * 0.1

        origin = np.array([0.0, 0.0, 0.0])

        num_points_per_axis = 50
        axis_points = []
        axis_colors = []

        # X-axis: red (right)
        for t in np.linspace(0, 1, num_points_per_axis):
            point = origin + t * np.array([axis_scale, 0, 0])
            axis_points.append(point)
            axis_colors.append([255, 0, 0])

        # Y-axis: green (forward)
        for t in np.linspace(0, 1, num_points_per_axis):
            point = origin + t * np.array([0, axis_scale, 0])
            axis_points.append(point)
            axis_colors.append([0, 255, 0])

        # Z-axis: blue (up)
        for t in np.linspace(0, 1, num_points_per_axis):
            point = origin + t * np.array([0, 0, axis_scale])
            axis_points.append(point)
            axis_colors.append([0, 0, 255])

        # Origin (white)
        axis_points.append(origin)
        axis_colors.append([255, 255, 255])

        axis_points = np.array(axis_points)
        axis_colors = np.array(axis_colors)

        self.axes_handle = server.scene.add_point_cloud(
            name="/reference/coordinate_axes",
            points=axis_points,
            colors=axis_colors,
            point_size=self.point_size * 3.0,
            point_shape="circle",
        )

    def _create_cameras(self, server: viser.ViserServer):
        """Create camera frustum visualization."""
        if not self.camera_data:
            return

        try:
            cmap = matplotlib.colormaps['gist_rainbow']
        except (AttributeError, KeyError):
            from matplotlib.cm import get_cmap
            cmap = get_cmap('gist_rainbow')

        frame_ids_sorted = sorted(self.camera_data.keys())

        for i, frame_id in enumerate(frame_ids_sorted):
            if frame_id not in self.camera_data:
                continue

            try:
                cam_info = self.camera_data[frame_id]
                intrinsic_3x3 = cam_info['intrinsic']

                fy = intrinsic_3x3[1, 1]
                fov_y = 2 * np.arctan(self.image_height / (2 * fy))
                aspect = self.image_width / self.image_height

                rgba_color = cmap(i / max(len(frame_ids_sorted) - 1, 1))
                camera_color = tuple(int(255 * x) for x in rgba_color[:3])

                handle = server.scene.add_camera_frustum(
                    name=f"/camera/frame_{frame_id}",
                    fov=fov_y,
                    aspect=aspect,
                    scale=0.05,
                    wxyz=rotation_matrix_to_wxyz(cam_info['R_world_cam']),
                    position=cam_info['position'],
                    color=camera_color,
                )

                self.camera_handles.append({
                    'handle': handle,
                    'frame_id': frame_id
                })

            except Exception as e:
                print(f"Error creating camera for frame {frame_id}: {e}")
                continue

    def _bind_events(self):
        """Bind GUI events."""
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

        @self.gui_show_mode.on_click
        def _(_) -> None:
            self._update_display()

        @self.gui_show_background.on_update
        def _(_) -> None:
            self._update_display()

        @self.gui_show_foreground.on_update
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

        @self.gui_point_size.on_update
        def _(_) -> None:
            self._update_point_size()

    def _reset_camera_callback(self, event: viser.GuiEvent):
        """Reset the connected client's camera to the initial scene view."""
        if not event.client or self.scene_center is None:
            return
        if self.initial_camera_position is not None:
            event.client.camera.position = self.initial_camera_position
        event.client.camera.look_at = self.scene_center
        event.client.camera.up_direction = np.array([0.0, 0.0, 1.0])

    def _update_point_size(self):
        """Update point size for all point clouds."""
        new_size = self.gui_point_size.value
        with self.server.atomic():
            if self.background_handle:
                self.background_handle.point_size = new_size

            for handle in self.foreground_handles:
                handle.point_size = new_size

            if self.axes_handle is not None:
                self.axes_handle.point_size = new_size * 3.0

    def _update_display(self):
        """Update display state (supports multi-version simultaneous display)."""
        current_timestep = self.gui_timestep.value
        show_mode = self.gui_show_mode.value
        show_background = self.gui_show_background.value
        show_foreground = self.gui_show_foreground.value
        show_camera = self.gui_show_camera.value
        show_axes = self.gui_show_axes.value

        if current_timestep < len(self.foreground_frames):
            current_frame = self.foreground_frames[current_timestep]
            self.gui_frame_id_label.value = current_frame['frame_id']

        with self.server.atomic():
            if self.background_handle:
                self.background_handle.visible = show_background

            for i, handle in enumerate(self.foreground_handles):
                if not show_foreground:
                    handle.visible = False
                elif show_mode == "Current Frame":
                    handle.visible = (i == current_timestep)
                else:
                    handle.visible = True

            if self.camera_handles:
                current_frame_id = None
                if current_timestep < len(self.foreground_frames):
                    current_frame_id = self.foreground_frames[current_timestep]['frame_id']

                for cam_info in self.camera_handles:
                    if not show_camera:
                        cam_info['handle'].visible = False
                    elif show_mode == "Current Frame":
                        cam_info['handle'].visible = (cam_info['frame_id'] == current_frame_id)
                    else:
                        cam_info['handle'].visible = True

            if self.axes_handle:
                self.axes_handle.visible = show_axes

    def run(self):
        """Run the viewer."""
        print(f"Step 1.2 smoothed point cloud viewer started at http://localhost:{self.port}")
        print(f"Data directory: {self.data_dir}, {self.num_frames} foreground frames")

        prev_timestep = self.gui_timestep.value
        while True:
            if self.is_playing and self.num_frames > 0:
                next_timestep = (self.gui_timestep.value + 1) % self.num_frames
                self.gui_timestep.value = next_timestep

            if self.gui_timestep.value != prev_timestep:
                self._update_display()
                prev_timestep = self.gui_timestep.value

            time.sleep(1.0 / self.gui_framerate.value)


def main():
    args = parse_args()

    if not os.path.exists(args.data_dir):
        print(f"Error: data directory does not exist: {args.data_dir}")
        return 1

    try:
        viewer = SmoothedPointcloudViewer(
            data_dir=args.data_dir,
            port=args.port,
            point_size=args.point_size,
            fps=args.fps,
            subsample=args.subsample,
            num_frames=args.num_frames,
            show_bbox=args.show_bbox
        )

        viewer.run()
        return 0

    except ValueError as e:
        print(f"Error: {e}")
        return 1
    except KeyboardInterrupt:
        return 0
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
