#!/usr/bin/env python3
"""Point cloud rendering from point clouds and camera trajectories."""

import os
import sys
import argparse

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from core.rendering import PointCloudRenderer
from utils.args import create_base_parser, merge_args_with_config


def create_parser():
    """Create argument parser."""
    parser = create_base_parser('Step 1.4: Point cloud rendering')

    parser.add_argument(
        '--data_dir',
        type=str,
        required=False,
        help='Data directory, defaults to project.output_prepared from config'
    )

    traj_group = parser.add_mutually_exclusive_group()
    traj_group.add_argument(
        '--trajectory_json',
        type=str,
        help='Trajectory JSON file'
    )
    traj_group.add_argument(
        '--arc_mode',
        action='store_true',
        help='Enable arc trajectory mode'
    )

    arc_group = parser.add_argument_group('Arc mode parameters')
    arc_group.add_argument('--arc_type', type=str,
                          choices=['yaw', 'pitch', 'roll'],
                          help='Arc type: yaw | pitch | roll')
    arc_group.add_argument('--arc_angle', type=float, help='Arc angle (degrees)')
    arc_group.add_argument('--arc_num_frames', type=int, help='Number of arc trajectory frames')
    arc_group.add_argument('--arc_save_json', type=str, help='Save arc trajectory to JSON file')
    arc_group.add_argument('--fixed_video_time', type=int, default=None,
                           help='Orbit mode: use a fixed frame (e.g. 0) for all views')
    arc_group.add_argument('--output_gif', action='store_true', help='Generate orbit.gif from rendered frames')
    arc_group.add_argument('--foreground_only', action='store_true', help='Render foreground only (no background)')
    arc_group.add_argument('--output_rgba', action='store_true',
                           help='Output RGBA with transparent background')
    arc_group.add_argument('--arc_radius', type=float, default=None,
                           help='Arc radius (direct value, same unit as scene)')
    arc_group.add_argument('--arc_radius_scale', type=float, default=None,
                           help='Arc radius scale factor, >1 moves camera back; mutually exclusive with --arc_radius')
    arc_group.add_argument('--foreground_1_view', action='store_true',
                           help='Use single-view foreground point cloud (*_foreground_1_view.ply)')

    render_group = parser.add_argument_group('Rendering parameters')
    render_group.add_argument('--point_radius_px', type=float, help='Point radius (pixels)')
    render_group.add_argument('--image_height', type=int, help='Image height')
    render_group.add_argument('--image_width', type=int, help='Image width')
    render_group.add_argument('--fps', type=int, help='Video frame rate')
    return parser


def main():
    parser = create_parser()
    args = parser.parse_args()

    config = merge_args_with_config(args)

    data_dir = args.data_dir if args.data_dir else config.get('project.output_prepared')

    if not data_dir:
        raise ValueError("Must specify data directory via --data_dir or project.output_prepared in config")

    if args.arc_type is not None:
        config.update('stage_1.rendering.arc_type', args.arc_type)
    if args.arc_angle is not None:
        config.update('stage_1.rendering.arc_angle', args.arc_angle)
    if args.point_radius_px is not None:
        config.update('stage_1.rendering.point_radius_px', args.point_radius_px)
    if args.image_height is not None:
        config.update('stage_1.rendering.image_height', args.image_height)
    if args.image_width is not None:
        config.update('stage_1.rendering.image_width', args.image_width)
    if args.fps is not None:
        config.update('stage_1.rendering.fps', args.fps)

    num_frames = config.get('common.num_frames')
    if args.arc_mode and getattr(args, 'arc_num_frames', None) is not None:
        num_frames = args.arc_num_frames

    renderer = PointCloudRenderer.from_config(config)

    renderer.render_trajectory(
        data_dir=data_dir,
        trajectory_json=args.trajectory_json,
        arc_mode=args.arc_mode,
        arc_type=config.get('stage_1.rendering.arc_type', 'yaw'),
        arc_angle=config.get('stage_1.rendering.arc_angle'),
        num_frames=num_frames,
        fixed_video_time=getattr(args, 'fixed_video_time', None),
        output_gif=getattr(args, 'output_gif', False),
        foreground_only=getattr(args, 'foreground_only', False),
        output_rgba=getattr(args, 'output_rgba', False),
        arc_radius=getattr(args, 'arc_radius', None),
        arc_radius_scale=getattr(args, 'arc_radius_scale', None),
        foreground_1_view=getattr(args, 'foreground_1_view', False),
    )


if __name__ == "__main__":
    main()
