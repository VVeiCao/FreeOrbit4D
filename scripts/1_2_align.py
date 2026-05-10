#!/usr/bin/env python3
"""Point cloud alignment: align 5-view point clouds to background space with Kalman smoothing."""

import os
import sys
import argparse

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from core.alignment import PointCloudAligner
from utils.args import create_base_parser, merge_args_with_config


def create_parser():
    """Create argument parser."""
    parser = create_base_parser('Step 1.2: Point cloud alignment (with Kalman smoothing)')

    parser.add_argument(
        '--folder',
        type=str,
        required=False,
        help='Data directory, defaults to project.output_prepared from config'
    )

    parser.add_argument('--min_points', type=int, help='Minimum correspondence point count')

    erosion_group = parser.add_argument_group('Morphological erosion (mask edge removal)')
    erosion_group.add_argument('--erosion_kernel_size', type=int,
                              help='Erosion kernel size (odd, 5-21, default 11)')
    erosion_group.add_argument('--erosion_iterations', type=int,
                              help='Erosion iterations (1-3, default 1)')

    corr_group = parser.add_argument_group('Correspondence pair filtering')
    corr_group.add_argument('--corr_conf_threshold', type=float,
                           help='Confidence percentile threshold for correspondence pairs')
    corr_group.add_argument('--corr_outlier_nb_neighbors', type=int,
                           help='Neighbor count for correspondence outlier detection')
    corr_group.add_argument('--corr_outlier_std_ratio', type=float,
                           help='Std ratio for correspondence outlier detection')

    complete_group = parser.add_argument_group('Complete point cloud filtering')
    complete_group.add_argument('--complete_conf_threshold', type=float,
                               help='Confidence percentile threshold for complete point cloud')
    complete_group.add_argument('--complete_outlier_nb_neighbors', type=int,
                               help='Neighbor count for complete cloud outlier detection')
    complete_group.add_argument('--complete_outlier_std_ratio', type=float,
                               help='Std ratio for complete cloud outlier detection')

    smooth_group = parser.add_argument_group('Temporal smoothing (Kalman filter)')
    smooth_group.add_argument('--no_smoothing', action='store_true',
                             help='Disable Kalman smoothing (enabled by default)')
    smooth_group.add_argument('--kalman_process_noise', type=float,
                             help='Process noise Q (smaller = smoother)')
    smooth_group.add_argument('--kalman_measurement_noise', type=float,
                             help='Measurement noise R')
    smooth_group.add_argument('--kalman_z_penalty', type=float,
                             help='Z-axis smoothing penalty (<1 = smoother, suggested 0.1-0.5, default 1.0)')

    debug_group = parser.add_argument_group('Debug options')
    debug_group.add_argument('--save_debug', action='store_true',
                            help='Save debug point clouds (10+ PLY files per frame)')
    debug_group.add_argument('--debug_single_view', action='store_true',
                            help='Debug visualization shows single view only (view 0)')

    return parser


def main():
    parser = create_parser()
    args = parser.parse_args()

    config = merge_args_with_config(args)

    folder = args.folder if args.folder else config.get('project.output_prepared')

    if not folder:
        raise ValueError("Must specify data directory via --folder or project.output_prepared in config")

    if args.erosion_kernel_size is not None:
        config.update('stage_1.alignment.erosion_kernel_size', args.erosion_kernel_size)
    if args.erosion_iterations is not None:
        config.update('stage_1.alignment.erosion_iterations', args.erosion_iterations)
    if args.no_smoothing:
        config.update('stage_1.alignment.enable_smoothing', False)
    if args.kalman_process_noise is not None:
        config.update('stage_1.alignment.kalman_process_noise', args.kalman_process_noise)
    if args.kalman_measurement_noise is not None:
        config.update('stage_1.alignment.kalman_measurement_noise', args.kalman_measurement_noise)
    if args.kalman_z_penalty is not None:
        config.update('stage_1.alignment.kalman_z_penalty', args.kalman_z_penalty)
    if args.corr_conf_threshold is not None:
        config.update('stage_1.alignment.corr_conf_threshold', args.corr_conf_threshold)
    if args.complete_conf_threshold is not None:
        config.update('stage_1.alignment.complete_conf_threshold', args.complete_conf_threshold)
    if args.min_points is not None:
        config.update('stage_1.alignment.min_points', args.min_points)
    if args.save_debug:
        config.update('stage_1.alignment.save_debug', True)
    if args.debug_single_view:
        config.update('stage_1.alignment.debug_show_all_views', False)

    aligner = PointCloudAligner.from_config(config)

    aligner.align_all_frames(
        folder=folder,
        num_frames=config.get('common.num_frames')
    )


if __name__ == "__main__":
    main()
