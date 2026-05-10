#!/usr/bin/env python3
"""Background point cloud generation with morphological foreground/background separation."""

import os
import sys
import argparse

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from core.pointcloud import BackgroundPointCloudGenerator
from utils.args import create_base_parser, merge_args_with_config


def create_parser():
    """Create argument parser."""
    parser = create_base_parser('Step 1.1: Background point cloud generation (DPG)')

    parser.add_argument(
        '--folder',
        type=str,
        required=False,
        help='Data directory (step 0.1 output), defaults to project.output_prepared from config'
    )

    parser.add_argument('--point_source', type=str, choices=['point_map', 'backproject'],
                       help='Point cloud source')
    parser.add_argument('--pad_pixels', type=int, help='Mask dilation pixels')
    parser.add_argument('--conf_threshold', type=float, help='Confidence threshold')
    parser.add_argument('--voxel_size', type=float, help='Voxel downsampling size')
    parser.add_argument('--outlier_nb_neighbors', type=int, help='Outlier detection neighbor count')
    parser.add_argument('--outlier_std_ratio', type=float, help='Outlier detection std ratio')

    return parser


def main():
    parser = create_parser()
    args = parser.parse_args()

    config = merge_args_with_config(args)

    folder = args.folder if args.folder else config.get('project.output_prepared')

    if not folder:
        raise ValueError("Must specify data directory via --folder or project.output_prepared in config")

    if args.point_source:
        config.update('stage_1.background.point_source', args.point_source)
    if args.pad_pixels:
        config.update('stage_1.background.pad_pixels', args.pad_pixels)
    if args.conf_threshold:
        config.update('stage_1.background.confidence_threshold', args.conf_threshold)
    if args.voxel_size:
        config.update('stage_1.background.voxel_size', args.voxel_size)
    if args.outlier_nb_neighbors:
        config.update('stage_1.background.outlier_nb_neighbors', args.outlier_nb_neighbors)
    if args.outlier_std_ratio:
        config.update('stage_1.background.outlier_std_ratio', args.outlier_std_ratio)

    generator = BackgroundPointCloudGenerator.from_config(config)

    generator.load_model()

    generator.generate_global_background(data_dir=folder)


if __name__ == "__main__":
    main()
