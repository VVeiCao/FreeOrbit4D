#!/usr/bin/env python3
"""Foreground point cloud generation with 5-view VGGT inference."""

import os
import sys
import argparse

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from core.pointcloud import ForegroundPointCloudGenerator
from utils.args import create_base_parser, merge_args_with_config


def create_parser():
    """Create argument parser."""
    parser = create_base_parser('Step 1.0: Foreground point cloud generation (5-view, VGGT)')

    parser.add_argument(
        '--folder',
        type=str,
        required=False,
        help='Data directory (step 0.1 output), defaults to project.output_prepared from config'
    )

    return parser


def main():
    parser = create_parser()
    args = parser.parse_args()

    config = merge_args_with_config(args)

    folder = args.folder if args.folder else config.get('project.output_prepared')

    if not folder:
        raise ValueError("Must specify data directory via --folder or project.output_prepared in config")

    generator = ForegroundPointCloudGenerator.from_config(config)

    generator.load_model()

    generator.batch_process(
        folder=folder,
        num_frames=config.get('common.num_frames')
    )


if __name__ == "__main__":
    main()
