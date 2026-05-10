#!/usr/bin/env python3
"""Data preparation: inverse-transform multi-view images to training size."""

import os
import sys
import argparse

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from core.data_processor import DataProcessor
from utils.args import create_base_parser, merge_args_with_config


def create_parser():
    """Create argument parser."""
    parser = create_base_parser('Step 0.1: Data preparation')

    parser.add_argument(
        '--input_dir',
        type=str,
        required=False,
        help='Input directory (step 0 output), defaults to project.output_multiview from config'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=False,
        help='Output directory, defaults to project.output_prepared from config'
    )

    parser.add_argument(
        '--output_height',
        type=int,
        help='Output image height'
    )
    parser.add_argument(
        '--output_width',
        type=int,
        help='Output image width'
    )
    return parser


def main():
    parser = create_parser()
    args = parser.parse_args()

    config = merge_args_with_config(args)

    input_dir = args.input_dir if args.input_dir else config.get('project.output_multiview')
    output_dir = args.output_dir if args.output_dir else config.get('project.output_prepared')

    if not input_dir:
        raise ValueError("Must specify input directory via --input_dir or project.output_multiview in config")
    if not output_dir:
        raise ValueError("Must specify output directory via --output_dir or project.output_prepared in config")

    if args.output_height is not None:
        config.update('stage_0.preparation.output_height', args.output_height)
    if args.output_width is not None:
        config.update('stage_0.preparation.output_width', args.output_width)

    processor = DataProcessor.from_config(config)

    processor.process_scene(
        input_dir=input_dir,
        output_dir=output_dir,
        num_frames=config.get('common.num_frames')
    )


if __name__ == '__main__':
    main()
