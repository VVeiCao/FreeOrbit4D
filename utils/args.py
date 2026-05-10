"""Command-line argument parsing and config merging utilities."""

import argparse
from typing import Optional
from pathlib import Path
from .config import Config


def create_base_parser(description: str) -> argparse.ArgumentParser:
    """Create a base argument parser with common options shared across scripts."""
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    config_group = parser.add_argument_group('Config')
    config_group.add_argument(
        '--config',
        type=str,
        help='Config file path (YAML, relative to configs/ or absolute)'
    )

    common_group = parser.add_argument_group('Common')
    common_group.add_argument(
        '--device',
        type=str,
        choices=['cuda', 'cpu'],
        help='Compute device'
    )
    common_group.add_argument(
        '--seed',
        type=int,
        help='Random seed'
    )
    common_group.add_argument(
        '--num_frames',
        type=int,
        help='Number of frames to process (None for all)'
    )

    return parser


def merge_args_with_config(args: argparse.Namespace) -> Config:
    """Merge CLI arguments with config file. Priority: CLI > config file > defaults."""
    if hasattr(args, 'config') and args.config:
        config = Config(args.config)
    else:
        config = Config()

    if hasattr(args, 'device') and args.device is not None:
        config.update('common.device', args.device)

    if hasattr(args, 'seed') and args.seed is not None:
        config.update('common.seed', args.seed)
        config.update('stage_2.seed', args.seed)

    if hasattr(args, 'num_frames') and args.num_frames is not None:
        config.update('common.num_frames', args.num_frames)

    if hasattr(args, 'arc_angle') and args.arc_angle is not None:
        config.update('stage_1.rendering.arc_angle', args.arc_angle)

    if hasattr(args, 'point_radius_px') and args.point_radius_px is not None:
        config.update('stage_1.rendering.point_radius_px', args.point_radius_px)

    if hasattr(args, 'image_height') and args.image_height is not None:
        config.update('stage_1.rendering.image_height', args.image_height)

    if hasattr(args, 'image_width') and args.image_width is not None:
        config.update('stage_1.rendering.image_width', args.image_width)

    return config


def get_config_value(args: argparse.Namespace,
                     config: Config,
                     arg_name: str,
                     config_path: str,
                     default: any = None) -> any:
    """Get a config value with CLI argument taking priority over config file."""
    if hasattr(args, arg_name):
        value = getattr(args, arg_name)
        if value is not None:
            return value

    return config.get(config_path, default)
