#!/usr/bin/env python3
"""
Pipeline execution entrypoint for automated workflow management.

Pipeline choices:
  stage_0    Stage 0: data preparation (0_0 -> 0_1)
  stage_1a   Stage 1A: point-cloud processing (1_0 -> 1_1 -> 1_2)
  stage_1b   Stage 1B: rendering (1_4)
  stage_1    Complete Stage 1 (1A + 1B)
  stage_2    Stage 2: video generation (2_0)
  full       End-to-end pipeline (0 -> 1 -> 2)

--resume_from values:
  stage_0:  multiview | preparation
  stage_1a: foreground | background | align
  stage_1:  foreground | background | align | render
  full:     multiview | preparation | foreground | background | align | render | video

Examples:
------------------------------------------------------------------

1. Point-cloud processing
   python run_pipeline.py stage_1a --config configs/scenes/camel.yaml

2. Rendering
   python run_pipeline.py stage_1b --config configs/scenes/camel.yaml

3. Video generation
   python run_pipeline.py full --config configs/scenes/robot.yaml --resume_from background

4. Complete Stage 1 flow
   python run_pipeline.py stage_1 --config configs/scenes/car-turn_yaw_-120.yaml --resume_from background

5. End-to-end flow
python run_pipeline.py full --config configs/scenes/unitree6.yaml && python run_pipeline.py full --config configs/scenes/unitree7.yaml
export CUDA_VISIBLE_DEVICES=1
   python run_pipeline.py full --config configs/eval/car-roundabout_yaw_-120.yaml --resume_from video --trajectory_json original_global_camera.json
   cp outputs/rendering/lecun2/arc_yaw_-120_scale_1p0/inference/reference_image.png outputs/rendering/lecun2/test2/inference/reference_image.png
   python run_pipeline.py full --config configs/scenes/car-roundabout-camel.yaml --resume_from render --trajectory_json test2.json

   python run_pipeline.py full --config configs/scenes/lecun4.yaml --resume_from render --trajectory_json test1.json
   python run_pipeline.py full --config configs/eval/bear_yaw_-120.yaml --resume_from render --trajectory_json test_1.json

   python run_pipeline.py full --config configs/eval/parkour_yaw_-120.yaml --resume_from render --trajectory_json round_1.json


6. Resume from a step
   python run_pipeline.py stage_1a --config configs/scenes/camel.yaml --resume_from align

7. Debug mode with fewer frames
   python run_pipeline.py stage_1a --config configs/scenes/camel.yaml --num_frames 3
"""

import sys
import argparse
from pathlib import Path

# Add the project root to PYTHONPATH.
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from utils.config import Config
from utils.logging import setup_logger
from utils.seed import resolve_seed, set_global_seed
from pipeline.stage_0_pipeline import Stage0Pipeline
from pipeline.stage_1a_pointcloud_pipeline import Stage1APointCloudPipeline
from pipeline.stage_1b_rendering_pipeline import Stage1BRenderingPipeline
from pipeline.stage_1_pipeline import Stage1Pipeline
from pipeline.stage_2_pipeline import Stage2VideoPipeline
from pipeline.full_pipeline import FullPipeline

logger = setup_logger('pipeline')


def create_parser():
    """Create the argument parser."""
    parser = argparse.ArgumentParser(
        description='Pipeline execution tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Point-cloud processing
  python run_pipeline.py stage_1a --config configs/scenes/camel.yaml
  
  # Rendering
  python run_pipeline.py stage_1b --config configs/scenes/camel.yaml
  
  # Video generation
  python run_pipeline.py stage_2 --config configs/scenes/camel.yaml
  
  # Full pipeline
  python run_pipeline.py full --config configs/scenes/camel.yaml

See the module docstring for more details.
        """
    )
    
    # Pipeline selection.
    parser.add_argument(
        'pipeline',
        choices=['stage_0', 'stage_1a', 'stage_1b', 'stage_1', 'stage_2', 'full'],
        help='''Pipeline to run:
  stage_0  - Stage 0 (0_0 -> 0_1)
  stage_1a - Stage 1A (1_0 -> 1_1 -> 1_2)
  stage_1b - Stage 1B (1_4 rendering)
  stage_1  - Complete Stage 1 (1A + 1B)
  stage_2  - Stage 2 (2_0 video generation)
  full     - End-to-end pipeline (0 -> 1 -> 2)'''
    )
    
    # Config file.
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to the YAML config file'
    )
    
    # Resume.
    parser.add_argument(
        '--resume_from',
        type=str,
        metavar='STEP',
        help='''Resume from a specific step:
  stage_0:  multiview | preparation
  stage_1a: foreground | background | align
  stage_1:  foreground | background | align | render
  full:     multiview | preparation | foreground | background | align | render | video'''
    )
    
    # Common arguments.
    parser.add_argument('--num_frames', type=int, help='Number of frames to process for debugging')
    parser.add_argument('--seed', type=int, help='Random seed overriding common.seed in the config')
    
    # Stage 1B / Stage 1 arguments.
    parser.add_argument('--trajectory_json', type=str, 
                       help='Trajectory JSON file for stage_1b or stage_1')
    parser.add_argument('--arc_angle', type=float,
                       help='Arc trajectory angle in degrees, overriding the config')
    parser.add_argument('--arc_radius_scale', type=float,
                       help='Arc trajectory radius scale, overriding the config')
    
    # Stage 2 arguments.
    parser.add_argument('--trajectory_name', type=str,
                       help='Trajectory subdirectory name for stage_2, for example arc_yaw_-120_scale_1p0')
    
    return parser


def main():
    """Run the selected pipeline."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Load config.
    config = Config(args.config)
    
    # Override config values from CLI arguments.
    if args.num_frames is not None:
        config.update('common.num_frames', args.num_frames)
    if args.seed is not None:
        config.update('common.seed', args.seed)
        config.update('stage_2.seed', args.seed)
    if args.arc_angle is not None:
        config.update('stage_1.rendering.arc_angle', args.arc_angle)
    if args.arc_radius_scale is not None:
        config.update('stage_1.rendering.arc_radius_scale', args.arc_radius_scale)

    seed = resolve_seed(config)
    deterministic = bool(config.get('common.deterministic', False))
    set_global_seed(seed, deterministic=deterministic)
    
    logger.info("=" * 60)
    logger.info(f"Pipeline: {args.pipeline}")
    logger.info(f"Config file: {args.config}")
    logger.info(f"Seed: {seed} (deterministic={deterministic})")
    if args.resume_from:
        logger.info(f"Resume mode: starting from '{args.resume_from}'")
    logger.info("=" * 60)
    
    # Run the selected pipeline.
    try:
        if args.pipeline == 'stage_0':
            pipeline = Stage0Pipeline(config)
            pipeline.run(resume_from=args.resume_from)
            
        elif args.pipeline == 'stage_1a':
            pipeline = Stage1APointCloudPipeline(config)
            pipeline.run(resume_from=args.resume_from)
            
        elif args.pipeline == 'stage_1b':
            if args.resume_from:
                logger.warning("stage_1b has only one step; ignoring --resume_from")
            pipeline = Stage1BRenderingPipeline(config)
            pipeline.run(trajectory_json=args.trajectory_json)
            
        elif args.pipeline == 'stage_1':
            pipeline = Stage1Pipeline(config)
            pipeline.run(
                resume_from=args.resume_from,
                trajectory_json=args.trajectory_json
            )
            
        elif args.pipeline == 'stage_2':
            if args.resume_from:
                logger.warning("stage_2 has only one step; ignoring --resume_from")
            pipeline = Stage2VideoPipeline(config)
            pipeline.run(trajectory_name=args.trajectory_name)
            
        elif args.pipeline == 'full':
            pipeline = FullPipeline(config)
            pipeline.run(
                resume_from=args.resume_from,
                trajectory_json=args.trajectory_json,
                trajectory_name=args.trajectory_name
            )
        
        logger.info("\n✅ Pipeline completed successfully.")
        
    except Exception as e:
        logger.error(f"\n❌ Pipeline failed: {e}")
        logger.info("\n💡 Tips:")
        logger.info("  - Check that the config file is correct")
        logger.info("  - Check that the input data exists")
        raise


if __name__ == '__main__':
    main()
