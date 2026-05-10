"""
Stage 2 pipeline orchestration: video generation.

Flow:
  2_0 Wan2.2 video generation with automatic captioning and depth control

Input:
  - outputs/rendering/{scene}/arc_{type}_{angle}/inference/
    ├── reference_image.png
    ├── rendered_depths.mp4
    └── original_images.mp4

Output:
  - outputs/rendering/{scene}/arc_{type}_{angle}/inference/
    ├── generated_prompt.txt
    └── output_video.mp4
"""

import subprocess
import sys
from pathlib import Path
from typing import Optional

from utils.config import Config
from utils.logging import setup_logger
from utils.seed import resolve_seed
from utils.trajectory import format_arc_trajectory_name

logger = setup_logger('stage_2_video')


class Stage2VideoPipeline:
    """Stage 2: video generation."""
    
    def __init__(self, config: Config):
        """
        Args:
            config: Configuration object.
        """
        self.config = config
        # Use the scene-level rendering directory, not the trajectory subdirectory.
        self.rendering_dir = Path(config.get('project.output_rendering_base'))
    
    def run(self, trajectory_name: Optional[str] = None):
        """
        Run the video generation flow.
        
        Args:
            trajectory_name: Trajectory subdirectory name. If omitted, it is
                generated from the configured arc_type and arc_angle.
        """
        logger.info("=" * 60)
        logger.info("Stage 2: video generation")
        logger.info("=" * 60)
        
        # Select data directory.
        if trajectory_name:
            data_dir = self.rendering_dir / trajectory_name
        else:
            # Build the default directory name from arc_type and arc_angle.
            arc_type = self.config.get('stage_1.rendering.arc_type', 'yaw')
            arc_angle = self.config.get('stage_1.rendering.arc_angle', 90)
            arc_radius_scale = self.config.get('stage_1.rendering.arc_radius_scale', 1.0)
            trajectory_name = format_arc_trajectory_name(arc_type, arc_angle, arc_radius_scale)
            data_dir = self.rendering_dir / trajectory_name
        
        logger.info(f"📁 Rendering directory: {data_dir}")
        
        # Check required input files.
        inference_dir = data_dir / "inference"
        required_files = [
            inference_dir / "reference_image.png",
            inference_dir / "rendered_depths.mp4",
            inference_dir / "original_images.mp4"
        ]
        
        missing = [f for f in required_files if not f.exists()]
        if missing:
            raise FileNotFoundError(
                f"Missing required input files:\n" +
                "\n".join(f"  - {f}" for f in missing) +
                f"\n💡 Tip: run stage_1b first to generate rendering results"
            )
        
        logger.info("✓ Input file check passed")
        
        # Check whether an output already exists.
        output_video = inference_dir / "output_video.mp4"
        if output_video.exists():
            logger.info(f"⚠️  Output video already exists and will be overwritten: {output_video}")
        
        # Call the video generation script.
        logger.info("\n🎬 Starting video generation...")
        logger.info(f"   Using Wan2.2-VACE-Fun-A14B")
        logger.info(f"   Generating caption automatically with Qwen3-VL-2B")
        
        # Resolve parameters.
        seed = resolve_seed(self.config, 'stage_2.seed', default=1)
        deterministic = bool(self.config.get('common.deterministic', False))
        num_inference_steps = self.config.get('stage_2.num_inference_steps', 50)
        sigma_shift = self.config.get('stage_2.sigma_shift', 16.0)
        cfg_scale = self.config.get('stage_2.cfg_scale', 5.0)
        
        logger.info(f"   - Seed: {seed}")
        logger.info(f"   - Deterministic: {deterministic}")
        logger.info(f"   - Inference steps: {num_inference_steps}")
        logger.info(f"   - Sigma shift: {sigma_shift}")
        logger.info(f"   - CFG scale: {cfg_scale}")
        
        # Build command.
        script_path = Path(__file__).parent.parent / 'scripts' / '2_0_Wan2.2-VACE-Fun-A14B.py'
        cmd = [
            sys.executable, str(script_path),
            '--data_dir', str(data_dir),
            '--seed', str(seed),
            '--num_inference_steps', str(num_inference_steps),
            '--sigma_shift', str(sigma_shift),
            '--cfg_scale', str(cfg_scale),
        ]
        cmd.append('--deterministic' if deterministic else '--no-deterministic')
        
        # Run command.
        result = subprocess.run(cmd, check=True)
        
        if result.returncode == 0:
            logger.info("\n" + "=" * 60)
            logger.info("🎉 Video generation completed.")
            logger.info("=" * 60)
            logger.info(f"📹 Output video: {output_video}")
            logger.info(f"📝 Generated caption: {inference_dir / 'generated_prompt.txt'}")
        else:
            raise RuntimeError("Video generation failed")


if __name__ == '__main__':
    print("🧪 Testing Stage 2 pipeline...")
    
    from utils.config import Config
    
    config = Config('configs/scenes/camel.yaml')
    pipeline = Stage2VideoPipeline(config)
    
    print(f"\n✅ Pipeline created successfully")
    print(f"   - Rendering directory: {pipeline.rendering_dir}")
    arc_type = config.get('stage_1.rendering.arc_type', 'yaw')
    arc_angle = config.get('stage_1.rendering.arc_angle')
    arc_radius_scale = config.get('stage_1.rendering.arc_radius_scale', 1.0)
    print(f"   - Default trajectory: {format_arc_trajectory_name(arc_type, arc_angle, arc_radius_scale)}")
    print(f"\nUsage:")
    print(f"   pipeline.run()  # Use the default configured trajectory")
    print(f"   pipeline.run(trajectory_name='arc_yaw_-120_scale_1p0')  # Use a specific trajectory")
