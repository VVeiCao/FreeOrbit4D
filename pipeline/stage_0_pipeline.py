"""
Stage 0 pipeline orchestration: data preparation.

Flow:
  0_0 multiview generation -> 0_1 data preparation

Input:
  - Raw image directory
  - Mask directory

Output:
  - outputs/multiview/    (intermediate files)
  - outputs/prepared/     (prepared scene data)
"""

import os
import sys
import subprocess
from pathlib import Path
from typing import Optional

from utils.config import Config
from utils.logging import setup_logger
from utils.seed import resolve_seed
from core.data_processor import DataProcessor

logger = setup_logger('stage_0_pipeline')


class Stage0Pipeline:
    """Complete Stage 0 pipeline: data preparation."""
    
    STEPS = ['multiview', 'preparation']
    
    def __init__(self, config: Config):
        """
        Args:
            config: Configuration object.
        """
        self.config = config
        
        # Path configuration.
        self.input_images = Path(config.get('project.input_images'))
        self.input_masks = Path(config.get('project.input_masks'))
        self.output_multiview = Path(config.get('project.output_multiview'))
        self.output_prepared = Path(config.get('project.output_prepared'))
    
    def run(self, resume_from: Optional[str] = None):
        """
        Run the complete Stage 0 flow.
        
        Args:
            resume_from: Step to resume from. None starts from the beginning.
        """
        start_idx = self._get_start_index(resume_from)
        
        # Run steps.
        for i, step_name in enumerate(self.STEPS[start_idx:], start_idx + 1):
            logger.info("=" * 60)
            logger.info(f"Step {i}/{len(self.STEPS)}: {step_name}")
            logger.info("=" * 60)
            
            try:
                # Call the matching step method.
                step_method = getattr(self, f'_run_{step_name}')
                step_method()
                
                logger.info(f"✅ {step_name} completed\n")
                
            except Exception as e:
                logger.error(f"❌ {step_name} failed: {e}")
                raise
        
        logger.info("=" * 60)
        logger.info("🎉 Stage 0 pipeline completed.")
        logger.info("=" * 60)
        logger.info(f"📁 Output directory: {self.output_prepared}")
    
    def _run_multiview(self):
        """Step 1: multiview generation (0_0)."""
        logger.info(f"Input images: {self.input_images}")
        logger.info(f"Input masks: {self.input_masks}")
        logger.info(f"Output directory: {self.output_multiview}")
        
        if not self.input_images.exists():
            raise FileNotFoundError(f"Image directory does not exist: {self.input_images}")
        if not self.input_masks.exists():
            raise FileNotFoundError(f"Mask directory does not exist: {self.input_masks}")
        
        script_path = Path(__file__).parent.parent / 'scripts' / '0_0_gen_multiviews.py'
        
        seed = resolve_seed(self.config)
        deterministic = bool(self.config.get('common.deterministic', False))
        logger.info(f"Seed: {seed}")
        logger.info(f"Deterministic: {deterministic}")

        cmd = [
            sys.executable,
            str(script_path),
            '--input_images_dir', str(self.input_images),
            '--input_masks_dir', str(self.input_masks),
            '--output_folder', str(self.output_multiview),
            '--model_type', self.config.get('stage_0.multiview.model_type', 'sv4d2'),
            '--num_steps', str(self.config.get('stage_0.multiview.num_steps', 50)),
            '--seed', str(seed),
            f'--deterministic={str(deterministic).lower()}',
        ]
        num_frames = self.config.get('common.num_frames')
        if num_frames:
            cmd.extend(['--num_frames', str(num_frames)])
        
        logger.info(f"Running command: {' '.join(cmd[:3])} ...")
        result = subprocess.run(cmd, check=True)
        
        if result.returncode != 0:
            raise RuntimeError("Multiview generation failed")
        
        logger.info(f"✓ Multiview data generated: {self.output_multiview}")
    
    def _run_preparation(self):
        """Step 2: data preparation (0_1)."""
        logger.info(f"Input directory: {self.output_multiview}")
        logger.info(f"Output directory: {self.output_prepared}")
        
        # Validate input.
        multiview_dir = self.output_multiview / 'multiview_images'
        if not multiview_dir.exists():
            raise FileNotFoundError(
                f"Multiview data does not exist: {multiview_dir}\n"
                f"Run the multiview step first."
            )
        
        # Create processor.
        processor = DataProcessor.from_config(self.config)
        
        # Batch process frames.
        processor.process_scene(
            input_dir=str(self.output_multiview),
            output_dir=str(self.output_prepared),
            num_frames=self.config.get('common.num_frames')
        )
        
        logger.info(f"✓ Prepared data generated: {self.output_prepared}")
        
        # Summarize output.
        frame_dirs = list(self.output_prepared.glob('[0-9]*'))
        logger.info(f"✓ Processed {len(frame_dirs)} frames")
    
    def _get_start_index(self, resume_from: Optional[str]) -> int:
        """Get the starting step index."""
        if resume_from is None:
            return 0
        
        if resume_from not in self.STEPS:
            raise ValueError(
                f"Invalid step name: {resume_from}. "
                f"Choices: {', '.join(self.STEPS)}"
            )
        
        return self.STEPS.index(resume_from)


if __name__ == '__main__':
    print("🧪 Testing Stage 0 pipeline...")
    
    from utils.config import Config
    
    # Smoke-test construction.
    config = Config()
    pipeline = Stage0Pipeline(config)
    
    print(f"\n✅ Pipeline created successfully")
    print(f"   - Input images: {pipeline.input_images}")
    print(f"   - Input masks: {pipeline.input_masks}")
    print(f"   - Multiview output: {pipeline.output_multiview}")
    print(f"   - Prepared output: {pipeline.output_prepared}")
    print(f"\nFlow steps: {' -> '.join(pipeline.STEPS)}")
