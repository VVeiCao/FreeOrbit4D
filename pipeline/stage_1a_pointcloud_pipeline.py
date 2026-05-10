"""
Stage 1A pipeline orchestration: point-cloud generation and alignment.

Flow:
  1_0 foreground point-cloud generation -> 1_1 background point-cloud generation -> 1_2 alignment

Input:
  - outputs/prepared/

Output:
  - {frame_id}/pointcloud/*.ply
  - global_background.ply
"""

from pathlib import Path
from typing import Optional, List

from utils.config import Config
from utils.logging import setup_logger
from utils.file_io import find_frame_dirs
from core.pointcloud import ForegroundPointCloudGenerator, BackgroundPointCloudGenerator
from core.alignment import PointCloudAligner

logger = setup_logger('stage_1a_pointcloud')


class Stage1APointCloudPipeline:
    """Stage 1A: point-cloud generation and alignment."""
    
    STEPS = ['foreground', 'background', 'align']
    
    def __init__(self, config: Config):
        """
        Args:
            config: Configuration object.
        """
        self.config = config
        self.data_dir = Path(config.get('project.output_prepared'))
    
    def run(self, resume_from: Optional[str] = None):
        """
        Run the Stage 1A flow.
        
        Args:
            resume_from: Step to resume from. None starts from the beginning.
        """
        logger.info("=" * 60)
        logger.info("Stage 1A: point-cloud generation and alignment")
        logger.info("=" * 60)
        logger.info(f"Data directory: {self.data_dir}")
        logger.info("=" * 60)
        
        start_idx = self._get_start_index(resume_from)
        
        if resume_from:
            logger.info(f"\n-> Resuming from step '{resume_from}'\n")
        
        # Run steps.
        for i in range(start_idx, len(self.STEPS)):
            step_name = self.STEPS[i]
            logger.info(f"\n{'='*60}")
            logger.info(f"Step {i+1}/{len(self.STEPS)}: {step_name}")
            logger.info(f"{'='*60}")
            
            try:
                step_method = getattr(self, f'_run_{step_name}')
                step_method()
                logger.info(f"✅ {step_name} completed\n")
                
            except Exception as e:
                logger.error(f"❌ {step_name} failed: {e}")
                logger.info(f"💡 After fixing it, resume with: --resume_from {step_name}")
                raise
        
        logger.info("=" * 60)
        logger.info("🎉 Stage 1A completed.")
        logger.info("=" * 60)
    
    def _run_foreground(self):
        """Step 1: foreground point-cloud generation (1_0)."""
        logger.info("Using ForegroundPointCloudGenerator")
        
        generator = ForegroundPointCloudGenerator.from_config(self.config)
        generator.load_model()
        generator.batch_process(
            folder=str(self.data_dir),
            num_frames=self.config.get('common.num_frames')
        )
    
    def _run_background(self):
        """Step 2: background point-cloud generation (1_1)."""
        logger.info("Using BackgroundPointCloudGenerator")
        
        generator = BackgroundPointCloudGenerator.from_config(self.config)
        generator.load_model()
        generator.generate_global_background(data_dir=str(self.data_dir))
    
    def _run_align(self):
        """Step 3: point-cloud alignment (1_2)."""
        logger.info("Using PointCloudAligner")
        
        aligner = PointCloudAligner.from_config(self.config)
        aligner.align_all_frames(
            folder=str(self.data_dir),
            num_frames=self.config.get('common.num_frames')
        )
    
    def _get_start_index(self, resume_from: Optional[str]) -> int:
        """Get the starting step index."""
        if resume_from is None:
            return 0
        if resume_from not in self.STEPS:
            raise ValueError(
                f"Invalid step name: {resume_from}\n"
                f"Choices: {', '.join(self.STEPS)}"
            )
        return self.STEPS.index(resume_from)


if __name__ == '__main__':
    print("🧪 Testing Stage 1A pipeline...")
    
    from utils.config import Config
    
    config = Config()
    pipeline = Stage1APointCloudPipeline(config)
    
    print(f"\n✅ Pipeline created successfully")
    print(f"   - Data directory: {pipeline.data_dir}")
    print(f"\nFlow steps: {' -> '.join(pipeline.STEPS)}")
    print(f"\nSupported resume steps: --resume_from {step}" for step in pipeline.STEPS)
