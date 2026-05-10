"""
Complete Stage 1 pipeline: point-cloud processing and rendering.

Flow:
  Stage 1A: 1_0 -> 1_1 -> 1_2 (point clouds)
  Stage 1B: 1_4 (rendering)

Notes:
  - 1_3 is an independent interactive tool and is not part of the automated flow.
  - Rendering uses an automatically generated arc trajectory by default.
  - A trajectory generated with 1_3 can also be provided.
"""

from typing import Optional
from utils.config import Config
from utils.logging import setup_logger
from pipeline.stage_1a_pointcloud_pipeline import Stage1APointCloudPipeline
from pipeline.stage_1b_rendering_pipeline import Stage1BRenderingPipeline

logger = setup_logger('stage_1')


class Stage1Pipeline:
    """Complete Stage 1 pipeline."""
    
    def __init__(self, config: Config):
        """
        Args:
            config: Configuration object.
        """
        self.config = config
    
    def run(self, 
            resume_from: Optional[str] = None,
            trajectory_json: Optional[str] = None):
        """
        Run the complete Stage 1 flow.
        
        Args:
            resume_from: Step to resume from.
                - 'foreground', 'background', 'align': resume from a Stage 1A step
                - 'render': skip 1A and run only 1B
            trajectory_json: Trajectory file for rendering. None generates an arc trajectory.
        """
        logger.info("=" * 60)
        logger.info("Stage 1: point-cloud processing + rendering")
        logger.info("=" * 60)
        
        # Stage 1A: point-cloud processing.
        if resume_from != 'render':
            logger.info("\n### Stage 1A: point-cloud generation and alignment ###\n")
            stage_1a = Stage1APointCloudPipeline(self.config)
            stage_1a.run(resume_from=resume_from)
        else:
            logger.info("⏭️  Skipping Stage 1A because resume_from is 'render'")
        
        # Stage 1B: rendering.
        logger.info("\n### Stage 1B: rendering ###\n")
        stage_1b = Stage1BRenderingPipeline(self.config)
        stage_1b.run(trajectory_json=trajectory_json)
        
        logger.info("\n" + "=" * 60)
        logger.info("🎉 Stage 1 pipeline completed.")
        logger.info("=" * 60)


if __name__ == '__main__':
    print("🧪 Testing Stage 1 pipeline...")
    
    from utils.config import Config
    
    config = Config()
    pipeline = Stage1Pipeline(config)
    
    print(f"\n✅ Pipeline created successfully")
    print(f"\nComplete flow:")
    print(f"  Stage 1A:")
    print(f"    1. foreground - foreground point clouds")
    print(f"    2. background - background point cloud")
    print(f"    3. align - point-cloud alignment")
    print(f"  Stage 1B:")
    print(f"    4. render - rendering with arc trajectory")
    print(f"\nSupported resume steps:")
    print(f"  --resume_from foreground|background|align|render")
