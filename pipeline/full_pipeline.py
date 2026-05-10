"""
End-to-end pipeline orchestration from raw images to final video.

Flow:
  Stage 0: data preparation (0_0 -> 0_1)
  Stage 1: point-cloud processing + rendering (1_0 -> 1_1 -> 1_2 -> 1_4)
  Stage 2: video generation (2_0)

Input: raw images + masks
Output: final generated video
"""

from pathlib import Path
from typing import Optional

from utils.config import Config
from utils.logging import setup_logger
from utils.trajectory import format_arc_trajectory_name
from pipeline.stage_0_pipeline import Stage0Pipeline
from pipeline.stage_1_pipeline import Stage1Pipeline
from pipeline.stage_2_pipeline import Stage2VideoPipeline

logger = setup_logger('full_pipeline')


class FullPipeline:
    """End-to-end pipeline."""
    
    # All steps across stages.
    ALL_STEPS = [
        # Stage 0.
        'multiview', 'preparation',
        # Stage 1.
        'foreground', 'background', 'align', 'render',
        # Stage 2.
        'video'
    ]
    
    STAGE_0_STEPS = ['multiview', 'preparation']
    STAGE_1_STEPS = ['foreground', 'background', 'align', 'render']
    STAGE_2_STEPS = ['video']
    
    def __init__(self, config: Config):
        """
        Args:
            config: Configuration object.
        """
        self.config = config
    
    def run(self, 
            resume_from: Optional[str] = None,
            trajectory_json: Optional[str] = None,
            trajectory_name: Optional[str] = None):
        """
        Run the end-to-end flow.
        
        Args:
            resume_from: Step to resume from.
                - 'multiview' / 'preparation': resume from a Stage 0 step
                - 'foreground' / 'background' / 'align' / 'render': skip Stage 0 and start Stage 1
                - 'video': skip Stage 0 and Stage 1, then run only Stage 2
            trajectory_json: Trajectory JSON file for Stage 1 rendering.
            trajectory_name: Trajectory subdirectory name for Stage 2 video generation.
        """
        logger.info("=" * 60)
        logger.info("🚀 Full pipeline: end-to-end")
        logger.info("=" * 60)
        logger.info(f"Scene: {self.config.get('project.name', 'default')}")
        if resume_from:
            logger.info(f"Resume mode: starting from '{resume_from}'")
        if trajectory_json:
            logger.info(f"Using trajectory: {trajectory_json}")
        logger.info("=" * 60)
        
        # If trajectory_name is not specified but trajectory_json is provided,
        # derive it from the JSON filename for use across the flow.
        if not trajectory_name and trajectory_json:
            from pathlib import Path
            trajectory_name = Path(trajectory_json).stem
            logger.info(f"Derived trajectory name from trajectory_json: {trajectory_name}")
        
        # Save the actual trajectory name for the final output summary.
        self._actual_trajectory_name = trajectory_name
        
        # Select starting stage and step.
        if resume_from is None:
            # Start from Stage 0.
            run_stage_0 = True
            run_stage_1 = True
            run_stage_2 = True
            stage_0_resume_from = None
            stage_1_resume_from = None
        elif resume_from in self.STAGE_0_STEPS:
            # Start from a Stage 0 step.
            run_stage_0 = True
            run_stage_1 = True
            run_stage_2 = True
            stage_0_resume_from = resume_from
            stage_1_resume_from = None
        elif resume_from in self.STAGE_1_STEPS:
            # Skip Stage 0 and start from a Stage 1 step.
            run_stage_0 = False
            run_stage_1 = True
            run_stage_2 = True
            stage_1_resume_from = resume_from
        elif resume_from in self.STAGE_2_STEPS:
            # Skip Stage 0 and Stage 1; run only Stage 2.
            run_stage_0 = False
            run_stage_1 = False
            run_stage_2 = True
            stage_1_resume_from = None
        else:
            raise ValueError(
                f"Invalid step name: {resume_from}\n"
                f"Choices: {', '.join(self.ALL_STEPS)}"
            )
        
        # Run Stage 0.
        if run_stage_0:
            logger.info(f"\n{'#'*60}")
            logger.info(f"# Stage 0: data preparation")
            logger.info(f"{'#'*60}\n")
            
            stage_0 = Stage0Pipeline(self.config)
            stage_0.run(resume_from=stage_0_resume_from if resume_from in self.STAGE_0_STEPS else None)
        else:
            logger.info("\n⏭️  Skipping Stage 0")
        
        # Run Stage 1.
        if run_stage_1:
            logger.info(f"\n{'#'*60}")
            logger.info(f"# Stage 1: point-cloud processing + rendering")
            logger.info(f"{'#'*60}\n")
            
            stage_1 = Stage1Pipeline(self.config)
            stage_1.run(
                resume_from=stage_1_resume_from,
                trajectory_json=trajectory_json
            )
        else:
            logger.info("\n⏭️  Skipping Stage 1")
        
        # Run Stage 2.
        logger.info(f"\n{'#'*60}")
        logger.info(f"# Stage 2: video generation")
        logger.info(f"{'#'*60}\n")
        
        # trajectory_name has already been resolved above.
        stage_2 = Stage2VideoPipeline(self.config)
        stage_2.run(trajectory_name=trajectory_name)
        
        # Finish.
        logger.info("\n" + "=" * 60)
        logger.info("🎉 Full pipeline completed.")
        logger.info("=" * 60)
        self._print_outputs()
    
    def _print_outputs(self):
        """Print output file locations."""
        scene_name = self.config.get('project.name', 'default')
        output_root = self.config.get('project.output_root', 'outputs')
        
        # Use the actual trajectory name if one was provided; otherwise use the config default.
        if hasattr(self, '_actual_trajectory_name') and self._actual_trajectory_name:
            trajectory_name = self._actual_trajectory_name
        else:
            arc_type = self.config.get('stage_1.rendering.arc_type', 'yaw')
            arc_angle = self.config.get('stage_1.rendering.arc_angle', 90)
            arc_radius_scale = self.config.get('stage_1.rendering.arc_radius_scale', 1.0)
            trajectory_name = format_arc_trajectory_name(arc_type, arc_angle, arc_radius_scale)
        
        logger.info("\n📁 Output files:")
        logger.info(f"  - Multiview: {output_root}/multiview/{scene_name}")
        logger.info(f"  - Point clouds: {output_root}/prepared/{scene_name}")
        logger.info(f"  - Rendering: {output_root}/rendering/{scene_name}/{trajectory_name}")
        logger.info(f"  - Video: {output_root}/rendering/{scene_name}/{trajectory_name}/inference/output_video.mp4")


if __name__ == '__main__':
    print("🧪 Testing full pipeline...")
    
    from utils.config import Config
    
    config = Config()
    pipeline = FullPipeline(config)
    
    print(f"\n✅ Pipeline created successfully")
    print(f"   - Scene: {config.get('project.name', 'default')}")
    
    print(f"\nComplete flow:")
    print(f"  Stage 0: {' -> '.join(pipeline.STAGE_0_STEPS)}")
    print(f"  Stage 1: {' -> '.join(pipeline.STAGE_1_STEPS)}")
    
    print(f"\nSupported resume_from steps:")
    print(f"  {', '.join(pipeline.ALL_STEPS)}")
