"""
Stage 1B pipeline orchestration: rendering.

Flow:
  1_4 rendering with two trajectory modes

Trajectory modes:
  1. Automatic mode (default): generate an arc trajectory
  2. Manual mode: use a pre-generated trajectory JSON from 1_3

Input:
  - outputs/prepared/ point-cloud data
  - Optional trajectory JSON

Output:
  - outputs/rendering/{scene}/{trajectory_name}/
"""

from pathlib import Path
from typing import Optional

from utils.config import Config
from utils.logging import setup_logger
from utils.file_io import load_json
from utils.trajectory import format_arc_trajectory_name
from core.rendering import PointCloudRenderer

logger = setup_logger('stage_1b_rendering')


class Stage1BRenderingPipeline:
    """Stage 1B: rendering."""
    
    def __init__(self, config: Config):
        """
        Args:
            config: Configuration object.
        """
        self.config = config
        self.data_dir = Path(config.get('project.output_prepared'))
    
    def run(self, trajectory_json: Optional[str] = None):
        """
        Run the rendering flow.
        
        Args:
            trajectory_json: 
                - None: generate an arc trajectory automatically (default)
                - "path/to/trajectory.json": use a pre-generated trajectory
        """
        logger.info("=" * 60)
        logger.info("Stage 1B: rendering")
        logger.info("=" * 60)
        
        # Select trajectory mode.
        if trajectory_json is None:
            mode = "automatic mode (arc trajectory)"
            trajectory_source = self._auto_generate_trajectory()
        else:
            mode = "manual mode (pre-generated trajectory)"
            trajectory_source = self._validate_trajectory_json(trajectory_json)
        
        logger.info(f"Trajectory mode: {mode}")
        logger.info(f"Trajectory file: {trajectory_source}")
        
        # Render.
        logger.info("\nStarting rendering...")
        renderer = PointCloudRenderer.from_config(self.config)
        
        output_dir = renderer.render_trajectory(
            data_dir=str(self.data_dir),
            trajectory_json=str(trajectory_source)
        )
        
        logger.info("\n" + "=" * 60)
        logger.info("🎉 Rendering completed.")
        logger.info("=" * 60)
        logger.info(f"📁 Output directory: {output_dir}")
        logger.info(f"📹 Video files:")
        logger.info(f"  - {output_dir}/videos/rendered_images.mp4")
        logger.info(f"  - {output_dir}/videos/rendered_depths.mp4")
        logger.info(f"\n💡 Inference files for video generation:")
        logger.info(f"  - {output_dir}/inference/")
    
    def _auto_generate_trajectory(self) -> Path:
        """
        Generate an arc trajectory automatically.
        
        Returns:
            Path to the generated trajectory JSON file.
        """
        logger.info("\n🔄 Generating arc trajectory automatically...")
        
        # Read parameters from config.
        arc_type = self.config.get('stage_1.rendering.arc_type', 'yaw')
        arc_angle = self.config.get('stage_1.rendering.arc_angle', 90)
        arc_radius_scale = self.config.get('stage_1.rendering.arc_radius_scale', 1.0)
        
        logger.info(f"  - Arc type: {arc_type}")
        logger.info(f"  - Arc angle: {arc_angle}°")
        logger.info(f"  - Radius scale: {arc_radius_scale}")
        logger.info(f"  - Radius and elevation: computed from point-cloud data")
        
        # Build trajectory file path with type and angle.
        trajectory_name = format_arc_trajectory_name(arc_type, arc_angle, arc_radius_scale)
        trajectory_path = self.data_dir / f"{trajectory_name}.json"
        
        # Always regenerate to keep frame count in sync with the current data.
        if trajectory_path.exists():
            logger.info(f"⚠️  Removing old trajectory: {trajectory_path}")
            trajectory_path.unlink()
        
        # Generate new trajectory.
        logger.info(f"✓ Generating new trajectory...")
        renderer = PointCloudRenderer.from_config(self.config)
        
        trajectory_json = renderer.generate_arc_trajectory(
            data_dir=str(self.data_dir),
            arc_type=arc_type,
            arc_angle=arc_angle,
            num_frames=None,  # Auto-detect.
            save_json_path=trajectory_path.name,
            arc_radius_scale=arc_radius_scale
        )
        
        logger.info(f"✓ Trajectory generated: {trajectory_json}")
        return Path(trajectory_json)
    
    def _validate_trajectory_json(self, trajectory_json: str) -> Path:
        """
        Validate a manually provided trajectory JSON.
        
        Args:
            trajectory_json: Relative or absolute trajectory file path.
        
        Returns:
            Validated path.
        """
        logger.info(f"\n✓ Using manual trajectory: {trajectory_json}")
        
        # Convert to a Path object.
        traj_path = Path(trajectory_json)
        
        # Resolve relative paths against data_dir.
        if not traj_path.is_absolute():
            traj_path = self.data_dir / traj_path
        
        # Validate file existence.
        if not traj_path.exists():
            raise FileNotFoundError(
                f"Trajectory file does not exist: {traj_path}\n"
                f"💡 Tips:\n"
                f"  1. Check that the file path is correct\n"
                f"  2. Or generate a trajectory with 1_3_cam_traj.py:\n"
                f"     python scripts/1_3_cam_traj.py --data_dir {self.data_dir}"
            )
        
        # Validate JSON format.
        try:
            data = load_json(str(traj_path))
            
            if 'camera_path' not in data:
                raise ValueError("Trajectory JSON is missing the 'camera_path' field")
            
            logger.info(f"  - Trajectory frames: {len(data['camera_path'])}")
            
        except Exception as e:
            raise ValueError(f"Invalid trajectory JSON format: {e}")
        
        return traj_path


if __name__ == '__main__':
    print("🧪 Testing Stage 1B pipeline...")
    
    from utils.config import Config
    
    config = Config()
    pipeline = Stage1BRenderingPipeline(config)
    
    print(f"\n✅ Pipeline created successfully")
    print(f"   - Data directory: {pipeline.data_dir}")
    print(f"\nSupported trajectory modes:")
    print(f"   1. Automatic mode: generate arc trajectory (default)")
    print(f"   2. Manual mode: use pre-generated JSON")
    print(f"\nDefault arc parameters:")
    print(f"   - Angle: {config.get('stage_1.rendering.arc_angle')}°")
    print(f"   - Radius/elevation: computed from point cloud")
