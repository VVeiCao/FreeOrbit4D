"""Multi-view generation module using SV4D."""

import os
import sys
import numpy as np
from pathlib import Path
from contextlib import contextmanager
from typing import Optional, List, Dict, Any, Tuple


class MultiViewGenerator:
    """SV4D-based multi-view video generator with automatic environment management."""

    def __init__(self,
                 model_type: str = 'sv4d2',
                 num_steps: int = 50,
                 seed: int = 23,
                 device: str = 'cuda',
                 target_size: int = 576,
                 image_frame_ratio: float = 0.9,
                 encoding_t: int = 8,
                 decoding_t: int = 4,
                 fps: int = 10):
        """
        Args:
            model_type: Model type ('sv4d2' for 4 views or 'sv4d2_8views' for 8 views).
            num_steps: Diffusion sampling steps.
            seed: Random seed.
            device: Compute device.
            target_size: SV4D input size (default 576).
            image_frame_ratio: Object-to-frame ratio (default 0.9).
            encoding_t: Encoding batch size.
            decoding_t: Decoding batch size.
            fps: Video frame rate.
        """
        self.model_type = model_type
        self.num_steps = num_steps
        self.seed = seed
        self.device = device
        self.target_size = target_size
        self.image_frame_ratio = image_frame_ratio
        self.encoding_t = encoding_t
        self.decoding_t = decoding_t
        self.fps = fps
        self.model = None

        self.original_cwd = os.getcwd()
        self.generative_models_path = Path(__file__).parent.parent / "generative-models"

    @contextmanager
    def _environment_context(self):
        """Context manager that temporarily switches cwd and sys.path to generative-models/."""
        saved_cwd = os.getcwd()
        saved_path = sys.path.copy()

        try:
            if not self.generative_models_path.exists():
                raise FileNotFoundError(
                    f"generative-models directory not found: {self.generative_models_path}"
                )

            os.chdir(str(self.generative_models_path))

            if str(self.generative_models_path) not in sys.path:
                sys.path.append(str(self.generative_models_path))

            yield

        finally:
            os.chdir(saved_cwd)
            sys.path = saved_path

    def load_model(self):
        """Load the SV4D model.

        Returns:
            The loaded model object.
        """
        with self._environment_context():
            from scripts.demo.sv4d_helpers import load_model as _load_model

            model_path = Path(self.original_cwd) / "checkpoints" / "sv4d" / f"{self.model_type}.safetensors"

            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")

            print(f"Loading {self.model_type} model...")

            if self.model_type == "sv4d2":
                model_config = "scripts/sampling/configs/sv4d2.yaml"
            elif self.model_type == "sv4d2_8views":
                model_config = "scripts/sampling/configs/sv4d2_8views.yaml"
            else:
                raise ValueError(f"Unknown model type: {self.model_type}")

            self.model, _ = _load_model(
                model_config,
                self.device,
                num_frames=12,
                num_steps=self.num_steps,
                verbose=False,
                model_path=str(model_path)
            )

            print("SV4D model loaded.")

        return self.model

    def batch_process(self,
                     input_images_dir: str,
                     input_masks_dir: str,
                     output_folder: str,
                     num_frames: Optional[int] = None):
        """Generate multi-view images via SV4D inference.

        Due to the complexity of the full SV4D inference pipeline (sampling windows,
        inference loops, image matrix management), use scripts/0_0_gen_multiviews.py directly.

        Args:
            input_images_dir: Input images directory.
            input_masks_dir: Input masks directory.
            output_folder: Output directory.
            num_frames: Number of frames to process (None for all).
        """
        input_images_dir = str(Path(input_images_dir).resolve())
        input_masks_dir = str(Path(input_masks_dir).resolve())
        output_folder = str(Path(output_folder).resolve())

        print(f"Multi-view generation: images={input_images_dir}, masks={input_masks_dir}, "
              f"output={output_folder}, model={self.model_type}")

        with self._environment_context():
            import torch
            from fire import Fire

            raise NotImplementedError(
                "The full SV4D inference logic is too complex for this wrapper.\n"
                "Use the original script directly:\n\n"
                "python scripts/0_0_gen_multiviews.py \\\n"
                f"    --input_images_dir {input_images_dir} \\\n"
                f"    --input_masks_dir {input_masks_dir} \\\n"
                f"    --output_folder {output_folder} \\\n"
                f"    --model_type {self.model_type}\n\n"
                "The environment context manager is ready for future full implementation."
            )

    @classmethod
    def from_config(cls, config):
        """Create generator from config."""
        from utils.config import Config
        if not isinstance(config, Config):
            config = Config(config)

        return cls(
            model_type=config.get('stage_0.multiview.model_type', 'sv4d2'),
            num_steps=config.get('stage_0.multiview.num_steps', 50),
            seed=config.get('common.seed', 23),
            device=config.get('common.device', 'cuda'),
            target_size=config.get('stage_0.multiview.target_size', 576),
            image_frame_ratio=config.get('stage_0.multiview.image_frame_ratio', 0.9),
            encoding_t=config.get('stage_0.multiview.encoding_t', 8),
            decoding_t=config.get('stage_0.multiview.decoding_t', 4),
            fps=config.get('stage_0.multiview.fps', 10)
        )
