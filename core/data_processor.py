"""Data preprocessing pipeline for multi-view images."""

import os
import cv2
import numpy as np
import rembg
from PIL import Image
from pathlib import Path
from typing import Tuple, Dict, Any, Optional, List
from tqdm import tqdm

from utils.file_io import ensure_dir


class DataProcessor:
    """Preprocessor for inverse-transforming, re-matting, and cropping multi-view frames."""

    def __init__(self,
                 output_height: int = 480,
                 output_width: int = 832,
                 overwrite: bool = True):
        """
        Args:
            output_height: Output image height.
            output_width: Output image width.
            overwrite: Whether to overwrite existing output directory.
        """
        self.output_height = output_height
        self.output_width = output_width
        self.overwrite = overwrite
        self.rembg_session = None

    def _init_rembg(self):
        """Initialize rembg session."""
        if self.rembg_session is None:
            self.rembg_session = rembg.new_session()

    def simple_inverse_transform(self,
                                 frame_576: Image.Image) -> Tuple[Image.Image, np.ndarray]:
        """Approximate inverse transform: 576x576 -> target size with rembg re-matting.

        Args:
            frame_576: 576x576 input image.

        Returns:
            Tuple of (RGB image, binary mask).
        """
        side_len = int(self.output_height * 0.9)

        # Step 1: 576x576 -> side_len x side_len
        upscaled = frame_576.resize((side_len, side_len), Image.LANCZOS)
        upscaled_arr = np.array(upscaled)

        # Step 2: center on target-size canvas
        canvas = np.ones((self.output_height, self.output_width, 3), dtype=np.uint8) * 255
        start_y = (self.output_height - side_len) // 2
        start_x = (self.output_width - side_len) // 2

        canvas[start_y:start_y+side_len, start_x:start_x+side_len] = upscaled_arr
        canvas_pil = Image.fromarray(canvas)

        # Step 3: rembg hard-edge matting
        alpha_mask = rembg.remove(
            canvas_pil,
            session=self.rembg_session,
            only_mask=True,
            alpha_matting=False
        )
        alpha_mask = np.array(alpha_mask)

        # Binarize
        _, alpha_mask = cv2.threshold(alpha_mask, 127, 255, cv2.THRESH_BINARY)

        # Composite onto white background
        result_rgb = canvas.copy()
        result_rgb[alpha_mask < 127] = 255
        result_rgb = Image.fromarray(result_rgb)

        return result_rgb, alpha_mask

    def smart_crop_and_resize(self,
                              image: Any,
                              height: int,
                              width: int) -> Any:
        """Center-crop and resize image to target dimensions.

        Args:
            image: Input image (PIL or ndarray).
            height: Target height.
            width: Target width.

        Returns:
            Cropped and resized image (same type as input).
        """
        is_pil = isinstance(image, Image.Image)
        image_arr = np.array(image) if is_pil else image
        is_mask = len(image_arr.shape) == 2

        if is_mask:
            image_arr = image_arr[:, :, np.newaxis]

        image_height, image_width = image_arr.shape[:2]

        if image_height == height and image_width == width:
            return image if is_pil else (image_arr if not is_mask else image_arr[:, :, 0])

        if image_height / image_width < height / width:
            cropped_width = int(image_height / height * width)
            left = (image_width - cropped_width) // 2
            image_arr = image_arr[:, left:left + cropped_width]
        else:
            cropped_height = int(image_width / width * height)
            top = (image_height - cropped_height) // 2
            image_arr = image_arr[top:top + cropped_height, :]

        if is_mask:
            return cv2.resize(image_arr[:, :, 0], (width, height), interpolation=cv2.INTER_NEAREST)
        else:
            result_pil = Image.fromarray(image_arr).resize((width, height), Image.LANCZOS)
            return result_pil if is_pil else np.array(result_pil)

    def process_frame(self,
                     frame_idx: int,
                     input_dir: Path,
                     output_dir: Path,
                     view_dirs: List[Path]) -> None:
        """Process a single frame across all views.

        Args:
            frame_idx: Frame index.
            input_dir: Input directory.
            output_dir: Output directory.
            view_dirs: List of view directories.
        """
        frame_name = f"{frame_idx:05d}"

        frame_dir = output_dir / frame_name
        images_dir = frame_dir / "images"
        masks_dir = frame_dir / "masks"

        ensure_dir(images_dir)
        ensure_dir(masks_dir)

        # Process multi-view images
        for view_idx, view_dir in enumerate(view_dirs):
            view_image_path = view_dir / f"{frame_name}.png"
            if not view_image_path.exists():
                raise FileNotFoundError(f"Multi-view image not found: {view_image_path}")

            view_image = Image.open(view_image_path)

            view_rgb, mask = self.simple_inverse_transform(view_image)

            view_rgb.save(images_dir / f"{frame_name}_{view_idx}.png")
            cv2.imwrite(str(masks_dir / f"{frame_name}_{view_idx}_mask.png"), mask)

        # Process original image
        original_img = Image.open(input_dir / "downsampled" / "original" / f"{frame_name}.png").convert('RGB')
        original_mask = np.ones((original_img.height, original_img.width), dtype=np.uint8) * 255

        original_cropped = self.smart_crop_and_resize(original_img, self.output_height, self.output_width)
        original_mask_cropped = self.smart_crop_and_resize(original_mask, self.output_height, self.output_width)

        original_cropped.save(images_dir / f"{frame_name}_original.png")
        cv2.imwrite(str(masks_dir / f"{frame_name}_original_mask.png"), original_mask_cropped)

        # Process foreground image
        foreground_img = Image.open(input_dir / "downsampled" / "object" / f"{frame_name}.png").convert('RGB')
        foreground_mask_img = cv2.imread(str(input_dir / "downsampled" / "mask" / f"{frame_name}.png"), cv2.IMREAD_UNCHANGED)

        if len(foreground_mask_img.shape) == 3:
            foreground_mask = (foreground_mask_img[:, :, 3] if foreground_mask_img.shape[2] == 4
                             else foreground_mask_img[:, :, 0])
        else:
            foreground_mask = foreground_mask_img

        if foreground_mask.max() < 255:
            foreground_mask = (foreground_mask.astype(np.float32) / foreground_mask.max() * 255).astype(np.uint8)

        foreground_cropped = self.smart_crop_and_resize(foreground_img, self.output_height, self.output_width)
        foreground_mask_cropped = self.smart_crop_and_resize(foreground_mask, self.output_height, self.output_width)

        foreground_cropped.save(images_dir / f"{frame_name}_foreground.png")
        cv2.imwrite(str(masks_dir / f"{frame_name}_foreground_mask.png"), foreground_mask_cropped)
        cv2.imwrite(str(masks_dir / f"{frame_name}_background_mask.png"), 255 - foreground_mask_cropped)

    def process_scene(self,
                     input_dir: str,
                     output_dir: str,
                     num_frames: Optional[int] = None):
        """Process a complete scene.

        Args:
            input_dir: Input directory (output of step 0).
            output_dir: Output directory.
            num_frames: Number of frames to process (None for all).
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)

        print(f"Data preparation: input={input_dir}, output={output_dir}")

        if output_dir.exists():
            if self.overwrite:
                import shutil
                shutil.rmtree(output_dir)
            else:
                raise FileExistsError(f"Output directory already exists: {output_dir}")

        ensure_dir(output_dir)

        self._init_rembg()

        multiview_dir = input_dir / "multiview_images"
        view_dirs = sorted([d for d in multiview_dir.iterdir()
                          if d.is_dir() and d.name.startswith('v')])

        if not view_dirs:
            raise ValueError(f"No view directories found in: {multiview_dir}")

        n_frames = len(list(view_dirs[0].glob("*.png")))

        if num_frames is not None:
            n_frames = min(n_frames, num_frames)

        print(f"Found {len(view_dirs)} views, {n_frames} frames, output size: {self.output_width}x{self.output_height}")

        for frame_idx in tqdm(range(n_frames), desc="Processing frames"):
            try:
                self.process_frame(frame_idx, input_dir, output_dir, view_dirs)
            except Exception as e:
                print(f"Failed to process frame {frame_idx:05d}: {e}")
                raise

        print(f"Done. Processed {n_frames} frames, {len(view_dirs)} views -> {output_dir}")

    @classmethod
    def from_config(cls, config, overwrite: bool = True):
        """Create processor from config."""
        from utils.config import Config
        if not isinstance(config, Config):
            config = Config(config)

        return cls(
            output_height=config.get('stage_0.preparation.output_height', 480),
            output_width=config.get('stage_0.preparation.output_width', 832),
            overwrite=overwrite
        )
