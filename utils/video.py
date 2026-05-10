"""Video reading, writing, and concatenation utilities."""

import os
import cv2
import subprocess
import numpy as np
from PIL import Image
from typing import List
from pathlib import Path


# ============================================================================
# Video reading
# ============================================================================

def read_gif(input_path: str, n_frames: int = None) -> List[Image.Image]:
    """Read a GIF file and return a list of RGB PIL Images."""
    from PIL import ImageSequence

    frames = []
    video = Image.open(input_path)

    for img in ImageSequence.Iterator(video):
        frames.append(img.convert("RGB"))
        if n_frames is not None and len(frames) == n_frames:
            break

    return frames


def read_mp4(input_path: str, n_frames: int = None) -> List[Image.Image]:
    """Read an MP4 file and return a list of RGB PIL Images."""
    frames = []
    vidcap = cv2.VideoCapture(input_path)

    success, image = vidcap.read()
    while success:
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(rgb_image))

        if n_frames is not None and len(frames) == n_frames:
            break

        success, image = vidcap.read()

    vidcap.release()
    return frames


def read_video(input_path: str, n_frames: int = None) -> List[Image.Image]:
    """Auto-detect format (.gif/.mp4) and read video frames."""
    input_path = str(input_path)

    if input_path.endswith('.gif'):
        return read_gif(input_path, n_frames)
    elif input_path.endswith('.mp4'):
        return read_mp4(input_path, n_frames)
    else:
        raise ValueError(f"Unsupported video format: {input_path}")


# ============================================================================
# Video writing
# ============================================================================

def images_to_video(image_dir: str,
                   output_path: str,
                   fps: int = 10,
                   pattern: str = '*.png',
                   quality: int = 5) -> bool:
    """Convert an image sequence to an MP4 video using FFmpeg.

    Args:
        quality: Video quality (0-10, 0 = best, 10 = worst).
    """
    image_dir = Path(image_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    image_files = sorted(image_dir.glob(pattern))

    if len(image_files) == 0:
        print(f"Warning: no image files found: {image_dir}/{pattern}")
        return False

    try:
        first_file = image_files[0]
        file_ext = first_file.suffix

        cmd = [
            'ffmpeg',
            '-y',
            '-framerate', str(fps),
            '-pattern_type', 'glob',
            '-i', str(image_dir / pattern),
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-crf', str(18 + quality * 3),
            str(output_path)
        ]

        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        if result.returncode != 0:
            print(f"FFmpeg error: {result.stderr}")
            return False

        return True

    except Exception as e:
        print(f"Video generation failed: {e}")
        return False


def images_to_gif(image_dir: str,
                  output_path: str,
                  fps: int = 10,
                  pattern: str = "*.png",
                  rgba: bool = False) -> bool:
    """Convert an image sequence to a GIF. Supports RGBA for transparent backgrounds."""
    image_dir = Path(image_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    image_files = sorted(image_dir.glob(pattern))
    if not image_files:
        print(f"Warning: no images found: {image_dir}/{pattern}")
        return False

    use_unchanged = rgba
    frames = []
    for i, f in enumerate(image_files):
        if use_unchanged:
            img = cv2.imread(str(f), cv2.IMREAD_UNCHANGED)
        else:
            img = cv2.imread(str(f))
        if img is None:
            continue
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if img.shape[-1] == 4:
            use_unchanged = True
        frames.append(img)

    if not frames:
        return False

    duration_sec = 1.0 / fps if fps > 0 else 0.1

    if use_unchanged and frames[0].shape[-1] == 4:
        return _write_gif_rgba(frames, output_path, duration_sec)
    else:
        for i in range(len(frames)):
            if frames[i].shape[-1] == 4:
                frames[i] = frames[i][:, :, :3]
            frames[i] = cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB)
        try:
            import imageio.v3 as iio
        except ImportError:
            import imageio as iio
        iio.imwrite(output_path, frames, duration=duration_sec, loop=0)
        return True


def _write_gif_rgba(frames_bgra: list, output_path: Path, duration_sec: float) -> bool:
    """Write BGRA frames as a transparent GIF using PIL."""
    from PIL import Image
    rgbs = []
    alphas = []
    for bgra in frames_bgra:
        rgb = cv2.cvtColor(bgra[:, :, :3], cv2.COLOR_BGR2RGB)
        alpha = bgra[:, :, 3]
        rgb[alpha == 0] = [0, 0, 0]
        rgbs.append(rgb)
        alphas.append(alpha)
    first = Image.fromarray(rgbs[0], "RGB").convert("P", palette=Image.ADAPTIVE, colors=256)
    palette = first.getpalette()
    if len(palette) < 256 * 3:
        palette = palette + [0] * (256 * 3 - len(palette))
    trans_idx = 0
    for i in range(256):
        if palette[3 * i : 3 * i + 3] == [0, 0, 0]:
            trans_idx = i
            break
    pil_frames = []
    for i, (rgb, alpha) in enumerate(zip(rgbs, alphas)):
        if i == 0:
            pimg = first
        else:
            pimg = Image.fromarray(rgb, "RGB").quantize(palette=first, dither=0)
        arr = np.array(pimg)
        arr[alpha == 0] = trans_idx
        pimg = Image.fromarray(arr, mode="P")
        pimg.putpalette(palette)
        pimg.info["disposal"] = 2
        pil_frames.append(pimg)
    pil_frames[0].save(
        output_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=int(duration_sec * 1000),
        loop=0,
        transparency=trans_idx,
    )
    return True


def concat_videos(video_paths: List[str],
                 output_path: str,
                 fps: int = 10) -> bool:
    """Concatenate multiple videos into one using FFmpeg."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        concat_file = f.name
        for video_path in video_paths:
            f.write(f"file '{os.path.abspath(video_path)}'\n")

    try:
        cmd = [
            'ffmpeg',
            '-y',
            '-f', 'concat',
            '-safe', '0',
            '-i', concat_file,
            '-c', 'copy',
            str(output_path)
        ]

        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        if result.returncode != 0:
            print(f"FFmpeg error: {result.stderr}")
            return False

        return True

    finally:
        os.remove(concat_file)
