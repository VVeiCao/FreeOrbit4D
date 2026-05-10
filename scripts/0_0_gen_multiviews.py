#!/usr/bin/env python3
"""Step 0: Multi-view video generation using SV4D."""

import os
import sys
from glob import glob
from typing import List, Optional, Dict, Tuple, Any
from pathlib import Path

from tqdm import tqdm

ORIGINAL_CWD = os.getcwd()
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

generative_models_path = os.path.realpath(os.path.join(os.path.dirname(__file__), "../generative-models"))
sys.path.append(generative_models_path)
os.chdir(generative_models_path)

import numpy as np
import torch
import cv2
from fire import Fire
from PIL import Image
import imageio

from scripts.demo.sv4d_helpers import (
    load_model,
    read_video,
    run_img2vid,
)
from sgm.modules.encoders.modules import VideoPredictionEmbedderWithEncoder
from utils.seed import set_global_seed

VAE_FACTOR = 8
LATENT_CHANNELS = 4
DEFAULT_IMAGE_SIZE = 576
DEFAULT_NUM_STEPS = 50
DEFAULT_SEED = 23
DEFAULT_ENCODING_T = 8
DEFAULT_DECODING_T = 4
DEFAULT_FPS = 10
DEFAULT_IMAGE_FRAME_RATIO = 0.85

IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']


def read_gif(input_path, n_frames):
    from PIL import ImageSequence
    frames = []
    video = Image.open(input_path)
    for img in ImageSequence.Iterator(video):
        frames.append(img.convert("RGB"))
        if len(frames) == n_frames:
            break
    return frames


def read_mp4(input_path, n_frames):
    frames = []
    vidcap = cv2.VideoCapture(input_path)
    success, image = vidcap.read()
    while success:
        frames.append(Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)))
        success, image = vidcap.read()
        if len(frames) == n_frames:
            break
    return frames


def preprocess_video(
    input_path,
    mask_dir,
    n_frames=21,
    W=576,
    H=576,
    output_folder=None,
    image_frame_ratio=0.9,
    base_count=0,
):
    """Preprocess video: dynamic object-center tracking, crop and resize to 576x576."""
    if output_folder is None:
        output_folder = os.path.dirname(input_path)

    path = Path(input_path)
    is_video_file = False
    all_img_paths = []

    if path.is_file():
        if any([input_path.endswith(x) for x in [".gif", ".mp4"]]):
            is_video_file = True
        else:
            raise ValueError("Path is not a valid video file.")
    elif path.is_dir():
        all_img_paths = sorted([
            f for f in path.iterdir()
            if f.is_file() and f.suffix.lower() in [".jpg", ".jpeg", ".png"]
        ])[:n_frames]
    elif "*" in input_path:
        all_img_paths = sorted(glob(input_path))[:n_frames]
    else:
        raise ValueError(f"Invalid input path: {input_path}")

    if is_video_file and input_path.endswith(".gif"):
        images = read_gif(input_path, n_frames)[:n_frames]
    elif is_video_file and input_path.endswith(".mp4"):
        images = read_mp4(input_path, n_frames)[:n_frames]
    else:
        images = [Image.open(img_path) for img_path in all_img_paths]

    if len(images) != n_frames:
        raise ValueError(f"Input contains {len(images)} frames, expected {n_frames} frames.")

    images_v0 = []
    sample_image_arr = np.array(images[0])
    in_h, in_w = sample_image_arr.shape[:2]

    frame_centers = []
    max_size = 0

    mask_files = sorted([
        f for f in Path(mask_dir).iterdir()
        if f.is_file() and f.suffix.lower() in [".jpg", ".jpeg", ".png"]
    ])[:n_frames]

    if len(mask_files) != len(images):
        raise ValueError(f"Mask count ({len(mask_files)}) != image count ({len(images)})")

    for idx, image in enumerate(images):
        mask = cv2.imread(str(mask_files[idx]), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Cannot read mask file: {mask_files[idx]}")

        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        x, y, w, h = cv2.boundingRect(mask)
        center_x = x + w // 2
        center_y = y + h // 2
        frame_centers.append((center_x, center_y))
        max_size = max(max_size, w, h)

    side_len = int(max_size / image_frame_ratio) if image_frame_ratio is not None else max_size

    for frame_idx, image in enumerate(images):
        image_arr = np.array(image.convert("RGB"))
        center_x_i, center_y_i = frame_centers[frame_idx]

        x_start = center_x_i - side_len // 2
        y_start = center_y_i - side_len // 2
        x_end = x_start + side_len
        y_end = y_start + side_len

        canvas = np.ones((side_len, side_len, 3), dtype=np.uint8) * 255

        src_x_start = max(0, x_start)
        src_y_start = max(0, y_start)
        src_x_end = min(in_w, x_end)
        src_y_end = min(in_h, y_end)

        dst_x_start = src_x_start - x_start
        dst_y_start = src_y_start - y_start
        dst_x_end = dst_x_start + (src_x_end - src_x_start)
        dst_y_end = dst_y_start + (src_y_end - src_y_start)

        if src_x_end > src_x_start and src_y_end > src_y_start:
            cropped_region = image_arr[src_y_start:src_y_end, src_x_start:src_x_end]
            canvas[dst_y_start:dst_y_end, dst_x_start:dst_x_end] = cropped_region

        final_576 = cv2.resize(canvas, (W, H), interpolation=cv2.INTER_LANCZOS4)
        images_v0.append(final_576)

    multiview_videos_dir = os.path.join(output_folder, "multiview_images", "multiview_videos")
    os.makedirs(multiview_videos_dir, exist_ok=True)
    processed_file = os.path.join(multiview_videos_dir, f"{base_count:06d}_input.mp4")
    imageio.mimwrite(processed_file, images_v0, fps=10)

    return processed_file


sv4d2_configs = {
    "sv4d2": {
        "T": 12,
        "V": 4,
        "model_config": "scripts/sampling/configs/sv4d2.yaml",
        "version_dict": {
            "T": 12 * 4,
            "options": {
                "discretization": 1,
                "cfg": 2.0,
                "min_cfg": 2.0,
                "num_views": 4,
                "sigma_min": 0.002,
                "sigma_max": 700.0,
                "rho": 7.0,
                "guider": 2,
                "force_uc_zero_embeddings": [
                    "cond_frames",
                    "cond_frames_without_noise",
                    "cond_view",
                    "cond_motion",
                ],
                "additional_guider_kwargs": {
                    "additional_cond_keys": ["cond_view", "cond_motion"]
                },
            },
        },
    },
    "sv4d2_8views": {
        "T": 5,
        "V": 8,
        "model_config": "scripts/sampling/configs/sv4d2_8views.yaml",
        "version_dict": {
            "T": 5 * 8,
            "options": {
                "discretization": 1,
                "cfg": 2.5,
                "min_cfg": 1.5,
                "num_views": 8,
                "sigma_min": 0.002,
                "sigma_max": 700.0,
                "rho": 7.0,
                "guider": 5,
                "force_uc_zero_embeddings": [
                    "cond_frames",
                    "cond_frames_without_noise",
                    "cond_view",
                    "cond_motion",
                ],
                "additional_guider_kwargs": {
                    "additional_cond_keys": ["cond_view", "cond_motion"]
                },
            },
        },
    },
}


def apply_mask_to_image(image_path: str, mask_path: str) -> np.ndarray:
    """Apply mask to extract foreground on white background."""
    original_img = cv2.imread(str(image_path))
    mask_img = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)

    if original_img is None or mask_img is None:
        raise ValueError(f"Cannot read image or mask: {image_path}, {mask_path}")

    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)

    if len(mask_img.shape) == 3:
        mask_gray = mask_img[:, :, 3] if mask_img.shape[2] == 4 else cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)
    else:
        mask_gray = mask_img

    if mask_gray.max() < 255:
        mask_gray = (mask_gray.astype(np.float32) / mask_gray.max() * 255).astype(np.uint8)

    if original_img.shape[:2] != mask_gray.shape:
        mask_gray = cv2.resize(mask_gray, (original_img.shape[1], original_img.shape[0]))

    mask_normalized = mask_gray.astype(np.float32) / 255.0
    result = original_img.astype(np.float32) * mask_normalized[:, :, None]
    result += 255.0 * (1 - mask_normalized[:, :, None])

    return np.clip(result, 0, 255).astype(np.uint8)


def _calculate_inference_count(n_frames: int, T: int = 12, S_max: int = 11) -> int:
    """Calculate number of inference passes: (I-1)*S_max + T >= n_frames."""
    if n_frames <= T:
        return 1
    inference_count = 2
    while ((inference_count - 1) * S_max + T) < n_frames:
        inference_count += 1
    return inference_count


def _generate_sampling_windows(n_frames: int, T: int = 12, S_max: int = 11) -> List[List[int]]:
    """Generate inference windows: first I-1 shift by S_max, last aligns to end."""
    inference_count = _calculate_inference_count(n_frames, T, S_max)
    windows = []
    if inference_count == 1:
        windows.append([0, T - 1])
    else:
        for i in range(inference_count - 1):
            start = i * S_max
            windows.append([start, start + T - 1])
        last_start = n_frames - T
        windows.append([last_start, n_frames - 1])
    return windows


def _calculate_window_overlap(windows: List[List[int]]) -> int:
    if len(windows) <= 1:
        return 0
    return windows[-2][1] - windows[-1][0] + 1


def _validate_windows_coverage(windows: List[List[int]], n_frames: int) -> bool:
    if not windows:
        return False
    if windows[0][0] != 0:
        return False
    if windows[-1][1] != n_frames - 1:
        return False
    for i in range(len(windows) - 1):
        if windows[i + 1][0] > windows[i][1] + 1:
            return False
    return True


def calculate_video_strategy(total_frames: int) -> Dict[str, Any]:
    """Calculate sampling strategy: stride=1 -> truncate to 45 -> inference windows -> adjust to 4k+1."""
    T = 12
    MAX_TRUNCATE = 45
    S_max = 11

    stride = 1
    sampled_n = total_frames
    truncated_n = min(sampled_n, MAX_TRUNCATE)

    inference_i = _calculate_inference_count(truncated_n, T, S_max)
    windows = _generate_sampling_windows(truncated_n, T, S_max)
    last_overlap = _calculate_window_overlap(windows)

    infer_output = truncated_n
    k = (infer_output - 1) // 4
    final_n = 4 * k + 1

    indices = [i * stride for i in range(final_n)]
    used_frames = (final_n - 1) * stride + 1
    loss_rate = (total_frames - used_frames) / total_frames

    _validate_windows_coverage(windows, truncated_n)

    return {
        "stride": stride,
        "sampled_n": sampled_n,
        "truncated_n": truncated_n,
        "final_n": final_n,
        "inference_i": inference_i,
        "windows": windows,
        "loss_rate": loss_rate,
        "indices": indices,
        "last_overlap": last_overlap,
    }


def process_and_downsample(
    input_images_dir: str,
    input_masks_dir: str,
    output_folder: str,
    num_frames: Optional[int] = None,
) -> Tuple[str, int, Dict[str, Any]]:
    """Apply mask to extract foreground + smart sampling to 4k+1 frames."""
    image_files = []
    for ext in IMAGE_EXTENSIONS:
        image_files.extend(glob(os.path.join(input_images_dir, f'*{ext}')))

    image_files = sorted(image_files)
    total_frames = len(image_files)

    if total_frames == 0:
        raise ValueError(f"No images found in {input_images_dir}")

    strategy = calculate_video_strategy(total_frames)
    sampled_indices = strategy["indices"]

    if num_frames is not None and num_frames < len(sampled_indices):
        sampled_indices = sampled_indices[:num_frames]

        T = 12
        S_max = 11
        truncated_n = num_frames
        inference_i = _calculate_inference_count(truncated_n, T, S_max)
        windows = _generate_sampling_windows(truncated_n, T, S_max)
        last_overlap = _calculate_window_overlap(windows)

        strategy["final_n"] = num_frames
        strategy["indices"] = sampled_indices
        strategy["truncated_n"] = truncated_n
        strategy["inference_i"] = inference_i
        strategy["windows"] = windows
        strategy["last_overlap"] = last_overlap

    actual_frames = len(sampled_indices)

    downsampled_dir = os.path.join(output_folder, "downsampled")
    downsampled_original_dir = os.path.join(downsampled_dir, "original")
    downsampled_object_dir = os.path.join(downsampled_dir, "object")
    downsampled_mask_dir = os.path.join(downsampled_dir, "mask")

    for dir_path in [downsampled_original_dir, downsampled_object_dir, downsampled_mask_dir]:
        os.makedirs(dir_path, exist_ok=True)

    for output_idx, input_idx in enumerate(tqdm(sampled_indices, desc="Processing frames")):
        image_path = image_files[input_idx]
        image_stem = os.path.splitext(os.path.basename(image_path))[0]

        mask_path = _find_mask_file(input_masks_dir, image_stem)

        original_image = Image.open(image_path).convert('RGB')
        original_image.save(os.path.join(downsampled_original_dir, f"{output_idx:05d}.png"))

        masked_image = apply_mask_to_image(image_path, mask_path)
        Image.fromarray(masked_image).save(os.path.join(downsampled_object_dir, f"{output_idx:05d}.png"))

        mask_img = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        mask_gray = _extract_gray_mask(mask_img)
        cv2.imwrite(os.path.join(downsampled_mask_dir, f"{output_idx:05d}.png"), mask_gray)

    return downsampled_object_dir, actual_frames, strategy


def _find_mask_file(masks_dir: str, image_stem: str) -> str:
    mask_path = os.path.join(masks_dir, f"{image_stem}.png")
    if os.path.exists(mask_path):
        return mask_path
    for ext in IMAGE_EXTENSIONS:
        alt_path = os.path.join(masks_dir, f"{image_stem}{ext}")
        if os.path.exists(alt_path):
            return alt_path
    raise FileNotFoundError(f"Mask file not found: {mask_path}")


def _extract_gray_mask(mask_img: np.ndarray) -> np.ndarray:
    if len(mask_img.shape) == 3:
        return cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)
    return mask_img


def _prepare_camera_params(
    sv4d2_model: str,
    elevations_deg: Any,
    azimuths_deg: Optional[List[float]],
    n_views: int,
) -> Tuple[List[float], np.ndarray, np.ndarray, np.ndarray]:
    if isinstance(elevations_deg, (float, int)):
        elevations_deg = [elevations_deg] * n_views

    assert len(elevations_deg) == n_views, \
        f"elevations_deg requires {n_views} values, got {len(elevations_deg)}"

    if azimuths_deg is None:
        azimuths_deg = (
            np.array([0, 60, 120, 180, 240])
            if sv4d2_model == "sv4d2"
            else np.array([0, 30, 75, 120, 165, 210, 255, 300, 330])
        )

    assert len(azimuths_deg) == n_views, \
        f"azimuths_deg requires {n_views} values, got {len(azimuths_deg)}"

    polars_rad = np.array([np.deg2rad(90 - e) for e in elevations_deg])
    azimuths_rad = np.array([np.deg2rad((a - azimuths_deg[-1]) % 360) for a in azimuths_deg])

    return elevations_deg, azimuths_deg, polars_rad, azimuths_rad


def _initialize_img_matrix(
    n_frames: int,
    n_views: int,
    images_v0: torch.Tensor,
    device: str,
    H: int,
    W: int,
) -> List[List[Optional[torch.Tensor]]]:
    images_t0 = torch.zeros(n_views, 3, H, W).float().to(device)
    subsampled_views = np.arange(n_views)

    img_matrix = [[None] * n_views for _ in range(n_frames)]

    for i, v in enumerate(subsampled_views):
        img_matrix[0][i] = images_t0[v].unsqueeze(0)

    for t in range(n_frames):
        img_matrix[t][0] = images_v0[t]

    return img_matrix


def generate_multiview_video(
    input_images_dir: str,
    model_path: str,
    output_folder: str,
    n_frames: int,
    strategy: Dict[str, Any],
    num_steps: int = DEFAULT_NUM_STEPS,
    img_size: int = DEFAULT_IMAGE_SIZE,
    seed: int = DEFAULT_SEED,
    encoding_t: int = DEFAULT_ENCODING_T,
    decoding_t: int = DEFAULT_DECODING_T,
    device: str = "cuda",
    elevations_deg: Optional[List[float]] = 0.0,
    azimuths_deg: Optional[List[float]] = None,
    image_frame_ratio: Optional[float] = DEFAULT_IMAGE_FRAME_RATIO,
    deterministic: bool = True,
    verbose: Optional[bool] = False,
) -> Dict[str, Any]:
    """Run SV4D to generate multi-view video: preprocess -> load model -> windowed inference -> return matrix."""
    import json

    model_name = os.path.splitext(os.path.basename(model_path))[0]
    assert model_name in sv4d2_configs, f"Unknown model: {model_name}"

    config = sv4d2_configs[model_name]
    T, V = config["T"], config["V"]
    model_config = config["model_config"]
    version_dict = config["version_dict"].copy()

    H, W = img_size, img_size
    n_views = V + 1

    version_dict.update({
        "H": H, "W": W,
        "C": LATENT_CHANNELS,
        "f": VAE_FACTOR,
        "options": {**version_dict["options"], "num_steps": num_steps}
    })

    set_global_seed(seed, deterministic=deterministic)

    mask_dir = os.path.join(output_folder, "downsampled", "mask")
    if not os.path.exists(mask_dir):
        raise FileNotFoundError(f"Mask directory not found: {mask_dir}")

    processed_input_path = preprocess_video(
        input_images_dir, mask_dir, n_frames=n_frames,
        W=W, H=H, output_folder=output_folder,
        image_frame_ratio=image_frame_ratio, base_count=0,
    )
    images_v0 = read_video(processed_input_path, n_frames=n_frames, device=device)

    elevations_deg, azimuths_deg, polars_rad, azimuths_rad = _prepare_camera_params(
        model_name, elevations_deg, azimuths_deg, n_views
    )

    img_matrix = _initialize_img_matrix(n_frames, n_views, images_v0, device, H, W)

    model, _ = load_model(model_config, device, version_dict["T"], num_steps, verbose, model_path)
    model.en_and_decode_n_samples_a_time = decoding_t
    for emb in model.conditioner.embedders:
        if isinstance(emb, VideoPredictionEmbedderWithEncoder):
            emb.en_and_decode_n_samples_a_time = encoding_t

    windows = strategy["windows"]
    t0_list = [w[0] for w in windows]

    v0 = 0
    view_indices = np.arange(V) + 1
    subsampled_views = np.arange(n_views)

    for idx, t0 in enumerate(tqdm(t0_list, desc="Sampling")):
        if t0 + T > n_frames:
            t0 = n_frames - T

        frame_indices = t0 + np.arange(T)
        image = img_matrix[t0][v0]
        cond_motion = torch.cat([img_matrix[t][v0] for t in frame_indices], 0)
        cond_view = torch.cat([img_matrix[t0][v] for v in view_indices], 0)

        polars = polars_rad[subsampled_views[1:]][None].repeat(T, 0).flatten()
        azims = azimuths_rad[subsampled_views[1:]][None].repeat(T, 0).flatten()
        polars = (polars - polars_rad[v0] + np.pi / 2) % (np.pi * 2)
        azims = (azims - azimuths_rad[v0]) % (np.pi * 2)

        samples = run_img2vid(
            version_dict, model, image, seed,
            polars, azims, cond_motion, cond_view,
            decoding_t, cond_mv=(t0 != 0),
        )
        samples = samples.view(T, V, 3, H, W)

        for i, t in enumerate(frame_indices):
            for j, v in enumerate(view_indices):
                img_matrix[t][v] = samples[i, j][None] * 2 - 1

    return {
        "img_matrix": img_matrix,
        "view_indices": view_indices,
        "n_frames": n_frames,
        "H": H,
        "W": W,
    }


def save_multiview_outputs(
    multiview_data: Dict[str, Any],
    output_folder: str,
    fps: int = DEFAULT_FPS,
) -> None:
    """Save multi-view image sequences and videos."""
    img_matrix = multiview_data["img_matrix"]
    view_indices = multiview_data["view_indices"]
    n_frames = multiview_data["n_frames"]

    multiview_images_dir = os.path.join(output_folder, "multiview_images")
    multiview_videos_dir = os.path.join(multiview_images_dir, "multiview_videos")
    os.makedirs(multiview_videos_dir, exist_ok=True)

    for v in view_indices:
        frames = [img_matrix[t][v] for t in range(n_frames) if img_matrix[t][v] is not None]

        img_grid = [
            (((img[0].permute(1, 2, 0) + 1) / 2).cpu().numpy() * 255.0).astype(np.uint8)
            for img in frames
        ]

        vid_file = os.path.join(multiview_videos_dir, f"000000_v{v:03d}.mp4")
        imageio.mimwrite(vid_file, img_grid, fps=fps)

        view_dir = os.path.join(multiview_images_dir, f"v{v:03d}")
        os.makedirs(view_dir, exist_ok=True)
        for idx, img_array in enumerate(img_grid):
            Image.fromarray(img_array).save(os.path.join(view_dir, f"{idx:05d}.png"))


def main(
    input_images_dir: str = "data/DAVIS/JPEGImages/480p/camel",
    input_masks_dir: str = "data/DAVIS/Annotations/480p/camel",
    output_folder: str = "outputs/multiview/camel",
    num_frames: Optional[int] = None,
    model_type: str = "sv4d2",
    model_path: Optional[str] = None,
    num_steps: int = DEFAULT_NUM_STEPS,
    img_size: int = DEFAULT_IMAGE_SIZE,
    seed: int = DEFAULT_SEED,
    encoding_t: int = DEFAULT_ENCODING_T,
    decoding_t: int = DEFAULT_DECODING_T,
    device: str = "cuda",
    elevations_deg: Optional[List[float]] = 0.0,
    azimuths_deg: Optional[List[float]] = None,
    image_frame_ratio: Optional[float] = DEFAULT_IMAGE_FRAME_RATIO,
    deterministic: bool = True,
    verbose: Optional[bool] = False,
    fps: int = DEFAULT_FPS,
    overwrite: bool = True,
) -> None:
    """Main pipeline: image sequence -> multi-view video."""
    if not os.path.isabs(input_images_dir):
        input_images_dir = os.path.abspath(os.path.join(ORIGINAL_CWD, input_images_dir))
    if not os.path.isabs(input_masks_dir):
        input_masks_dir = os.path.abspath(os.path.join(ORIGINAL_CWD, input_masks_dir))
    if not os.path.isabs(output_folder):
        output_folder = os.path.abspath(os.path.join(ORIGINAL_CWD, output_folder))

    print(f"Step 0: Multi-view video generation")
    print(f"  Input: {input_images_dir}")
    print(f"  Output: {output_folder}")
    print(f"  Model: {model_type}")
    print(f"  Seed: {seed}")
    print(f"  Deterministic: {deterministic}")
    set_global_seed(seed, deterministic=deterministic)

    if model_path is None:
        model_path = f"checkpoints/sv4d/{model_type}.safetensors"
    if not os.path.isabs(model_path):
        model_path = os.path.abspath(os.path.join(ORIGINAL_CWD, model_path))

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    if os.path.exists(output_folder) and overwrite:
        import shutil
        shutil.rmtree(output_folder)
    elif os.path.exists(output_folder) and not overwrite:
        raise FileExistsError(f"Output folder already exists and overwrite=False: {output_folder}")

    os.makedirs(output_folder, exist_ok=True)

    downsampled_object_dir, actual_frames, strategy = process_and_downsample(
        input_images_dir, input_masks_dir, output_folder, num_frames,
    )

    multiview_data = generate_multiview_video(
        downsampled_object_dir, model_path, output_folder, actual_frames, strategy,
        num_steps, img_size, seed, encoding_t, decoding_t, device,
        elevations_deg, azimuths_deg, image_frame_ratio, deterministic, verbose,
    )

    save_multiview_outputs(multiview_data, output_folder, fps)

    print(f"Done. Frames: {actual_frames}, Output: {output_folder}")


if __name__ == "__main__":
    Fire(main)
