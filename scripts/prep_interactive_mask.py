"""
Data preprocessing: interactive mask annotation with SAM2 and Gradio.

Function:
    Annotate foreground masks interactively on a video or image sequence through
    a Gradio web UI. SAM2 propagates foreground/background clicks from one frame
    to the full sequence.

Output files, mirroring the per-scene layout in demo/{scene}/:
    data/user/{scene_name}/
    ├── images/
    │   ├── 00000.jpg
    │   └── ...
    ├── masks/
    │   ├── 00000.png          # Grayscale PNG, 0 background / 255 foreground
    │   └── ...
    └── annotation_meta.json   # Selected points / labels / frame index
    configs/user/{scene_name}.yaml   # Generated config file

Dependencies:
    pip install git+https://github.com/facebookresearch/sam2.git
    pip install gradio loguru

SAM2 checkpoint:
    Default path: checkpoints/sam2/sam2_hiera_large.pt
    Automatic download: bash download_checkpoints.sh sam2
    Manual download: wget https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt

Examples:
    python scripts/prep_interactive_mask.py
    python scripts/prep_interactive_mask.py --port 8890
    python scripts/prep_interactive_mask.py --checkpoint_dir /path/to/sam2_hiera_large.pt
"""

import torch

import os
import sys
import subprocess
import shutil
import time
import json
import tempfile
from pathlib import Path

import cv2
import gradio as gr
import imageio.v2 as iio
import numpy as np
from PIL import Image
from loguru import logger as guru

from sam2.build_sam import build_sam2_video_predictor

# Project root.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Video codecs that Gradio/OpenCV often cannot decode reliably.
INCOMPATIBLE_CODECS = ['av1', 'av01', 'hevc', 'h265', 'vp9', 'vp8']

# Required input resolution. The downstream pipeline (stage_0 multiview, stage_1
# rendering, stage_2 Wan2.2-VACE) is calibrated for DAVIS-style 854x480 video.
# Other resolutions silently break aspect-ratio assumptions in stage_0 prep
# (832x480 center-crop) and the Wan2.2-VACE model. We reject any other size at
# upload time rather than letting the pipeline fail downstream.
PIPELINE_INPUT_WIDTH = 854
PIPELINE_INPUT_HEIGHT = 480


def configure_cuda_for_sam2() -> torch.autocast:
    """Configure CUDA settings after argparse has handled CLI/help paths."""
    if not torch.cuda.is_available():
        raise RuntimeError("SAM2 interactive annotation requires a visible CUDA GPU.")
    autocast_context = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    autocast_context.__enter__()
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    return autocast_context


def get_video_codec(video_path):
    """Return the video codec name."""
    try:
        cmd = [
            'ffprobe', '-v', 'error',
            '-select_streams', 'v:0',
            '-show_entries', 'stream=codec_name',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result.stdout.strip().lower()
    except Exception as e:
        guru.warning(f"Failed to get video codec: {e}")
        return None


def check_and_convert_video(video_path):
    """Check the video codec and convert to H.264 when needed."""
    if not video_path or not os.path.exists(video_path):
        return video_path

    codec = get_video_codec(video_path)
    guru.info(f"Video codec: {codec}")

    if codec and codec in INCOMPATIBLE_CODECS:
        guru.info(f"Converting {codec} to H.264...")
        tmp_dir = tempfile.mkdtemp(prefix="freeorbit4d_")
        output_path = os.path.join(tmp_dir, f"converted_{int(time.time())}.mp4")

        cmd = [
            'ffmpeg', '-y', '-i', video_path,
            '-c:v', 'libx264', '-preset', 'fast',
            '-crf', '23', '-pix_fmt', 'yuv420p', '-an',
            output_path
        ]
        try:
            subprocess.run(cmd, capture_output=True, text=True, timeout=300, check=True)
            cap = cv2.VideoCapture(output_path)
            if cap.isOpened() and int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) > 0:
                cap.release()
                guru.info(f"Video conversion succeeded: {output_path}")
                return output_path
            cap.release()
        except Exception as e:
            guru.warning(f"Video conversion failed: {e}")

    return video_path


def isimage(p):
    return os.path.splitext(p.lower())[-1] in [".png", ".jpg", ".jpeg"]


def draw_points(img, points, labels):
    """Draw selected points on the image. Green is positive, red is negative."""
    out = img.copy()
    for p, label in zip(points, labels):
        x, y = int(p[0]), int(p[1])
        color = (0, 255, 0) if label == 1.0 else (255, 0, 0)
        out = cv2.circle(out, (x, y), 10, color, -1)
    return out


def compose_img_mask(img, color_mask, fac=0.5):
    """Overlay a color mask on an image."""
    out_f = fac * img / 255 + (1 - fac) * color_mask / 255
    return (255 * out_f).astype("uint8")


class MaskAnnotator:
    """Interactive SAM2 mask annotator."""

    def __init__(self, checkpoint_dir, model_cfg):
        self.checkpoint_dir = checkpoint_dir
        self.model_cfg = model_cfg
        self.sam_model = None

        self.selected_points = []
        self.selected_labels = []
        self.cur_label_val = 1.0

        self.frame_index = 0
        self.image = None
        self.cur_mask = None
        self.cur_logit = None
        self.masks_all = []

        self.img_dir = ""
        self.img_paths = []
        self.video_name = None
        self.inference_state = None
        self._temp_dirs = []  # Track temporary directories and clean them after saving.

        self._init_sam_model()

    def _init_sam_model(self):
        if self.sam_model is not None:
            return
        if not os.path.exists(self.checkpoint_dir):
            error_msg = (
                f"SAM2 checkpoint does not exist: {self.checkpoint_dir}\n\n"
                "Download command:\n"
                "  wget https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt\n"
                f"Place the file at: {self.checkpoint_dir}"
            )
            guru.error(error_msg)
            raise FileNotFoundError(error_msg)
        self.sam_model = build_sam2_video_predictor(self.model_cfg, self.checkpoint_dir)
        guru.info(f"SAM2 model loaded: {self.checkpoint_dir}")

    def extract_scene_name(self, path):
        """Extract a scene name from a path."""
        if not path:
            return "sequence"
        name = Path(path).stem if Path(path).is_file() else Path(path).name
        if not name or name in ['.', '..']:
            name = Path(path).parent.name
        import re
        clean = re.sub(r'[^\w\-_]', '_', name)
        return clean if clean else "sequence"

    def clear_points(self):
        self.selected_points.clear()
        self.selected_labels.clear()
        return None, None, None, "All points cleared. Select points again."

    def _clear_image(self):
        self.image = None
        self.frame_index = 0
        self.cur_mask = None
        self.cur_logit = None
        self.masks_all = []

    def reset(self):
        self._clear_image()
        if self.inference_state is not None:
            self.sam_model.reset_state(self.inference_state)

    def set_img_dir(self, img_dir):
        self._clear_image()
        self.img_dir = img_dir
        if not os.path.exists(img_dir):
            guru.error(f"Directory does not exist: {img_dir}")
            return 0
        self.img_paths = [
            os.path.abspath(os.path.join(img_dir, p))
            for p in sorted(os.listdir(img_dir)) if isimage(p)
        ]
        guru.info(f"Found {len(self.img_paths)} images")
        self.video_name = self.extract_scene_name(img_dir)
        return len(self.img_paths)

    def set_input_image(self, i=0):
        if i < 0 or i >= len(self.img_paths):
            return self.image
        self.clear_points()
        self.frame_index = i
        self.image = iio.imread(self.img_paths[i])
        return self.image

    def get_sam_features(self):
        try:
            self.inference_state = self.sam_model.init_state(video_path=self.img_dir)
            self.sam_model.reset_state(self.inference_state)
            guru.info("SAM feature extraction completed")
            return "SAM feature extraction completed. Click points on the image, then submit to start tracking.", self.image
        except Exception as e:
            error_msg = f"SAM feature extraction failed: {e}"
            guru.error(error_msg)
            return error_msg, self.image

    def set_positive(self):
        self.cur_label_val = 1.0
        return "Selecting positive points (foreground)"

    def set_negative(self):
        self.cur_label_val = 0.0
        return "Selecting negative points (background)"

    def add_point(self, frame_idx, i, j):
        self.selected_points.append([j, i])
        self.selected_labels.append(self.cur_label_val)
        mask, logit = self._get_sam_mask(
            frame_idx,
            np.array(self.selected_points, dtype=np.float32),
            np.array(self.selected_labels, dtype=np.int32),
        )
        self.cur_mask = mask
        self.cur_logit = logit
        return mask

    def _get_sam_mask(self, frame_idx, input_points, input_labels):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            _, out_obj_ids, out_mask_logits = self.sam_model.add_new_points_or_box(
                inference_state=self.inference_state,
                frame_idx=frame_idx,
                obj_id=0,
                points=input_points,
                labels=input_labels,
            )
        mask = (out_mask_logits[0] > 0.0).squeeze().cpu().numpy()
        logit = out_mask_logits[0].squeeze().cpu().numpy()
        return mask, logit

    def run_tracker(self):
        """Propagate masks to all frames."""
        images = [iio.imread(p)[:, :, :3] for p in self.img_paths]
        self.masks_all = []

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            for out_frame_idx, out_obj_ids, out_mask_logits in self.sam_model.propagate_in_video(
                self.inference_state, start_frame_idx=0
            ):
                mask = (out_mask_logits[0] > 0.0).squeeze().cpu().numpy()
                self.masks_all.append(mask)

        # Generate preview video.
        out_frames = []
        for img, mask in zip(images, self.masks_all):
            colored_mask = np.zeros_like(img)
            colored_mask[mask] = [0, 255, 0]
            out_frames.append(compose_img_mask(img, colored_mask, 0.5))

        tmp_dir = tempfile.mkdtemp(prefix="freeorbit4d_")
        self._temp_dirs.append(tmp_dir)
        out_vidpath = os.path.join(tmp_dir, "tracked_masks.mp4")
        iio.mimwrite(out_vidpath, out_frames)

        msg = f"Tracking completed for {len(self.masks_all)} frames. Enter a scene name and save if the result looks good."
        return out_vidpath, msg

    def save_to_data_dir(self, scene_name):
        """Save in the per-scene layout matching demo/{scene}/{images,masks}/."""
        if not self.masks_all or len(self.masks_all) == 0:
            return "Run mask tracking first"
        if not scene_name or not scene_name.strip():
            return "Enter a scene name"

        scene_name = scene_name.strip()
        scene_dir = os.path.join(PROJECT_ROOT, "data", "user", scene_name)
        images_dir = os.path.join(scene_dir, "images")
        masks_dir = os.path.join(scene_dir, "masks")
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(masks_dir, exist_ok=True)

        for i, (img_path, mask) in enumerate(zip(self.img_paths, self.masks_all)):
            # Save image as JPEG.
            img = Image.open(img_path).convert("RGB")
            img.save(os.path.join(images_dir, f"{i:05d}.jpg"))

            # Save mask as grayscale PNG with values 0/255.
            mask_uint8 = (mask.astype(np.uint8) * 255)
            Image.fromarray(mask_uint8, mode='L').save(
                os.path.join(masks_dir, f"{i:05d}.png")
            )

        # Generate config.
        config_dir = os.path.join(PROJECT_ROOT, "configs", "user")
        os.makedirs(config_dir, exist_ok=True)
        config_path = os.path.join(config_dir, f"{scene_name}.yaml")

        config_content = f"""_base_: ../default.yaml

project:
  name: {scene_name}
  input_images: data/user/{scene_name}/images
  input_masks: data/user/{scene_name}/masks
  output_root: outputs/user

# Camera trajectory — customize for your scene.
# arc_type: yaw (horizontal orbit) | pitch (vertical tilt) | roll (lateral roll)
# arc_angle: signed degrees (e.g. -120 for left orbit, +90 for right)
# arc_radius_scale: 1.0 = scene-fit radius; <1 closer, >1 farther
stage_1:
  camera:
    mode: arc
    arc_type: yaw
    arc_angle: -120
    num_keyframes: 8

  rendering:
    arc_type: yaw
    arc_angle: -120
    arc_radius_scale: 1.0
"""
        with open(config_path, 'w') as f:
            f.write(config_content)

        # Save annotation metadata at the scene root (sibling of images/ and
        # masks/) so the image/mask dirs stay clean and match demo/ layout.
        meta = {
            "scene_name": scene_name,
            "num_frames": len(self.masks_all),
            "selected_points": self.selected_points,
            "selected_labels": self.selected_labels,
            "frame_index": self.frame_index,
        }
        meta_path = os.path.join(scene_dir, "annotation_meta.json")
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)

        msg = (
            f"Save completed.\n"
            f"  Images: {images_dir} ({len(self.masks_all)} frames)\n"
            f"  Mask: {masks_dir}\n"
            f"  Config: {config_path}\n\n"
            f"Default camera trajectory: arc / yaw / -120 deg / radius_scale 1.0\n"
            f"  - arc_type:         yaw (horizontal orbit) | pitch (vertical tilt) | roll (lateral roll)\n"
            f"  - arc_angle:        signed degrees, e.g. -120 (left orbit) or +90 (right orbit)\n"
            f"  - arc_radius_scale: 1.0 = scene-fit; <1 closer, >1 farther\n"
            f"Edit stage_1.camera / stage_1.rendering in {config_path} to customize.\n\n"
            f"Next step:\n"
            f"  python run_pipeline.py full --config configs/user/{scene_name}.yaml"
        )
        # Clean up temporary directories.
        for d in self._temp_dirs:
            if os.path.exists(d):
                shutil.rmtree(d, ignore_errors=True)
                guru.info(f"Cleaned temporary directory: {d}")
        self._temp_dirs.clear()
        # If img_dir is a system temporary directory from video extraction, clean it too.
        if self.img_dir and self.img_dir.startswith(tempfile.gettempdir()):
            shutil.rmtree(self.img_dir, ignore_errors=True)
            guru.info(f"Cleaned temporary frame directory: {self.img_dir}")

        guru.info(msg)
        return msg


def make_demo(checkpoint_dir, model_cfg):
    annotator = MaskAnnotator(checkpoint_dir, model_cfg)

    with gr.Blocks(title="FreeOrbit4D - Interactive Mask Annotation") as demo:
        gr.Markdown("# FreeOrbit4D - Interactive Mask Annotation (SAM2)")
        instruction = gr.Textbox(
            "Upload a video, then click on the image to annotate the foreground object.",
            label="Instructions", interactive=False,
        )

        # ===== Input =====
        gr.Markdown(
            f"""### ⚠️ Resolution requirement

This pipeline **only supports videos at exactly `{PIPELINE_INPUT_WIDTH} x {PIPELINE_INPUT_HEIGHT}`**.
Any other resolution will be **rejected on Load** — no auto-resize happens here, because downstream stages
(stage_0 multiview, stage_1 rendering, stage_2 Wan2.2-VACE) are calibrated for this exact size and will fail
or produce garbage on other inputs.

**If your video is a different size, pre-resize it first with ffmpeg:**

```bash
ffmpeg -i input.mp4 -vf scale={PIPELINE_INPUT_WIDTH}:{PIPELINE_INPUT_HEIGHT} \\
       -c:v libx264 -pix_fmt yuv420p out.mp4
```
"""
        )
        input_video_field = gr.File(
            label="Upload video file",
            file_types=[".mp4", ".avi", ".mov", ".mkv"],
        )
        load_video_button = gr.Button("Step 1: Load Video")

        # Step 2: choose start frame.
        with gr.Group(visible=False) as frame_select_group:
            gr.Markdown("### Choose Start Frame and Sampling Parameters")
            with gr.Row():
                video_stride = gr.Number(1, label="Sampling stride", minimum=1, step=1)
                video_num_frames = gr.Number(
                    45, label="Number of frames (locked at 45)",
                    minimum=45, maximum=45, step=1, interactive=False,
                )
            preview_slider = gr.Slider(label="Start frame (drag to choose and preview)", minimum=0, maximum=1, value=0, step=1)
            preview_image = gr.Image(label="Frame preview")
            video_info = gr.Textbox(label="Info", interactive=False)
            extract_button = gr.Button("Step 2: Confirm and Extract", variant="primary")

        # ===== Annotation =====
        frame_index = gr.Slider(label="Frame index", minimum=0, maximum=1, value=0, step=1)

        with gr.Row():
            with gr.Column():
                reset_button = gr.Button("Reset")
                input_image = gr.Image(None, label="Input frame")
                with gr.Row():
                    pos_button = gr.Button("Positive Point (foreground)")
                    neg_button = gr.Button("Negative Point (background)")
                clear_button = gr.Button("Clear Points")

            with gr.Column():
                output_img = gr.Image(label="Current selection")
                submit_button = gr.Button("Submit Mask and Track")
                final_video = gr.Video(label="Mask tracking result")

        # ===== Save =====
        with gr.Row():
            scene_name_field = gr.Text(
                "my_scene", label="Scene name (scene_name)",
                info="Saves to data/user/{scene_name}/{images,masks}/ (mirrors demo/{scene}/ layout)",
            )
            save_button = gr.Button("Save to data/user/", variant="primary")

        # ===== State =====
        # Store all preloaded frames after load_video.
        _all_frames_dir = gr.State(None)    # Temporary directory containing all frame JPGs.
        _all_frames_total = gr.State(0)     # Total frame count.

        # ===== Event bindings =====

        def load_video(video_file):
            """Step 1: extract all video frames to a temporary directory and show the preview slider."""
            if video_file is None:
                return (
                    gr.Group(visible=False),  # frame_select_group
                    gr.Slider(),              # preview_slider
                    None,                     # preview_image
                    "Upload a video first",   # video_info
                    None,                     # _all_frames_dir
                    0,                        # _all_frames_total
                )

            converted = check_and_convert_video(video_file)
            cap = cv2.VideoCapture(converted)
            if not cap.isOpened():
                return (
                    gr.Group(visible=False), gr.Slider(), None,
                    "Could not open video file", None, 0,
                )

            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # Hard-block any video that does not match the pipeline's required
            # input resolution. Downstream stages are calibrated for this exact
            # size; resizing here would push the failure deeper into the pipeline.
            if orig_w != PIPELINE_INPUT_WIDTH or orig_h != PIPELINE_INPUT_HEIGHT:
                cap.release()
                err = (
                    f"Rejected: video is {orig_w}x{orig_h} but the pipeline requires exactly "
                    f"{PIPELINE_INPUT_WIDTH}x{PIPELINE_INPUT_HEIGHT}. "
                    f"Pre-resize with: "
                    f"ffmpeg -i input.mp4 -vf scale={PIPELINE_INPUT_WIDTH}:{PIPELINE_INPUT_HEIGHT} "
                    f"-c:v libx264 -pix_fmt yuv420p out.mp4"
                )
                guru.warning(err)
                return (
                    gr.Group(visible=False), gr.Slider(), None,
                    err, None, 0,
                )

            guru.info(f"Loading video: {total} frames, {orig_w}x{orig_h}")

            temp_dir = tempfile.mkdtemp(prefix="freeorbit4d_allframes_")
            count = 0
            for idx in range(total):
                ret, frame = cap.read()
                if not ret:
                    break
                cv2.imwrite(os.path.join(temp_dir, f"{idx:05d}.jpg"), frame)
                count += 1
            cap.release()

            guru.info(f"Extracted {count} frames to {temp_dir}")

            # Read the first frame as preview.
            first_frame_path = os.path.join(temp_dir, "00000.jpg")
            first_img = iio.imread(first_frame_path) if os.path.exists(first_frame_path) else None

            # Default stride=1 and num_frames=45 means the last frame is start + 44 < count.
            default_nframes = 45
            default_stride = 1
            slider_max = max(0, count - (default_nframes - 1) * default_stride - 1)
            # Gradio requires minimum < maximum. When the only valid start is 0,
            # bump the slider to (0, 1) so it stays draggable; extract_frames
            # will validate the actual range.
            slider_ui_max = max(1, slider_max)
            info_msg = (
                f"Video loaded: {count} frames, {orig_w}x{orig_h}.\n"
                f"Drag the slider to choose the start frame (range 0-{slider_max}), then click Confirm and Extract."
            )

            return (
                gr.Group(visible=True),                                        # frame_select_group
                gr.Slider(minimum=0, maximum=slider_ui_max, value=0, step=1, interactive=True),  # preview_slider
                first_img,                                                      # preview_image
                info_msg,                                                       # video_info
                temp_dir,                                                       # _all_frames_dir
                count,                                                          # _all_frames_total
            )

        def preview_frame(slider_val, frames_dir):
            """Show the selected frame while dragging the slider."""
            if not frames_dir or not os.path.isdir(frames_dir):
                return None
            frame_path = os.path.join(frames_dir, f"{int(slider_val):05d}.jpg")
            if os.path.exists(frame_path):
                return iio.imread(frame_path)
            return None

        def update_slider_range(stride, num_frames, total_frames):
            """Update the start-frame slider range when stride or frame count changes."""
            total = int(total_frames)
            if total <= 0:
                return gr.Slider(), ""
            stride_val = max(1, int(stride))
            nframes = max(1, int(num_frames))
            # Required last frame index: start + (nframes - 1) * stride < total.
            slider_max = max(0, total - (nframes - 1) * stride_val - 1)
            slider_ui_max = max(1, slider_max)
            info_msg = f"Valid start-frame range: 0 - {slider_max} ({total} total frames, stride {stride_val}, taking {nframes} frames)"
            return gr.Slider(minimum=0, maximum=slider_ui_max, value=0, step=1, interactive=True), info_msg

        def extract_frames(video_file, start_frame, stride, num_frames, frames_dir, total_frames):
            """Step 2: select a frame subset by start/stride/count and send it to SAM2."""
            if not frames_dir or not os.path.isdir(frames_dir):
                return "Click Load Video first", gr.Slider(), None, ""

            start = max(0, int(start_frame))
            stride_val = max(1, int(stride))
            nframes = int(num_frames)
            total = int(total_frames)

            # Compute selected frame indices.
            selected_indices = list(range(start, total, stride_val))

            if len(selected_indices) < nframes:
                return (
                    f"Not enough frames. Starting at frame {start} with stride {stride_val}, "
                    f"only {len(selected_indices)} frames are available, but {nframes} are required. "
                    f"Adjust the parameters. The video has {total} frames.",
                    gr.Slider(), None, ""
                )

            selected_indices = selected_indices[:nframes]

            # Copy selected frames to a new temporary directory and renumber them.
            temp_dir = tempfile.mkdtemp(prefix="freeorbit4d_frames_")
            for i, idx in enumerate(selected_indices):
                src = os.path.join(frames_dir, f"{idx:05d}.jpg")
                dst = os.path.join(temp_dir, f"{i:05d}.jpg")
                shutil.copy2(src, dst)

            num_imgs = annotator.set_img_dir(os.path.abspath(temp_dir))
            slider = gr.Slider(minimum=0, maximum=num_imgs - 1, value=0, step=1)
            first_image = annotator.set_input_image(0)
            sam_msg, sam_img = annotator.get_sam_features()

            scene = annotator.extract_scene_name(video_file)
            msg = (
                f"Extracted {nframes} frames (start frame {start}, stride {stride_val}). {sam_msg}"
            )
            return msg, slider, sam_img if sam_img is not None else first_image, scene

        def get_select_coords(frame_idx, img, evt: gr.SelectData):
            if img is None:
                return None
            i = evt.index[1]
            j = evt.index[0]
            binary_mask = annotator.add_point(frame_idx, i, j)
            colored_mask = np.zeros_like(img)
            colored_mask[binary_mask] = [0, 255, 0]
            out = compose_img_mask(img, colored_mask, 0.5)
            out = draw_points(out, annotator.selected_points, annotator.selected_labels)
            return out

        def run_tracker_with_message():
            vid, msg = annotator.run_tracker()
            return vid, msg

        def save_data(scene_name):
            return annotator.save_to_data_dir(scene_name)

        # ===== Bindings: video tab =====
        load_video_button.click(
            load_video,
            [input_video_field],
            [frame_select_group, preview_slider, preview_image, video_info,
             _all_frames_dir, _all_frames_total],
        )
        preview_slider.change(
            preview_frame,
            [preview_slider, _all_frames_dir],
            [preview_image],
        )
        # Update slider range when stride or frame count changes.
        video_stride.change(
            update_slider_range,
            [video_stride, video_num_frames, _all_frames_total],
            [preview_slider, video_info],
        )
        video_num_frames.change(
            update_slider_range,
            [video_stride, video_num_frames, _all_frames_total],
            [preview_slider, video_info],
        )
        # Confirm and extract using preview_slider as the start frame.
        extract_button.click(
            extract_frames,
            [input_video_field, preview_slider, video_stride, video_num_frames,
             _all_frames_dir, _all_frames_total],
            [instruction, frame_index, input_image, scene_name_field],
        )

        # ===== Bindings: annotation =====
        frame_index.change(annotator.set_input_image, [frame_index], [input_image])
        input_image.select(get_select_coords, [frame_index, input_image], [output_img])

        reset_button.click(annotator.reset)
        clear_button.click(annotator.clear_points, outputs=[output_img, final_video, instruction, instruction])
        pos_button.click(annotator.set_positive, outputs=[instruction])
        neg_button.click(annotator.set_negative, outputs=[instruction])
        submit_button.click(run_tracker_with_message, outputs=[final_video, instruction])
        save_button.click(save_data, [scene_name_field], [instruction])

    return demo


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="FreeOrbit4D - Interactive Mask Annotation (SAM2)")
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument(
        "--checkpoint_dir", type=str,
        default=os.path.join(PROJECT_ROOT, "checkpoints", "sam2", "sam2_hiera_large.pt"),
    )
    parser.add_argument("--model_cfg", type=str, default="sam2_hiera_l.yaml")
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint_dir):
        print(f"\nSAM2 checkpoint does not exist: {args.checkpoint_dir}")
        print("\nDownload:")
        print("  wget https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt")
        print(f"\nPlace it at: {args.checkpoint_dir}")
        exit(1)

    configure_cuda_for_sam2()
    demo = make_demo(args.checkpoint_dir, args.model_cfg)
    demo.launch(server_name="127.0.0.1", server_port=args.port or None)
