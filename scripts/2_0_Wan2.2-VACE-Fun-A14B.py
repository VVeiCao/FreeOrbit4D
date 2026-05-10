"""Step 2.0: Wan2.2 video generation with automatic captioning."""

import torch
import cv2
import argparse
import sys
from pathlib import Path
from PIL import Image
import numpy as np
from diffsynth.utils.data import save_video, VideoData
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
import os
os.environ["DIFFSYNTH_DOWNLOAD_SOURCE"] = "huggingface"

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.args import create_base_parser, merge_args_with_config
from utils.seed import resolve_seed, set_global_seed
from utils.trajectory import format_arc_trajectory_name


def check_image_size(image_path, target_width=832, target_height=480):
    img = Image.open(image_path)
    width, height = img.size
    return width == target_width and height == target_height


def check_video_size(video_path, target_width=832, target_height=480):
    cap = cv2.VideoCapture(str(video_path))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return width == target_width and height == target_height


def generate_caption_from_video(video_path, device="cuda:0"):
    """Generate caption from video using Qwen3-VL-2B (all frames)."""
    try:
        from transformers import AutoProcessor
        from qwen_vl_utils import process_vision_info

        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        frames = []
        indices = np.linspace(0, total_frames - 1, total_frames, dtype=int)
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)
                frames.append(pil_image)
        cap.release()

        model_name = "Qwen/Qwen3-VL-2B-Instruct"
        from transformers import Qwen3VLForConditionalGeneration
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name, dtype=torch.bfloat16, trust_remote_code=True
        ).to(device)
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

        if len(frames) == 1:
            vision_content = {"type": "image", "image": frames[0]}
        else:
            vision_content = {"type": "video", "video": frames, "fps": 1.0}

        messages = [{
            "role": "user",
            "content": [
                vision_content,
                {
                    "type": "text",
                    "text": "Given the input video, write ONE English caption for video generation. Describe the main subject identity and distinctive appearance (color/material/texture/unique marks), the scene/background, lighting and overall color tone, and the continuous action over time. Emphasize: photorealistic, natural lighting, consistent colors, stable textures, sharp details, smooth motion. Do NOT mention camera movement, shot type, lenses, viewpoint change, or any technical conditioning. Do NOT invent new objects. No uncertainty words. 40-80 words."
                }
            ]
        }]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt",
        )
        inputs = inputs.to(device)

        generated_ids = model.generate(**inputs, max_new_tokens=400)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        caption = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        print(f"Generated caption: {caption}")

        del model
        del processor
        torch.cuda.empty_cache()
        return caption

    except ImportError as e:
        print(f"Warning: Cannot load Qwen3-VL model: {e}")
        print(f"Install: pip install transformers>=4.40.0 qwen-vl-utils")
        return "A scene with natural lighting and photorealistic details"
    except Exception as e:
        print(f"Warning: Caption generation failed: {e}")
        import traceback
        traceback.print_exc()
        return "A scene with natural lighting and photorealistic details"


def main():
    """Generate video using Wan2.2-VACE-Fun-A14B model."""
    parser = create_base_parser('Step 2.0: Wan2.2 video generation')
    parser.add_argument('--data_dir', type=str, required=False,
                        help='Data directory path')
    parser.add_argument('--trajectory_name', type=str, default=None,
                        help='Trajectory subdirectory name (e.g. arc_yaw_-120_scale_1p0)')
    parser.add_argument('--num_inference_steps', type=int, default=None, help='Inference steps')
    parser.add_argument('--sigma_shift', type=float, default=None, help='Noise shift')
    parser.add_argument('--cfg_scale', type=float, default=None, help='CFG guidance scale')
    parser.add_argument('--output', type=str, default=None, help='Output video path')
    parser.add_argument('--use_saved_prompt', action='store_true',
                        help='Use saved generated_prompt.txt instead of regenerating')
    parser.add_argument('--deterministic', action=argparse.BooleanOptionalAction, default=None,
                        help='Enable deterministic PyTorch/CUDA settings')

    args = parser.parse_args()
    config = merge_args_with_config(args)

    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        rendering_dir = config.get('project.output_rendering_base')
        if not rendering_dir:
            raise ValueError("Must specify data directory via --data_dir or config project.output_rendering_base")
        rendering_path = Path(rendering_dir)
        if args.trajectory_name:
            trajectory_name = args.trajectory_name
        else:
            arc_type = config.get('stage_1.rendering.arc_type', 'yaw')
            arc_angle = config.get('stage_1.rendering.arc_angle', 90)
            arc_radius_scale = config.get('stage_1.rendering.arc_radius_scale', 1.0)
            trajectory_name = format_arc_trajectory_name(arc_type, arc_angle, arc_radius_scale)
        data_dir = rendering_path / trajectory_name

    seed = args.seed if args.seed is not None else resolve_seed(config, 'stage_2.seed', default=1)
    deterministic = args.deterministic if args.deterministic is not None else bool(config.get('common.deterministic', False))
    set_global_seed(seed, deterministic=deterministic)
    num_inference_steps = args.num_inference_steps if args.num_inference_steps is not None else config.get('stage_2.num_inference_steps', 50)
    sigma_shift = args.sigma_shift if args.sigma_shift is not None else config.get('stage_2.sigma_shift', 16.0)
    cfg_scale = args.cfg_scale if args.cfg_scale is not None else config.get('stage_2.cfg_scale', 5.0)
    reference_image_path = data_dir / "inference" / "reference_image.png"
    control_video_path = data_dir / "inference" / "rendered_depths.mp4"

    if not reference_image_path.exists():
        raise FileNotFoundError(f"Reference image not found: {reference_image_path}")
    if not control_video_path.exists():
        raise FileNotFoundError(f"Control video not found: {control_video_path}")

    image_ok = check_image_size(reference_image_path, target_width=832, target_height=480)
    video_ok = check_video_size(control_video_path, target_width=832, target_height=480)
    if not (image_ok and video_ok):
        raise ValueError("Image or video size mismatch (must be 832x480)")

    prompt_save_path = data_dir / "inference" / "generated_prompt.txt"
    output_path = data_dir / "inference" / "output_video.mp4" if args.output is None else Path(args.output)

    if args.use_saved_prompt and prompt_save_path.exists():
        with open(prompt_save_path, 'r', encoding='utf-8') as f:
            prompt = f.read().strip()
        print(f"Using saved prompt: {prompt}")
    else:
        if args.use_saved_prompt:
            print(f"Warning: --use_saved_prompt enabled but file not found: {prompt_save_path}")
        original_video_path = data_dir / "inference" / "original_images.mp4"
        if not original_video_path.exists():
            raise FileNotFoundError(f"Original video not found for caption generation: {original_video_path}")
        prompt = generate_caption_from_video(original_video_path, device="cuda:0")
        with open(prompt_save_path, 'w', encoding='utf-8') as f:
            f.write(prompt)

    print(f"Loading Wan2.2 model...")

    vram_config = {
        "offload_dtype": "disk",
        "offload_device": "disk",
        "onload_dtype": torch.bfloat16,
        "onload_device": "cpu",
        "preparing_dtype": torch.bfloat16,
        "preparing_device": "cuda:0",
        "computation_dtype": torch.bfloat16,
        "computation_device": "cuda:0",
    }

    import glob
    wan_ckpt = os.path.join(project_root, "checkpoints", "wan2.2")

    tokenizer_path = os.path.join(wan_ckpt, "tokenizer", "umt5-xxl")
    if not os.path.isdir(tokenizer_path):
        raise FileNotFoundError(
            f"Wan tokenizer not found: {tokenizer_path}\n"
            f"Run: bash download_checkpoints.sh wan"
        )

    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda:0",
        redirect_common_files=False,
        model_configs=[
            ModelConfig(path=glob.glob(os.path.join(wan_ckpt, "high_noise_model", "diffusion_pytorch_model*.safetensors")), **vram_config),
            ModelConfig(path=glob.glob(os.path.join(wan_ckpt, "low_noise_model", "diffusion_pytorch_model*.safetensors")), **vram_config),
            ModelConfig(path=os.path.join(wan_ckpt, "models_t5_umt5-xxl-enc-bf16.pth"), **vram_config),
            ModelConfig(path=os.path.join(wan_ckpt, "Wan2.1_VAE.pth"), **vram_config),
        ],
        tokenizer_config=ModelConfig(path=tokenizer_path),
        vram_limit=torch.cuda.mem_get_info(0)[1] / (1024 ** 3) - 2,
    )

    print(f"Generating video: steps={num_inference_steps}, seed={seed}, deterministic={deterministic}, sigma_shift={sigma_shift}, cfg_scale={cfg_scale}")

    cap = cv2.VideoCapture(str(control_video_path))
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    control_video = VideoData(str(control_video_path), height=480, width=832)
    reference_image = Image.open(reference_image_path)

    negative_prompt = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走，五条腿，六条腿，脚不接触地面，抖动画面, 人有三只手，人有四只手"

    video = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        vace_video=control_video,
        vace_reference_image=reference_image,
        num_frames=num_frames,
        num_inference_steps=num_inference_steps,
        seed=seed,
        sigma_shift=sigma_shift,
        cfg_scale=cfg_scale,
        tiled=True
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_video(video, str(output_path), fps=15, quality=5)
    print(f"Video saved to: {output_path}")


if __name__ == "__main__":
    main()
