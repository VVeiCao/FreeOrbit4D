# Pipeline

FreeOrbit4D performs training-free camera redirection by building a foreground-complete 4D proxy, rendering novel views along a target camera trajectory, and refining the rendered video.

| Stage | What it does |
|---|---|
| Stage 0 — Multiview Preparation | Generates auxiliary views for each input frame and prepares per-frame image/mask layouts. |
| Stage 1A — 4D Proxy Reconstruction | Reconstructs dynamic foreground point clouds and a fused static background, then aligns them via correspondence-aware alignment into a foreground-complete 4D proxy. |
| Stage 1B — Novel-View Rendering | Renders RGB and depth videos from the 4D proxy along a target camera trajectory. |
| Stage 2 — Video Refinement | Refines the rendered novel-view video with depth, reference image, and text conditioning. |

`{trajectory_name}` is generated from the configured camera trajectory, including radius scale, e.g. `arc_yaw_-120_scale_1p0`. For a custom trajectory JSON, it is generated from the JSON filename.

## Run a Single Stage

Each stage can be run individually with `python run_pipeline.py <stage> --config <config>`. Examples below use `configs/scenes/camel.yaml`.
When a stage is run again, its existing outputs are replaced. Use `--resume_from` to skip completed steps after an interruption.

### Stage 0

```bash
python run_pipeline.py stage_0 --config configs/scenes/camel.yaml
```

**Input** — `demo/camel/`:

```text
images/  00000.jpg, 00001.jpg, ...
masks/   00000.png, 00001.png, ...
```

**Output** — `outputs/multiview/camel/` and `outputs/prepared/camel/`:

```text
outputs/multiview/camel/
├── downsampled/{original,object,mask}/
└── multiview_images/{v001..v004}/, multiview_videos/

outputs/prepared/camel/
└── 00000/, 00001/, ...   # per-frame {images, masks}
```

### Stage 1

```bash
python run_pipeline.py stage_1 --config configs/scenes/camel.yaml
```

**Input** — `outputs/prepared/camel/` (per-frame `{images, masks}` from Stage 0).

**Output** — adds aligned point clouds and rendered novel views:

```text
outputs/prepared/camel/
├── 00000/pointcloud/
│   ├── 00000_foreground_1_view.ply
│   ├── 00000_foreground_5_views.ply
│   ├── 00000_foreground_5_views_aligned.ply
│   └── 00000_foreground_5_views_aligned_smooth.ply
├── ...
├── global_background.ply
├── global_camera.json
└── {trajectory_name}.json

outputs/rendering/camel/{trajectory_name}/
├── raw_images/{original_images,rendered_images,rendered_depths}/
├── videos/{original_images,rendered_images,rendered_depths}.mp4
└── inference/{reference_image.png, original_images.mp4, rendered_depths.mp4}
```

### Stage 2

```bash
python run_pipeline.py stage_2 --config configs/scenes/camel.yaml
```

**Input** — `outputs/rendering/camel/{trajectory_name}/inference/` from Stage 1 (`reference_image.png`, `original_images.mp4`, `rendered_depths.mp4`).

**Output** — final refined video plus the auto-generated text prompt used for refinement:

```text
outputs/rendering/camel/{trajectory_name}/inference/
├── generated_prompt.txt
└── output_video.mp4
```
