# Process Your Own Video

Companion reference for the annotation UI. For launch commands and the end-to-end flow, see the README's [Run on Your Own Video](../README.md#run-on-your-own-video) section.

## Input Requirements

| | Required | Why |
|---|---|---|
| Subject | **a single, fully-visible foreground object** | The 4D proxy assumes one foreground entity that stays mostly in-frame and un-occluded. Multiple objects, severe occlusion, or subjects that exit/re-enter the frame produce broken masks or tracking failure downstream. |
| Resolution | exactly `854 x 480` | Downstream stages (stage_0 multiview, stage_1 rendering, stage_2 Wan2.2-VACE) are calibrated for this exact size and silently break otherwise. |
| Frame count | fixed at 45 | Sweet-spot length we settled on (SV4D needs `4k+1` frames). Video must have ≥ 45 frames; the UI lets you pick the start of the window. |
| Container | mp4 / avi / mov / mkv | Decoded with OpenCV. AV1 / HEVC / VP9 may need re-encoding. |

Pre-resize to `854x480` if needed. Don't stretch a non-`16:9` clip directly — distorted subjects propagate into a squashed 4D proxy. Pick one:

```bash
# Already 16:9 — just scale.
ffmpeg -i input.mp4 -vf "scale=854:480" -c:v libx264 -pix_fmt yuv420p out.mp4

# Non-16:9 — center-crop (fills the frame, trims edges).
ffmpeg -i input.mp4 -vf "scale=854:480:force_original_aspect_ratio=increase,crop=854:480" -c:v libx264 -pix_fmt yuv420p out.mp4
```

## UI Walkthrough

1. **Upload Video** tab → choose your file → **Step 1: Load Video**. The tool extracts every frame to a temp directory and rejects the video upfront if the resolution is wrong.
2. **Drag the start-frame slider** to pick which 45-frame window you want. Preview updates as you drag.
3. Click **Step 2: Confirm and Extract**. The selected 45 frames are sent to SAM2 for feature extraction; the first frame appears in the annotator.
4. **Annotate the foreground object** on the displayed frame:
   - **Positive Point (foreground)** → click on the object you want to keep.
   - **Negative Point (background)** → click on background regions to exclude.
   - **Clear Points** wipes the current frame's clicks; **Reset** wipes everything.
   - Use the **Frame index** slider to switch frames if you want to add corrections on a later frame.
5. Click **Submit Mask and Track**. SAM2 propagates the mask through all 45 frames and shows the result as a video.
6. Enter a **Scene name**, then **Save to data/user/**. The tool writes images, masks, an annotation metadata file, and a ready-to-run config.

## Outputs

```
data/user/{scene_name}/
├── images/                     # 45 jpg frames, 854x480
├── masks/                      # 45 png masks, 0/255 grayscale
└── annotation_meta.json        # selected points, labels, source frame index

configs/user/{scene_name}.yaml  # generated, inherits configs/default.yaml
```

The generated yaml inherits all defaults and pre-fills a `yaw -120°` orbit; edit it directly to tweak the trajectory.

Pipeline outputs land under `outputs/user/multiview/{scene_name}/`, `outputs/user/prepared/{scene_name}/`, and `outputs/user/rendering/{scene_name}/{trajectory_name}/`. See [pipeline.md](pipeline.md) for stage-by-stage details, and [trajectory_editor.md](trajectory_editor.md) for fine-grained trajectory authoring after the scene is reconstructed.
