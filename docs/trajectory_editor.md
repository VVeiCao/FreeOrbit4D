# Trajectory Editor

Companion reference for `scripts/1_3_cam_traj.py`. For the full launch + re-render flow, see the README's [Custom camera trajectories](../README.md#custom-camera-trajectories) section.

## Controls

- `Frame`, `Play`, `FPS`: preview the input video timeline.
- `Playback View`: `First Person` follows the generated camera path; `Third Person` keeps an external view so you can inspect camera frustums and the trajectory.
- `Reset View`: reset the browser camera to the scene view.
- `Radius Scale`: move the generated orbit closer to or farther from the scene.
- `Yaw`, `Pitch`, `Roll`: enable rotation around each axis; the angle sliders set the rotation amount.
- `Keyframes`: number of sparse cameras used to define the generated path.
- `Generate Trajectory`: create and preview the trajectory in the viewer.
- `Filename` + `Save`: write the trajectory JSON to the prepared data directory, e.g. `outputs/prepared/camel/my_trajectory.json`.
- `Saved Trajectories`: show or hide existing trajectory JSON files for inspection.

## Save and Run

1. Set the trajectory parameters.
2. Enter a filename such as `my_trajectory.json`.
3. Click `Generate Trajectory`, then `Save`.
4. Re-render with `--resume_from render --trajectory_json my_trajectory.json` (see README).

The output lands at `outputs/rendering/{scene_name}/{trajectory_basename}/inference/output_video.mp4`.
If the trajectory output directory already exists, rerendering with the same trajectory filename replaces it.
