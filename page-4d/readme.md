# PAGE-4D Evaluation

This repository contains the **evaluation code for PAGE-4D**.

## Input Data
The input images are stored in the directory:
```online_img3/1/*.png```

## Environment Setup
The environment setup and dependencies follow those provided by [VGGT](https://github.com/facebookresearch/vggt).

Please refer to that repository for detailed installation instructions.

## Running Evaluation
After the environment has been set up, run the following command to evaluate PAGE-4D:
```bash
python run_fig1_auto.py
```

This command will generate prediction results for both VGGT and PAGE-4D.

•	The results of PAGE-4D are stored in:

online_img3/fig1_update_dpg/*.ply

•	The results of VGGT are stored in:

online_img3/fig1_update_vggt/*.ply

## Visualization

You can visualize the generated .ply files using MeshLab or any other standard 3D viewer. 

Each .ply file represents a reconstructed 3D point cloud generated from the input images, allowing comparison between PAGE-4D and VGGT predictions.

---

