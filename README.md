<p align="center">
  <img src="assets/title.png" width="400">
</p>

### <p align="center">[FreeOrbit4D: Training-Free Arbitrary Camera Redirection for Monocular Videos via Geometry-Complete 4D Reconstruction](https://arxiv.org/abs/2601.18993)</p>

[![arXiv](https://img.shields.io/badge/arXiv-2601.18993-b31b1b.svg)](https://arxiv.org/abs/2601.18993)
[![Paper](https://img.shields.io/badge/Paper-PDF-blue.svg)](https://arxiv.org/pdf/2601.18993)
[![Project Page](https://img.shields.io/badge/Project-Page-green.svg)](https://cvmlgroup.web.illinois.edu/freeorbit4d/)

[Wei Cao](https://vveicao.github.io/)<sup>1</sup>, [Hao Zhang](https://haoz19.github.io/)<sup>1</sup>, [Fengrui Tian](https://tianfr.github.io/)<sup>2</sup>, [Yulun Wu](https://yulunwu0108.github.io/)<sup>1</sup>, [Yingying Li](https://www.yingying.li/)<sup>1</sup>, [Shenlong Wang](https://shenlong.web.illinois.edu/)<sup>1</sup>, [Ning Yu](https://ningyu1991.github.io/)<sup>3</sup>, [Yaoyao Liu](https://yaoyaoliu.web.illinois.edu/)<sup>1</sup>

<sup>1</sup>University of Illinois Urbana-Champaign, <sup>2</sup>University of Pennsylvania, <sup>3</sup>Eyeline Labs

FreeOrbit4D is a training-free framework that redirects monocular videos to arbitrary camera trajectories. It recovers a geometry-complete 4D proxy by decoupling foreground/background reconstruction, aligning multi-view point clouds via dense 3D-3D correspondences, and using the projected geometry as structural grounding for conditional video generation.

## Demo

<table>
<tr>
<td align="center"><b>Input Video</b></td>
<td align="center"><b>Output Video</b></td>
</tr>
<tr>
<td align="center"><video src="https://vveicao.github.io/projects/freeorbit4d/assets/camel/input_video.mp4" autoplay loop muted playsinline width="320"></video></td>
<td align="center"><video src="https://vveicao.github.io/projects/freeorbit4d/assets/camel/output_video.mp4" autoplay loop muted playsinline width="320"></video></td>
</tr>
<tr>
<td align="center"><video src="https://vveicao.github.io/projects/freeorbit4d/assets/breakdance/input_video.mp4" autoplay loop muted playsinline width="320"></video></td>
<td align="center"><video src="https://vveicao.github.io/projects/freeorbit4d/assets/breakdance/output_video.mp4" autoplay loop muted playsinline width="320"></video></td>
</tr>
<tr>
<td align="center"><video src="https://vveicao.github.io/projects/freeorbit4d/assets/unitree/input_video.m4v" autoplay loop muted playsinline width="320"></video></td>
<td align="center"><video src="https://vveicao.github.io/projects/freeorbit4d/assets/unitree/output_video.mp4" autoplay loop muted playsinline width="320"></video></td>
</tr>
</table>

### Multiple Trajectories from a Single Input

<table>
<tr>
<td align="center"><b>Input Video</b></td>
<td align="center"><b>Trajectory #1</b></td>
<td align="center"><b>Trajectory #2</b></td>
</tr>
<tr>
<td align="center"><video src="https://vveicao.github.io/projects/freeorbit4d/assets/two_views/bear/original_images.mp4" autoplay loop muted playsinline width="240"></video></td>
<td align="center"><video src="https://vveicao.github.io/projects/freeorbit4d/assets/two_views/bear/output_video_1.mp4" autoplay loop muted playsinline width="240"></video></td>
<td align="center"><video src="https://vveicao.github.io/projects/freeorbit4d/assets/two_views/bear/output_video_2.mp4" autoplay loop muted playsinline width="240"></video></td>
</tr>
<tr>
<td align="center"><video src="https://vveicao.github.io/projects/freeorbit4d/assets/two_views/duck/original_images.mp4" autoplay loop muted playsinline width="240"></video></td>
<td align="center"><video src="https://vveicao.github.io/projects/freeorbit4d/assets/two_views/duck/output_video_1.mp4" autoplay loop muted playsinline width="240"></video></td>
<td align="center"><video src="https://vveicao.github.io/projects/freeorbit4d/assets/two_views/duck/output_video_2.mp4" autoplay loop muted playsinline width="240"></video></td>
</tr>
<tr>
<td align="center"><video src="https://vveicao.github.io/projects/freeorbit4d/assets/two_views/hike/original_images.mp4" autoplay loop muted playsinline width="240"></video></td>
<td align="center"><video src="https://vveicao.github.io/projects/freeorbit4d/assets/two_views/hike/output_video.mp4" autoplay loop muted playsinline width="240"></video></td>
<td align="center"><video src="https://vveicao.github.io/projects/freeorbit4d/assets/two_views/hike/output_video_2.mp4" autoplay loop muted playsinline width="240"></video></td>
</tr>
</table>

### Interactive 4D Reconstruction

Explore our 4D reconstructions interactively in your browser:

[![Camel](https://img.shields.io/badge/4D_Viewer-Camel-orange.svg)](https://vveicao.github.io/projects/freeorbit4d/build/?playbackPath=https://vveicao.github.io/projects/freeorbit4d/assets/camel/camel_4d_v14.viser&initDistanceScale=1&initHeightOffset=0.0)
[![Breakdance](https://img.shields.io/badge/4D_Viewer-Breakdance-orange.svg)](https://vveicao.github.io/projects/freeorbit4d/build/?playbackPath=https://vveicao.github.io/projects/freeorbit4d/assets/breakdance/breakdance_4d.viser&initDistanceScale=1&initHeightOffset=0.0)
[![Unitree](https://img.shields.io/badge/4D_Viewer-Unitree-orange.svg)](https://vveicao.github.io/projects/freeorbit4d/build/?playbackPath=https://vveicao.github.io/projects/freeorbit4d/assets/unitree/unitree_4d.viser&initDistanceScale=1&initHeightOffset=0.0)

For more results, please visit our [project page](https://cvmlgroup.web.illinois.edu/freeorbit4d/).

## Code Coming Soon

We are actively cleaning and organizing the codebase. Stay tuned!

**Star this repo to get notified when the code is released.**

## Citation

```bibtex
@article{cao2026freeorbit4d,
  title={FreeOrbit4D: Training-Free Arbitrary Camera Redirection for Monocular Videos via Geometry-Complete 4D Reconstruction},
  author={Cao, Wei and Zhang, Hao and Tian, Fengrui and Wu, Yulun and Li, Yingying and Wang, Shenlong and Yu, Ning and Liu, Yaoyao},
  journal={arXiv preprint arXiv:2601.18993},
  year={2026}
}
```
