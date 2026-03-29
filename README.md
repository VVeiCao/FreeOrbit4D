<p align="center">
  <img src="assets/title.png" width="400">
</p>

### <p align="center">[FreeOrbit4D: Training-Free Arbitrary Camera Redirection for Monocular Videos via Geometry-Complete 4D Reconstruction](https://arxiv.org/abs/2601.18993)</p>

<p align="center">
  <a href="https://arxiv.org/abs/2601.18993"><img src="https://img.shields.io/badge/arXiv-2601.18993-b31b1b.svg"></a>
  <a href="https://arxiv.org/pdf/2601.18993"><img src="https://img.shields.io/badge/Paper-PDF-blue.svg"></a>
  <a href="https://cvmlgroup.web.illinois.edu/freeorbit4d/"><img src="https://img.shields.io/badge/Project-Page-green.svg"></a>
</p>

<p align="center">
  <a href="https://vveicao.github.io/">Wei Cao</a><sup>1</sup>,
  <a href="https://haoz19.github.io/">Hao Zhang</a><sup>1</sup>,
  <a href="https://tianfr.github.io/">Fengrui Tian</a><sup>2</sup>,
  <a href="https://yulunwu0108.github.io/">Yulun Wu</a><sup>1</sup>,
  <a href="https://www.yingying.li/">Yingying Li</a><sup>1</sup>,
  <a href="https://shenlong.web.illinois.edu/">Shenlong Wang</a><sup>1</sup>,
  <a href="https://ningyu1991.github.io/">Ning Yu</a><sup>3</sup>,
  <a href="https://yaoyaoliu.web.illinois.edu/">Yaoyao Liu</a><sup>1</sup>
</p>

<p align="center">
  <sup>1</sup>University of Illinois Urbana-Champaign, <sup>2</sup>University of Pennsylvania, <sup>3</sup>Eyeline Labs
</p>

<p align="center"><b>TL;DR:</b> FreeOrbit4D is an effective training-free method for large-angle camera redirection via geometry-complete 4D proxy.</p>

<div align="center">
<table>
<tr>
<td align="center"><b>Input Video</b></td>
<td align="center"><b>Interactive 4D</b><br><sub>(click to explore)</sub></td>
<td align="center"><b>Output Video</b></td>
</tr>
<tr>
<td align="center"><img src="assets/demo/camel_input.gif" width="280"></td>
<td align="center"><a href="https://vveicao.github.io/projects/freeorbit4d/build/?playbackPath=https://vveicao.github.io/projects/freeorbit4d/assets/camel/camel_4d_v14.viser&initDistanceScale=1&initHeightOffset=0.0"><img src="assets/demo/camel_inter.png" width="280"></a></td>
<td align="center"><img src="assets/demo/camel_output.gif" width="280"></td>
</tr>
<tr>
<td align="center"><img src="assets/demo/breakdance_input.gif" width="280"></td>
<td align="center"><a href="https://vveicao.github.io/projects/freeorbit4d/build/?playbackPath=https://vveicao.github.io/projects/freeorbit4d/assets/breakdance/breakdance_4d.viser&initDistanceScale=1&initHeightOffset=0.0"><img src="assets/demo/breakdance_inter.png" width="280"></a></td>
<td align="center"><img src="assets/demo/breakdance_output.gif" width="280"></td>
</tr>
<tr>
<td align="center"><img src="assets/demo/unitree_input.gif" width="280"></td>
<td align="center"><a href="https://vveicao.github.io/projects/freeorbit4d/build/?playbackPath=https://vveicao.github.io/projects/freeorbit4d/assets/unitree/unitree_4d.viser&initDistanceScale=1&initHeightOffset=0.0"><img src="assets/demo/unitree_inter.png" width="280"></a></td>
<td align="center"><img src="assets/demo/unitree_output.gif" width="280"></td>
</tr>
</table>
</div>

### Multiple Trajectories from a Single Input

<div align="center">
<table>
<tr>
<td align="center"><b>Input Video</b></td>
<td align="center"><b>Trajectory #1</b></td>
<td align="center"><b>Trajectory #2</b></td>
</tr>
<tr>
<td align="center"><img src="assets/demo/bear_input.gif" width="240"></td>
<td align="center"><img src="assets/demo/bear_traj1.gif" width="240"></td>
<td align="center"><img src="assets/demo/bear_traj2.gif" width="240"></td>
</tr>
<tr>
<td align="center"><img src="assets/demo/duck_input.gif" width="240"></td>
<td align="center"><img src="assets/demo/duck_traj1.gif" width="240"></td>
<td align="center"><img src="assets/demo/duck_traj2.gif" width="240"></td>
</tr>
<tr>
<td align="center"><img src="assets/demo/hike_input.gif" width="240"></td>
<td align="center"><img src="assets/demo/hike_traj1.gif" width="240"></td>
<td align="center"><img src="assets/demo/hike_traj2.gif" width="240"></td>
</tr>
</table>
</div>

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
