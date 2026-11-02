# Body Motion with ViPE cameras

[Inference guide](DEMO.md#automatic-body-video) · [Installation](INSTALL.md)

For a moving camera, select `--camera-mode vipe`. GEM runs
[ViPE](https://github.com/nv-tlabs/vipe) automatically on the selected video clip,
then uses its camera rotations and intrinsics for Body Motion:

```bash
gem-infer motion-video --task body --input video.mp4 \
  --config docs/configs/body_video_native77.yaml \
  --camera-mode vipe --vipe-executable /path/to/vipe/.venv/bin/vipe \
  --start 0 --duration 5 --device cuda --postprocess \
  --soma-assets assets/soma --output motion.pt
```

Prepare the [GEM checkpoints](PRETRAINED_MODELS.md#download-checkpoints),
[SOMA assets](INSTALL.md#soma-code-and-assets), and ViPE installation below first.
Use your ViPE executable's absolute path. If `vipe` is already on PATH, omit
`--vipe-executable`. `--postprocess` enables optional Contact/IK correction.
Render the saved result with the usual Motion renderer:

```bash
gem-render --task body --input motion.pt --soma-assets assets/soma \
  --device cuda --output-dir outputs/motion
```

## Install ViPE separately

Follow the official [ViPE installation instructions](https://github.com/nv-tlabs/vipe/blob/main/docs/installation.md)
in a separate environment. ViPE builds native CUDA extensions and requires
CUDA-enabled PyTorch and a CUDA toolkit with `nvcc`. Keeping its environment
separate avoids changing GEM's PyTorch and native dependencies. ViPE is optional
and is not installed by GEM's `video` extra.

For example, the upstream source installation uses:

```bash
git clone https://github.com/nv-tlabs/vipe.git /path/to/vipe
cd /path/to/vipe
conda env create -f envs/cu128.yml
conda activate cu128
uv sync
/path/to/vipe/.venv/bin/vipe --help
```

Return to the GEM checkout and activate the GEM environment to run `gem-infer`.
The absolute ViPE executable selects its own Python environment. Use a ViPE
installation that includes the `pose_only_long` pipeline, GEM's default.
`--vipe-pipeline NAME` selects another compatible pinhole pipeline.

## Choose the camera mode

| Mode | Use |
| --- | --- |
| `vipe` | Automatically estimate camera motion and intrinsics with ViPE. |
| `static` | Fixed camera; use identity camera rotations. |
| `background` | Existing approximate SIFT/homography estimate; retained as the default. |
| `provided` | Use a configured `camera.rotations` file on the normalized timeline. |

For a fixed camera, replace the ViPE options with `--camera-mode static`:

```bash
gem-infer motion-video --task body --input video.mp4 \
  --config docs/configs/body_video_native77.yaml \
  --camera-mode static --device cuda --output motion.pt
```

You can instead put the ViPE settings in your Motion configuration:

```yaml
camera:
  mode: vipe
  vipe:
    executable: /path/to/vipe/.venv/bin/vipe
    pipeline: pose_only_long
```

CLI camera options override the configuration. The optional
[`scripts/demo_motion_vipe.sh`](../scripts/demo_motion_vipe.sh) shortcut supplies
`motion-video --task body --camera-mode vipe`; pass the remaining arguments from
the first example unchanged. The regular `gem-infer` command is sufficient.

## Camera and timeline conventions

GEM trims the requested interval and normalizes it to **30 fps before running
ViPE**. Camera estimates, person tracking, Body Pose features, and Motion share
that clip and frame IDs. ViPE intrinsics are used for both Body Pose feature
extraction and Motion, unless you explicitly set `camera.intrinsics` to a
constant 3×3 calibration matrix for the normalized image.

This integration supports **pinhole video**. Rectify distorted wide-angle or
fisheye footage to a pinhole video before running GEM. Selecting ViPE's
`wide_angle` pipeline does not add a Body Motion distortion model.

The released native77 final775 model uses camera rotation conditioning. ViPE
camera-to-world poses are retained as metadata, but their translation and depth
are not consumed by this model. Its predicted world motion is therefore not an
absolute trajectory anchored to ViPE's reconstructed scene.

If ViPE fails or produces incompatible camera output, inference reports the
error. It does not silently switch to static or background cameras.
