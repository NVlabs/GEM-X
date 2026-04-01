# Demo

## Pipeline Overview

```
Input Video → Human Detection (YOLOX) → 2D Keypoints (VitPose) → Features (SAM3D) → GEM Model → 3D Pose (SOMA)
                                              ↓                                                        ↓
                                      2D Keypoint Overlay                              (Optional) Retarget → G1 Robot Motion
```

All demo scripts use **[YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) + [ByteTrack](https://github.com/ifzhang/ByteTrack)** for person detection and tracking.

## Full 3D Pipeline (`demo_soma.py`)

Run full inference on a video:

```bash
python scripts/demo/demo_soma.py \
  --video path/to/video.mp4 \
  --output_root outputs \
  --ckpt inputs/pretrained/gem_soma.ckpt
```

> **Note:** The `--ckpt` argument is optional. If omitted, the script will automatically download the pretrained checkpoint from [HuggingFace](https://huggingface.co/nvidia/GEM-X).

### Arguments

| Argument | Default | Description |
|---|---|---|
| `--video` | — | Input video path (required) |
| `--ckpt` | `null` | Pretrained checkpoint path |
| `-s` / `--static_cam` | off | Assume static camera (disables VO) |
| `--output_root` | `outputs/demo_soma` | Root directory for outputs |
| `--verbose` | off | Save debug overlays (bbox, pose) |
| `--render_mhr` | off | Render MHR identity model |
| `--retarget` | off | Retarget motion to Unitree G1 robot (requires soma-retargeter) |

### Outputs

Results are saved to `<output_root>/<video_name>/`:

| File | Description |
|---|---|
| `0_kp2d77_overlay.mp4` | 2D keypoint overlay on input video |
| `<video_name>_1_incam.mp4` | In-camera mesh overlay |
| `<video_name>_2_global.mp4` | Global-coordinate render |
| `<video_name>_3_incam_global_horiz.mp4` | Side-by-side (or 2x2 grid with `--retarget`) |
| `preprocess/bbx.pt` | Detected bounding boxes |
| `preprocess/vitpose.pt` | 2D keypoints (77 joints) |
| `preprocess/hpe_results.pt` | Full 3D pose prediction |
| `<video_name>_retarget_g1.bvh` | G1 robot motion in BVH format (with `--retarget`) |
| `<video_name>_retarget_g1.csv` | G1 robot joint angles (with `--retarget`) |
| `<video_name>_4_g1_retarget.mp4` | G1 robot motion video (with `--retarget`) |

### Preprocessing Fallbacks

- When no pre-computed `bbx.pt` exists, the demo runs human detection via YOLOX + ByteTrack.
- If VO modules are unavailable, the demo falls back to a static camera trajectory.

## Accelerated Pipeline (`demo_soma_onnx.py`)

ONNX/TensorRT-accelerated variant of `demo_soma.py`. Replaces PyTorch inference with ONNX Runtime for VitPose, SAM-3D-Body, and the GEM denoiser.

```bash
python scripts/demo/demo_soma_onnx.py \
  --video path/to/video.mp4
```

### Prerequisites

ONNX models are automatically downloaded from [HuggingFace](https://huggingface.co/nvidia/GEM-X) on first run if not found locally. To export your own ONNX models instead:

```bash
python tools/export/export_vitpose_onnx.py
python tools/export/export_sam3db_onnx.py
python tools/export/export_denoiser_onnx.py --ckpt <path>
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| `--video` | — | Input video path (required) |
| `--ckpt` | `null` | Pretrained checkpoint path |
| `-s` / `--static_cam` | off | Assume static camera (disables VO) |
| `--output_root` | `outputs/demo_soma_onnx` | Root directory for outputs |
| `--verbose` | off | Save debug overlays |
| `--force_pytorch` | off | Force PyTorch inference even if ONNX/TRT available |
| `--no-imgfeat` | off | Skip SAM3DB, use 2D keypoints only |
| `--ddim` | off | DDIM sampling (50 steps) instead of regression — slower but higher quality |
| `--retarget` | off | Retarget motion to Unitree G1 robot |

### Outputs

Same as `demo_soma.py`. When `--retarget` is used, the final composite is a 2x2 grid (kp2d, incam, global, retarget).

## Humanoid Robot Retargeting (`--retarget`)

Retarget the recovered SOMA motion to a Unitree G1 humanoid robot:

```bash
python scripts/demo/demo_soma.py \
  --video path/to/video.mp4 \
  --retarget
```

This requires the soma-retargeter package (see [Installation](INSTALL.md)). The output includes a G1 robot motion video and joint angle CSV. When `--retarget` is used, the final composite video shows a 2x2 grid: 2D keypoints, in-camera mesh, global mesh, and G1 robot motion.

## 2D Keypoint-Only Demo (`demo_2d_keypoints.py`)

A lightweight demo that runs only detection and 2D keypoint extraction — no GEM model, no 3D rendering, no Hydra config.

```bash
python scripts/demo/demo_2d_keypoints.py \
  --video path/to/video.mp4
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| `--video` | — | Input video path (required) |
| `--output_dir` | `outputs/demo_2d_kp/<video_name>/` | Output directory |
| `--detector_name` | `vitdet` | Human detector: `vitdet` or `sam3` |
| `--conf_thr` | `0.5` | Confidence threshold for visualization |
| `--save_raw` | off | Keep intermediate `.pt` files |

### Output

- `<video_name>_kp2d77_overlay.mp4` — 2D keypoint overlay video

## Accessing Results Programmatically

```python
import torch

# Load 2D keypoints
vitpose = torch.load("outputs/demo_soma/<video>/preprocess/vitpose.pt")
# vitpose shape: (num_frames, 77, 3) — x, y, confidence

# Load bounding boxes
bbx = torch.load("outputs/demo_soma/<video>/preprocess/bbx.pt")
bbx_xyxy = bbx["bbx_xyxy"]  # (num_frames, 4)
bbx_xys = bbx["bbx_xys"]    # (num_frames, 3) — center_x, center_y, scale

# Load 3D prediction
pred = torch.load("outputs/demo_soma/<video>/preprocess/hpe_results.pt")
```
