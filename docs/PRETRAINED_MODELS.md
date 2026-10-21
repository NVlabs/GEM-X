# Pretrained Models

[Overview](../README.md) · [Installation](INSTALL.md) ·
[Inference guide](DEMO.md) · [Model overview](MODEL_OVERVIEW.md)

GEM provides **Body Pose**, **Hand Pose**, and **Body Motion**. Supply model
checkpoints, detector weights and SOMA geometry assets locally; they are separate
from the Python package. The paths in this guide are examples of local file
placement.

The Body Pose and Body Motion download examples below use the unified release from
[nvidia/GEM-X on Hugging Face](https://huggingface.co/nvidia/GEM-X). Download only
the files for your task into the same local `weights/` directory.

| Model | Reference checkpoint/configuration | Configuration choice | Role |
| --- | --- | --- | --- |
| Body Pose | `body_s240000.ckpt` | `body` | Single-image camera-frame body pose and frozen features for Motion |
| Hand Pose | `egohand_s060000.ckpt` | `hand` | Single-image hand pose; also runs frame by frame on video |
| Body Motion — native77 | `motion_native77_e774_s292175.ckpt` | [Native77 video example](configs/body_video_native77.yaml) | Temporal body motion from 77 2D keypoint observations and Body Pose features |
| Body Motion — native29 alternate | Separate native29 checkpoint with matching settings | [Native29 video example](configs/body_video_native29.yaml) | Temporal body motion from 29 selected 2D keypoint observations and matching Body Pose features |

Native77 Body Motion uses Body Pose `s240000` features and the original
`vitpose.pth` observation model at inference. Its detector identifier is
`vitpose77-dinov3-huge-metrosim-256x192-v1`. This release does not include
Sapiens-conditioned Motion weights.

Native29 and native77 are Body Motion observation layouts with different learned
embeddings. Each requires its own matching checkpoint; changing the joint count
does not convert between them. A *native training checkpoint* means a full
checkpoint saved by the GEM trainer and is independent of these layout names.

## Download checkpoints

Install the Hugging Face CLI if needed:

```bash
python -m pip install huggingface_hub
```

For Body Pose:

```bash
hf download nvidia/GEM-X body_s240000.ckpt body_config.json --local-dir weights
```

Hand Pose weights come from the official
[nvidia/EgoHand-v1 repository](https://huggingface.co/nvidia/EgoHand-v1).
Access is managed by that repository. Sign in with a Hugging Face account that
has been granted access before downloading the matching checkpoint:

```bash
hf auth login
hf download nvidia/EgoHand-v1 egohand_s060000.ckpt --local-dir weights
```

For automatic hand boxes, download the detector from
[nvidia/RT-DETR-Hand-Detector-v1](https://huggingface.co/nvidia/RT-DETR-Hand-Detector-v1):

```bash
hf download nvidia/RT-DETR-Hand-Detector-v1 hand_detector.onnx --local-dir weights
```

For Body Motion from prepared conditions:

```bash
hf download nvidia/GEM-X motion_native77_e774_s292175.ckpt \
  motion_stats_raw.npz motion_native77.yaml --local-dir weights
```

For automatic RGB-video Motion, also download its Body Pose feature producer,
original ViTPose observation model and video configuration:

```bash
hf download nvidia/GEM-X body_s240000.ckpt vitpose.pth \
  body_video_native77.yaml --local-dir weights
```

The downloaded video YAML resolves checkpoints relative to `weights/`. The
[repository example](configs/body_video_native77.yaml) resolves the same files
from `docs/configs/`; keep each configuration's relative paths when using it.
Automatic RGB-video inference also needs the person detector below. For
Contact/IK postprocessing or mesh rendering, follow the versioned
[SOMA asset download steps](INSTALL.md#soma-code-and-assets). Automatic Hand
boxes use the separately distributed [RT-DETR Hand detector](DEMO.md#automatic-hand-boxes).

<a id="person-detector-for-body-motion"></a>

### Person detector for Body Pose and Motion

Automatic Body Pose and Body Motion share the YOLOX-X HumanArt ONNX model from
[OpenMMLab's published ONNX archive](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_x_8xb8-300e_humanart-a39d44ed.zip).
Download the archive and extract its `end2end.onnx` to the configured filename.
Run these commands from the GEM repository root:

```bash
mkdir -p weights
curl --fail --location \
  https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_x_8xb8-300e_humanart-a39d44ed.zip \
  --output weights/yolox_humanart.zip
python - <<'PY'
from pathlib import Path
from shutil import copyfileobj
from zipfile import ZipFile

with ZipFile("weights/yolox_humanart.zip") as archive:
    member = next(name for name in archive.namelist() if name.endswith("/end2end.onnx"))
    with archive.open(member) as source, Path("weights/yolox_humanart.onnx").open("wb") as target:
        copyfileobj(source, target)
PY
```

Install `.[video]` for CPU ONNXRuntime. Body Pose uses
`weights/yolox_humanart.onnx` automatically when `--boxes` is omitted; override
its location with `--body-detector PATH` and its device with `--detector-device`.
The default detector device is `cpu`, independently of the Pose `--device`.
Automatic Body Pose selects one primary person; supplied boxes support multiple
people and do not need the detector. See [Body inference](DEMO.md#body-image).

Both Motion video configurations resolve `weights/yolox_humanart.onnx` from their
documented locations. If the detector is stored elsewhere, set
`detector.checkpoint` to that file; `detector.device: cpu` is independent of the
Pose/Motion `--device` setting.

For the selected native77 video workflow, `weights/` should contain
`body_s240000.ckpt`, `vitpose.pth`, `motion_native77_e774_s292175.ckpt`,
`motion_stats_raw.npz`, `body_video_native77.yaml` and `yolox_humanart.onnx`.
The extra `motion_native77.yaml` file is the configuration for cached-condition
inference. The person detector and SOMA assets come from the upstream locations
above and are separate from the GEM-X checkpoint bundle.

## Body Pose

The `body` preset specifies the DINOv3 ViT-H/16+ backbone, body head
and camera convention. It predicts 76 local body rotations, 128 identity
coefficients and 57 packed scale values: fixed global scale 1 plus 56 local
scales. The body training preset uses raw local56 output instead. Use the
release configuration that belongs to your checkpoint.

With a local Body Pose checkpoint, matching SOMA assets and the
[person detector](#person-detector-for-body-motion), run automatic inference:

```bash
gem-infer pose-image --task body --input image.jpg \
  --release body --checkpoint weights/body_s240000.ckpt \
  --soma-assets /path/to/soma-assets --device cuda --output body.pt
```

Use `pose-video --task body` for the same model on video frames. Both automatic
Pose workflows need no ViTPose or Body Motion checkpoint. Supply
[boxes and camera intrinsics](DEMO.md#supplied-body-boxes) with `--boxes` to bypass
the detector, or `--intrinsics` to replace the default camera estimate.

The Body Pose checkpoint includes backbone and head weights. Inference requires no
separate DINO backbone download. Its SOMA 0.3.3 compatibility profile requires
the matching body rig, procedural/reference data and corrective assets. See
[SOMA code and assets](INSTALL.md#soma-code-and-assets).

## Hand Pose: s060000

**Weight source and availability:** Use the official
[nvidia/EgoHand-v1 repository](https://huggingface.co/nvidia/EgoHand-v1)
for `egohand_s060000.ckpt`, using an account authorized to access it. Hand weights
are not distributed in the GEM-X bundle. The examples below use the downloaded
checkpoint in `weights/`.

Use `--release hand` with `egohand_s060000.ckpt`. This is also the default preset
for `--task hand`; its model identifier is `gem-pose-hand-ga-s060000`. It
reconstructs left and right hands.

Prepare a [hand box annotation](DEMO.md#hand-image-or-framewise-video) and the
`SOMAHand.npz` asset with its MHR identity model and mesh assets, then run:

```bash
gem-infer pose-image --task hand --input image.jpg --boxes hand_boxes.json \
  --release hand --checkpoint weights/egohand_s060000.ckpt \
  --soma-assets /path/to/soma-assets --device cuda --output hands.pt
```

The same checkpoint runs through `pose-video` frame by frame, with optional
tracking, box stabilization and parameter smoothing.

The model uses DINOv3 ViT-H/16+ with camera-ray positional features, log-lambda
depth and a separate 25-point 2D decoder. It predicts a wrist rotation, 24 finger
rotations, 5 identity coefficients and 27 scale values (one global plus 26 local
MHR controls). Calibrated 2D predictions and SOMA joints determine lateral camera
translation; reconstruction preserves the physical hand side. See
[Model overview](MODEL_OVERVIEW.md#gem-pose) for the architecture.

For calibrated ego cameras, `--camera FILE --task hand --hand-view ego` uses
direct PINHOLE inference or full-image undistortion for distorted cameras. The
latter requires a local RT-DETR hand detector. Optional `--perspective-crop`
supports per-region rectification and supplied source-image boxes with distorted
calibration. See [calibrated cameras](DEMO.md#calibrated-ego-hand-cameras) for
camera formats and source-coordinate output behavior.

For fine-tuning, [hand_finetune.yaml](configs/hand_finetune.yaml) loads the same
backbone, parameter head, separate 2D head and learned RLE density through
`model.initialization`. The matching `hand_training_config()` adds training
stochastic depth without changing the architecture. Use a complete checkpoint
that includes the learned RLE density weights; a prediction-only export without
them cannot initialize this training objective. The matching
`egohand_s060000.ckpt` includes those density weights. Follow the
[Hand fine-tuning guide](HAND_TRAINING.md#egohand-fine-tuning) to supply real
prepared data, including `img_size`, and start a new training run from these weights.

The official GA checkpoint has been verified with strict model loading,
real-image left/right inference and a short FP16 fine-tuning run that updates
the backbone, parameter head, 2D head and learned RLE density. Saving,
restoring native GEM training state and running inference from the resulting
checkpoint also passed. The training check uses synthetic prepared samples to
verify optimization and recovery; it does not measure dataset accuracy.

## Body Motion: choose a matched set

Use the downloaded `weights/body_video_native77.yaml` or the
[repository example](configs/body_video_native77.yaml). Both select the original
ViTPose weights and their release identifier. Set local paths if you place assets
elsewhere; paths resolve relative to the configuration file. The matched set contains:

- **Motion checkpoint and network configuration.** Match the architecture,
  trained observation layout, 655-coordinate output codec and physical-time
  settings. The examples use 1024-dimensional image features and a 512-dimensional,
  12-layer temporal network.
- **Normalization statistics.** Historical checkpoints require the exact
  `motion_stats_raw.npz` used by that release. Full native GEM training checkpoints
  include the codec buffers; for those, remove `motion.stats` from the video
  configuration. The loader rejects an additional statistics sidecar when the
  checkpoint already supplies these buffers.
- **Body Pose feature producer.** The examples use Body Pose `s240000` and its final
  pose query token. Its feature identifier is
  `gem-pose-body-s240000-v1:pose_tokens`. A different checkpoint,
  crop convention or feature tensor is not an interchangeable source merely
  because it also has 1024 channels.
- **Native77 ViTPose observation model.** The original `vitpose.pth` uses
  `vitpose77-dinov3-huge-metrosim-256x192-v1` in both `observations.release_id`
  and `motion.condition_spec.observation_source`. Keep these identifiers matched
  to the actual detector weights. The separate native29 template retains
  caller-supplied detector identifiers. Both Motion variants start from native77
  detections; native29 selects its defined subset, while native77 retains all points.
- **Person detector.** Automatic video inference additionally uses the configured
  local YOLOX ONNX checkpoint for person detection and tracking.

Use `motion.condition_spec.observation_joint_set: soma77_native_v1` with
`motion.network.obs_num_joints: 77` for native77. The
[native29 example](configs/body_video_native29.yaml) uses `soma29_nofinger_v1`
and `obs_num_joints: 29`, and requires separately trained native29 weights.
Preserve the camera/time settings and Body Pose feature identifier from the
checkpoint's training recipe.

Install the `video` extra and [FFmpeg](INSTALL.md#optional-dependencies), then
complete the [checkpoint downloads](#download-checkpoints),
[person detector setup](#person-detector-for-body-motion) and
[SOMA asset setup](INSTALL.md#soma-code-and-assets) for the desired output.
For camera/world parameters with Contact/IK postprocessing:

```bash
python -m pip install -e '.[video]'
gem-infer motion-video --task body --input video.mp4 \
  --config weights/body_video_native77.yaml --device cuda \
  --postprocess --soma-assets assets/soma --output motion.pt
```

On the supported Linux x86-64 rendering environment, install `.[render]` and the
[Mesa EGL dependencies](INSTALL.md#rendering-dependencies), then render:

```bash
gem-render --task body --input motion.pt --soma-assets assets/soma \
  --output-dir rendered_motion
```

Omit `--postprocess --soma-assets assets/soma` from the inference command for raw
Motion parameters without geometry postprocessing.

The command produces camera/world body parameters and retains prepared conditions
for replay. Cached-condition inference can use
`motion-cache` without rerunning either image model or the detectors. Its release
configuration contains the Motion `network`, `condition_spec` and optional
`time_options` fields directly; its historical statistics sidecar is supplied
with `--stats`. See [cached Motion](DEMO.md#cached-temporal-motion) for the full
contract and command. SOMA assets are needed when reconstructing meshes or applying
contact/IK, rather than for the parameter-only Motion forward.

## Initialization, inference and full resume

Choose the loading mode that matches your task:

| Task | Checkpoint requirement |
| --- | --- |
| Inference | Supported model weights and their matching release configuration |
| Pose fine-tuning | Complete model weights loaded through `model.initialization`; starts a new training run |
| Full resume | A GEM training checkpoint and compatible training/data configuration; restores optimizer, sampler and loop state |

The downloaded Body file contains model weights for inference and fine-tuning
initialization. The Motion file contains inference weights. These GEM-X release
exports omit optimizer, sampler and training-loop state; they cannot resume an
interrupted training run.

Motion inference weights require a caller-supplied initialization path for
training. See [Recovery](TRAINING.md#recovery) for the supported loading modes.

Native Pose training checkpoints save the release configuration, so Pose
inference can omit `--release` to use that saved configuration. Body weights use
the matching Body preset or explicit release JSON/YAML; Hand weights use `hand`.

See [Training & Evaluation](TRAINING.md) for the three training paths,
[Body training](POSE_TRAINING.md) and [Hand training](HAND_TRAINING.md) for their
data contracts, and [Evaluation](EVAL.md) for the supported benchmark protocols.
