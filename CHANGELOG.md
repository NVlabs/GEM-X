# Changelog

## Unreleased

- Run Body Pose on images or videos without supplied boxes, reusing YOLOX
  person detection and ByteTrack for the primary person. Preserve source video
  timing and retain explicit boxes for caller-selected or multiple people.
- Identify the selected Body Motion release as native77 final775
  (epoch 774, step 292175), with Body Pose features and 2D keypoint observations.
  The checkpoint bundle selects the original `vitpose.pth` observation model;
  existing teaser films retain their recorded random-blur epoch 10 detector.
  Native29 remains a supported alternate layout for
  separate matching weights; current demo instructions use native77 final775.
  Motion inference APIs accept only native29/native77 schemas, and release
  configurations explicitly specify the observation count.
- Select `egohand_s060000.ckpt` as the final Hand checkpoint for this release and
  update inference/fine-tuning examples. The `hand` preset now identifies it as
  `gem-pose-hand-ga-s060000`; architecture and processing modes are unchanged.
  Use the official [EgoHand repository](https://huggingface.co/nvidia/EgoHand-v1)
  for matching `s060000` Hand weights, with repository-authorized access.
  Provide Body Pose and Body Motion downloads and matching configurations from
  GEM-X. Those release exports omit full training-resume state. The existing
  Hand GIF and earlier source comparisons
  retain their `s040000` labels; see the
  [model guide](docs/PRETRAINED_MODELS.md#hand-pose-s060000) for validation scope.
- Add explicit ego/exo Hand Pose processing profiles and shared Body/Hand Pose
  square-pixel correction. `--camera` accepts PINHOLE, RATIONAL_POLYNOMIAL,
  KB4 and FISHEYE624 calibration: pinhole uses direct inference, while distorted
  cameras use full-image undistortion before detection and inference.
  Optional `--perspective-crop` instead rectifies each hand region after
  source-image detection. Saved boxes and raw/smoothed keypoints use source
  pixels; physical parameters and geometry use the source camera axes.
- Stabilize native ego-hand detector crops over the complete video with optional
  `--smooth-hand-boxes`, enabled by default for that processing profile.
  Supplied boxes retain their original tracks and gaps.
- Replace hand display averaging with adaptive translation, SO(3) rotation and
  shape filtering followed by SOMA geometry reconstruction. Save complete
  `smoothed.parameters` alongside matching geometry while retaining raw rows.
  `--smooth-hand` remains enabled for ego video and opt-in for exo video.
  GEM Hand architecture and `s040000` checkpoint compatibility are unchanged.
- Include the OpenMDW-1.1 license and notices for adapted Hand processing,
  documenting file and function exceptions to the default Apache 2.0 license.
- Unify Body Pose, Hand Pose and temporal Body Motion under shared backbone,
  camera and SOMA geometry interfaces, with task-specific model heads.
- Provide four inference workflows through `gem-infer`: `pose-image` and
  framewise `pose-video` for body or hand, plus `motion-video` and `motion-cache`
  for temporal body motion.
- Add shared `gem-train`, `gem-eval` and `gem-render` entry points for training,
  evaluation and rendering saved predictions.
- Support the optional Body Pose constant-then-cosine learning-rate schedule,
  complete scheduler recovery and configured training loggers.
- Extend online Pose evaluation with explicit evaluation steps, resume and
  natural-completion hooks, sample coverage and per-step JSON results.
- Provide a generic Pose training visualization callback for caller-owned
  samples, with PNG/W&B output and preservation of training RNG and model modes.
- Preserve FP32 signed-depth projection in Motion reprojection losses under AMP,
  preventing masked behind-camera points from producing nonfinite losses.
- Add the current Hand architecture with 8-band head RayPE, log-lambda depth,
  a separate 2D decoder and fixed-depth lateral translation solving. The single
  Hand preset is `hand`, with `hand_config()` / `hand_training_config()` APIs.
- Add Hand fine-tuning from `s040000`, including backbone, parameter/2D heads
  and learned RLE density initialization, separate learning rates and complete
  native checkpoint recovery.
- Preserve compatibility with supported Body Pose and native29/native77
  Body Motion pretrained checkpoints using their matching release
  configurations and normalization statistics.
- Refresh installation, pretrained-model, inference, training and evaluation
  documentation; retain the body demo GIFs and add an ARCTIC Hand Pose demo.

## v1.0.0 — Initial Release

- Monocular video 3D human pose estimation using SOMA body model (77 joints)
- Bundled 2D pose estimation model (77 SOMA keypoints)
- Demo scripts for full 3D pipeline (`demo_soma.py`) and 2D keypoint visualization (`demo_2d_keypoints.py`)
- Apache 2.0 licensed for commercial use
