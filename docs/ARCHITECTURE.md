# Architecture

GEM-X separates shared primitives, single-frame Pose models, temporal Body Motion,
and application input/output handling. This guide maps those responsibilities to
source modules. See [Model Overview](MODEL_OVERVIEW.md) for model behavior and
[Inference](DEMO.md) for commands.

| Namespace | Responsibility |
| --- | --- |
| `gem/common` | Image backbones, geometry, detectors, camera models, rendering, and SOMA adapters |
| `gem/pose` | Body/Hand Pose models, prediction, training, and evaluation |
| `gem/motion` | Temporal network, conditioning, parameter decoding, training, and postprocessing |
| `gem/inference` | Media inputs, preprocessing, command-line orchestration, result files, and visualization |

```text
gem/
├── common/
│   ├── backbones/       # Shared image DINOv3 implementation/weight conversion
│   ├── detectors/       # YOLOX/ByteTrack and RT-DETR/native hand association
│   ├── pose2d/          # Native77 2D keypoint observation extraction
│   ├── geometry/        # Rotations and coordinate primitives
│   ├── camera.py        # Shared camera operations, with explicit conventions
│   ├── camera_models.py # Generic pinhole, rational and fisheye calibration
│   ├── undistortion.py  # Full-image rectification and source box mapping
│   ├── evaluation.py    # Shared similarity/scale alignment mathematics
│   ├── rendering.py     # Optional Open3D mesh renderer; explicit camera inputs
│   └── soma/            # Adapters around the installed SOMA package
├── pose/
│   ├── heads/           # Body/hand parameter decoders
│   ├── conditioning/    # CLIFF/RayPE inputs
│   ├── data/            # Crops, prepared training/benchmark data and recovery
│   ├── geometry/        # Task-specific body/hand FK and side reconstruction
│   ├── training/        # Body/hand objectives, optimization, recovery and CLI
│   ├── evaluation/      # Native Hand metrics, prepared-data runner and callbacks
│   ├── postprocess.py   # Adaptive hand parameter smoothing and FK replay
│   ├── smoothing.py     # Parameter filtering math
│   ├── model.py         # Backbone + task head; differentiable, no FK
│   └── predictor.py     # Parameters + camera + FK + projection
├── motion/
│   ├── network/         # Temporal regression network
│   ├── codec/           # Native SOMA representation and world rollout
│   ├── frontend/        # Observation schemas, normalization and physical time
│   ├── data/            # Generic cache, mixing and sequence loaders
│   ├── training/        # Losses, augmentation, recovery and CLI
│   ├── evaluation/      # Sequence metric aggregation callbacks
│   ├── geometry.py      # Camera/world mesh reconstruction from saved parameters
│   ├── postprocess/     # Contact translation, CCD IK and native FK/precision facade
│   ├── model.py         # Lightning training/validation integration
│   └── pipeline.py      # Temporal network + parameter/camera/world decoding
└── inference/           # Image/video/cached Motion and saved-result rendering
```

## Networks and backbones

Body and Hand Pose reuse `common/backbones` with independent weights and task
heads. The backbone encodes RGB images; `motion/network` models temporal
conditions. Body Motion consumes frozen Body Pose query features and has its own
checkpoint.

`ImageCondition.feature_version` identifies the Pose release and feature type as
`release:pose_tokens`. `MotionConditionSpec.from_pose` carries that identity into
Motion together with the detector source and native observation schema.

| Component | Responsibility |
| --- | --- |
| [pose/model.py](../gem/pose/model.py) | Backbone, task head, and parameter/feature outputs |
| [pose/predictor.py](../gem/pose/predictor.py) | Camera decoding, task-specific FK, and projection |
| [pose/training](../gem/pose/training) | Body/Hand objectives, optimization, and checkpoint recovery |
| [motion/pipeline.py](../gem/motion/pipeline.py) | Temporal inference, codec decoding, and world rollout |
| [motion/model.py](../gem/motion/model.py) | Lightning training and validation integration |
| [inference/cli.py](../gem/inference/cli.py) | Application command dispatch |

`gem-infer` calls the application layer, and `gem-render` consumes saved results.
Training uses `gem-train pose --task body|hand` or `gem-train motion`.
`gem-eval pose` evaluates prepared Pose data; Motion validation runs through
`gem-train motion --validate`. The source wrappers call the same package entry
points. See [Training & Evaluation](TRAINING.md) and [Evaluation](EVAL.md).

Each model owns its evaluation interface. `common/evaluation.py` contains shared
alignment mathematics; `pose/evaluation` owns the Pose runner and native Hand
metrics; `motion/evaluation` aggregates sequence metrics from caller-provided
evaluators. Dataset readers supply the corresponding prepared data.

## Hand input and output ownership

The hand path separates detection, model prediction, and optional temporal
processing:

| Stage | Module |
| --- | --- |
| RT-DETR ONNX inference and raw detections | [common/detectors/hand.py](../gem/common/detectors/hand.py) |
| Ego left/right association and gap filling | [common/detectors/hand_tracking.py](../gem/common/detectors/hand_tracking.py) |
| Tracked-crop filtering | [common/detectors/hand_boxes.py](../gem/common/detectors/hand_boxes.py) |
| Automatic and saved-box input providers | [inference/hand.py](../gem/inference/hand.py) |
| Camera projection and inverse projection | [common/camera_models.py](../gem/common/camera_models.py), [common/fisheye.py](../gem/common/fisheye.py) |
| Full-image rectification | [common/undistortion.py](../gem/common/undistortion.py) |
| Perspective crops and source-coordinate restoration | [inference/camera.py](../gem/inference/camera.py) |
| Adaptive parameter filtering and FK reconstruction | [pose/smoothing.py](../gem/pose/smoothing.py), [pose/postprocess.py](../gem/pose/postprocess.py) |
| Saved-result overlays | [inference/render_pose.py](../gem/inference/render_pose.py) |

For distorted cameras without perspective crops, both detection and Pose inference
receive full-image rectification followed by optional pixel-aspect correction.
Annotations remain in the rectified base-camera raster until prediction and
parameter smoothing finish. Source restoration then maps bounding boxes and
raw/smoothed keypoints back to the input image while retaining 3D camera axes.

With `--perspective-crop`, detection or supplied boxes remain in the source image.
A virtual pinhole view is constructed per hand, and physical wrist orientation,
geometry, and projection are restored afterward. These calibrated paths use the
explicit ego-hand profile. [Camera options](DEMO.md#calibrated-ego-hand-cameras)
describe the supported formats and source-coordinate outputs.

`HandGeometry.forward_physical` reconstructs saved physical-side parameters,
including smoothed poses, without applying canonical mirroring again. Filtered
parameters are stored in `smoothed.parameters` alongside matching geometry;
raw predictions remain available.

## Contact/IK ownership

`motion/postprocess/contact.py` implements contact translation correction and
two-pass CCD. `motion/postprocess/api.py` provides the XLO/Warp fixed-scale FK
adapter, native joint hierarchy, and matrix-precision scope. Complete temporal
sequences preserve shape averaging, procedural transforms, and bind-pose
reposing; correctives are disabled for this path.

`inference/timing.py` reads saved physical clocks. `inference/postprocess.py`
checks the continuous 30 Hz sequence contract, retains raw parameters, and
selects the active output variant. It updates world translation and the body
pose shared by camera/world outputs. Conditions, contact logits, identity, scale,
root orientations, and camera-space translation keep their existing meanings.

Rendering selects an existing `raw` or `postprocessed` Motion result; it does
not apply another correction. Pose rendering selects `raw` or `smoothed`.

## Geometry ownership

`common/geometry` provides task-independent mathematics, and `common/soma` adapts
the installed SOMA runtime. `pose/geometry` converts task-specific parameter
formats and reconstructs physical hand sides. Pose and Motion use explicit
geometry profiles to preserve their checkpoint conventions.

`motion/geometry.py` reconstructs saved per-frame parameters with the MID Motion
profile and pose correctives. `common/rendering.py` owns the optional Open3D
context and compositing, receiving explicit vertices, faces, intrinsics, and
camera extrinsics. `inference/render.py` handles source-video alignment, display
framing, and MP4/JSON output. The world display offset is fixed per sequence and
does not change saved predictions.

SOMA geometry is frozen during training but remains differentiable with respect
to predicted parameters. Geometry assets and landmark projectors are external
to learned model checkpoints.

## Dependencies and private code

The installed runtime uses GEM modules and declared package dependencies. SOMA-X
0.3.3 is installed through pip; Transformers supplies DINOv3.

The `hand` and `video` extras provide CPU ONNXRuntime for detection; a CUDA
detector uses a separately configured CUDA provider. The `render` extra provides
Open3D 0.19 for the Linux x86-64 rendering path. ARM64 inference can save results
for rendering in that environment. See [Installation](INSTALL.md) for platform
requirements and system dependencies.

Dataset preparation, annotation, fitting, and dataset-specific evaluation remain
application integrations. Public interfaces provide prepared-data loaders,
training objectives, native SOMA25 Hand metrics, generic evaluation callbacks,
and reusable mathematics.

`inference/body_video.py` connects RGB decoding, person tracking, native77
2D keypoint observations, frozen Pose features, and Motion inference.
`inference/camera_motion.py` estimates camera rotation. Models are loaded one
stage at a time, and RGB frames are read in batches. The cache retains all 77
keypoints; the native Motion frontend selects the configured observation layout.
