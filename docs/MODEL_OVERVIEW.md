# Model Overview

GEM-X provides three models: **Body Pose**, **Hand Pose**, and **Body Motion**.
Body and Hand Pose reconstruct one image crop at a time. Body Motion predicts a
sequence using image features, 2D keypoint observations, and camera conditions.

| Model | Input | Output | Temporal behavior |
| --- | --- | --- | --- |
| Body Pose | RGB crop, bounding box, camera intrinsics | SOMA body parameters, camera-space joints and mesh | Independent image prediction |
| Hand Pose | RGB crop, hand side, bounding box, camera intrinsics | SOMA hand parameters, camera-space joints and mesh | Independent image prediction; optional tracking and smoothing |
| Body Motion | Body Pose features, 2D keypoint observations, camera and timing conditions | SOMA body parameters in camera and world coordinates | Temporal prediction across the sequence |

Use [Pretrained Models](PRETRAINED_MODELS.md) to select weights,
[Inference](DEMO.md) to run them, and [Architecture](ARCHITECTURE.md) to locate the
implementation. The sections below describe model behavior and coordinate conventions.

## GEM-Pose

Body Pose and Hand Pose use a shared DINOv3 backbone implementation with separate
weights and task-specific decoders. A normalized RGB crop passes through the
backbone and decoder to produce native parameters, crop-camera parameters,
auxiliary predictions, and the final decoder query.

`PoseModel.forward` returns these learned outputs. `PoseModel.encode` exposes
image features and parameters in evaluation mode without constructing geometry.
`PosePredictor` adds image preparation, camera decoding, SOMA forward kinematics
(FK), and projection.

### Body Pose

The Body Pose model predicts 76 joint-local rotations, 128 identity
coefficients, and packed scales `[GS1, local56]`, with global scale fixed at 1.
The body training preset uses the corresponding raw `local56` output convention.
The release configuration selects the parameter and geometry conventions that
match the checkpoint.

Body Pose geometry is expressed in the original camera coordinate frame, in
meters. Camera translation is already included in the saved joints and vertices.
The final decoder query provides the 1024-dimensional image feature used by
Body Motion.

### Hand Pose

The `hand` preset uses 8-band camera-ray positional encoding (RayPE), giving
32 ray features per image token, together with CLIFF camera conditioning. Its
parameter head predicts:

- A wrist rotation and 24 finger-joint rotations.
- Five identity coefficients.
- A learned global scale and 26 signed local MHR controls.
- Log-lambda depth, where `lambda = box_size * Z / focal_x`.

A separate 25-point decoder predicts 2D coordinates, uncertainty, presence, and
evidence maps. The predictor combines these coordinates with SOMA joints to solve
lateral camera translation at the predicted depth. It uses regressed-camera
translation when the solve is invalid. Consumers should use the returned full
camera translation so that the solved X/Y values are retained.

The image model uses canonical-left crops. The geometry adapter reconstructs
the requested physical hand side with the corresponding SOMA rig. Saved
parameters, joints, and vertices describe that physical side in the original
camera axes; translation is already applied to geometry.

### Hand processing and cameras

Hand boxes can come from annotations, saved per-hand box files, or the RT-DETR
ONNX detector. Both processing profiles use the same Hand Pose checkpoint:

| Profile | Automatic video boxes | Default smoothing |
| --- | --- | --- |
| `--hand-view exo` | Independent anatomical class assignments per frame | Off |
| `--hand-view ego` | Complete-clip left/right association and gap filling | Tracked-box filtering and adaptive parameter smoothing |

Parameter smoothing filters wrist/finger rotations on SO(3), camera translation,
and shape using physical frame times. It reconstructs matching joints and vertices
from the filtered physical-side parameters. Raw predictions remain available,
and missing predictions or long timestamp gaps split tracks.

The ego profile enables pixel-aspect correction except when perspective crops are
selected. Body Pose can also use the shared `--square-pixels` adapter. Calibrated
ego-hand inputs support PINHOLE, KB4, FISHEYE624, and RATIONAL_POLYNOMIAL cameras:

| Camera path | Processing |
| --- | --- |
| PINHOLE with `--camera` | Ordinary pinhole detection and prediction |
| Distorted camera with `--camera` | Full-image undistortion, pixel-aspect correction, detection, and prediction |
| `--camera` with `--perspective-crop` | Source-image detection or supplied boxes, followed by a virtual perspective crop per hand |

The full-image distorted-camera path requires automatic hand detection. Supplied
source-image boxes use perspective crops. Both paths return boxes and raw/smoothed
keypoints in source-image pixels, with geometry in the original camera axes.
Direct pinhole results use mesh overlays; distorted-camera and perspective-crop
results use source-camera skeleton overlays. See
[calibrated hand cameras](DEMO.md#calibrated-ego-hand-cameras) for input formats
and options.

### Training and evaluation

Body and Hand Pose train from prepared, normalized RGB crops and task-specific
targets. Dataset producers provide cropping, labels, and canonical-left Hand
preparation. The SOMA geometry is frozen but differentiable with respect to
predicted parameters.

Hand training also learns an RLE density alongside the two prediction heads.
Complete fine-tuning initialization includes those density weights. Training
uses GT correspondence masks for translation supervision, while inference uses
predicted presence. Every prepared Hand row supplies the full-image `img_size`;
GT camera targets remain `[s, tx, ty]` and are converted within the log-lambda loss.

Native GEM training checkpoints support both full training recovery and direct
Pose inference. Geometry assets remain external. See [Body Pose training](POSE_TRAINING.md)
and [Hand Pose training](HAND_TRAINING.md) for the data and checkpoint contracts.

`gem-eval pose` provides native SOMA25 Hand metrics and a caller-defined Body
evaluator interface. [Evaluation](EVAL.md) distinguishes benchmark metrics from
training-objective validation.

## GEM-Motion

The temporal model consumes Body Pose query features, 2D keypoint observations,
bounding boxes, camera intrinsics/angular velocity, and physical-time inputs.

### Conditioning and outputs

Native77 uses all 77 2D keypoint observations in the SOMA joint layout. The alternate
native29 layout selects a fixed subset from the same native77 output and requires
its own trained checkpoint. Joint ordering and image feature identity must match
the model; the observation source identifies the detector used for inference.

The network predicts a 655-dimensional native SOMA representation, camera values,
and contact logits. The codec converts these outputs into body parameters, and
the pipeline constructs camera-space and world-space predictions. See
[matched model components](PRETRAINED_MODELS.md#body-motion-choose-a-matched-set)
for checkpoint, statistics, feature, and detector requirements.

`motion-video` tracks one person and normalizes a clip of up to 60 seconds to
30 Hz. Detection, feature extraction, and Motion inference run in stages. Saved
conditions can be reused by `motion-cache`, which retains the configured physical
clock. Training combines video and single-image batches with supervision suited
to each source; see [Motion training](TRAINING.md#motion).

### Contact/IK and visualization

Optional contact/IK postprocessing uses six contact logits to correct world
translation and apply two-pass cyclic coordinate descent over native leg/arm
chains. The corrected body pose is shared by camera/world outputs. Identity,
scales, root orientations, and camera-space translation retain their saved values.
Original parameters remain under `raw_predictions`; repeated processing starts
from those values.

Contact/IK requires one complete, unpadded sequence with at least two frames at
continuous 30 Hz. Its FK path uses the XLO/Warp rig, Motion procedural/repose
settings, temporal shape averaging, and IEEE FP32 matrix products, with correctives
disabled. Raw inference follows its configured timing contract independently.
See [postprocessing](DEMO.md#contactik-postprocessing) for usage.

`gem-render` reconstructs saved Motion parameters with the SOMA 0.3.3 MID profile
and pose correctives. Each frame retains its saved identity, scale, pose, and
translation. Camera overlays use the saved intrinsics; world views use a fixed
camera and a sequence-wide display offset. Rendering selects the saved raw or
postprocessed variant without running the learned models again.

## Runtime boundaries

SOMA-X 0.3.3 is installed through pip, with compatible geometry assets supplied
separately. GEM's adapters preserve the parameter and coordinate conventions of
each checkpoint. Image backbones use Transformers DINOv3; Body Motion has a
separate temporal network.

Detector, training/evaluation, and rendering dependencies are provided through
the extras described in [Installation](INSTALL.md).
