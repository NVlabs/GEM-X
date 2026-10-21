# Inference

Use `gem-infer` for Body Pose, Hand Pose, and Body Motion. Choose a command by
input type and whether you need independent poses or temporal body motion.
All four commands require `--task`; Motion supports `body` only.

| Command | Use it for | Inputs | Task |
| --- | --- | --- | --- |
| `pose-image` | Pose from one image | Supplied boxes or automatic body/hand detection | `body`, `hand` |
| `pose-video` | Independent Pose predictions on video frames | Supplied tracks or automatic body/hand detection | `body`, `hand` |
| `motion-video` | Temporal Body Motion in camera and world coordinates | Video; detection, tracking, and features run automatically | `body` |
| `motion-cache` | Temporal Body Motion from prepared conditions, or contact/IK on saved predictions | A conditions or predictions `.pt` file | `body` |

`pose-video` runs the Pose model on each crop independently. Hand tracking and
smoothing are optional processing stages around that model; GEM does not include
a learned Hand Motion model. Body Motion uses temporal attention across a sequence.

Follow [Installation](INSTALL.md) and [Pretrained models](PRETRAINED_MODELS.md)
to prepare dependencies, weights, and SOMA assets, then choose a workflow:

- Pose: [Body image](#body-image), [Body video](#body-framewise-video), or
  [Hand image/video](#hand-image-or-framewise-video).
- Motion: [Automatic video](#automatic-body-video) or [cached conditions](#cached-temporal-motion).
- Optional output stages: [Contact/IK](#contactik-postprocessing),
  [Pose rendering](#render-saved-pose), and [Motion rendering](#render-saved-motion).
- Reference: [Calibrated hand cameras](#calibrated-ego-hand-cameras) and [output format](#outputs).

For all options, run `gem-infer <command> --help` or `gem-render --help`.
`python scripts/predict.py` and `python -m gem.inference` expose the same commands
as `gem-infer`.

## Body image

Install the `video` extra and download the
[YOLOX person detector](PRETRAINED_MODELS.md#person-detector-for-body-motion),
Body Pose checkpoint and matching SOMA assets. Omitting `--boxes` enables
automatic detection of the largest person in the image:

```bash
gem-infer pose-image --task body --input image.jpg \
  --release body --checkpoint weights/body_s240000.ckpt \
  --soma-assets assets/soma --device cuda --output body.pt
```

The detector defaults to `weights/yolox_humanart.onnx`, relative to the working
directory; override it with `--body-detector PATH`. `--detector-device` defaults
to `cpu` independently of the Pose model's `--device`. Automatic Body Pose uses
only the person detector, Body Pose checkpoint and SOMA assets; it does not
require ViTPose or a Body Motion checkpoint.

Without supplied camera intrinsics, Pose estimates a pinhole camera with
`fx = fy = max(image_width, image_height)` and the principal point at the image
center. For a calibrated camera, save a JSON/YAML file with an `intrinsics`
matrix and pass `--intrinsics camera.json`:

```json
{"intrinsics": [[800, 0, 320], [0, 800, 240], [0, 0, 1]]}
```

### Supplied Body boxes

To choose a particular person or reconstruct multiple people, supply
`--boxes boxes.json`. This bypasses automatic detection. Each square `bbox`
is `[center_x, center_y, size]` in full-image pixels:

```json
{
  "intrinsics": [[800, 0, 320], [0, 800, 240], [0, 0, 1]],
  "instances": [{"track_id": "person0", "bbox": [320, 240, 360]}]
}
```

```bash
gem-infer pose-image --task body --input image.jpg --boxes boxes.json \
  --release body --checkpoint weights/body_s240000.ckpt \
  --soma-assets assets/soma --device cuda --output body.pt
```

Add one instance with its own `track_id` for each person. `--intrinsics` overrides
the matrix in the annotation file.

Both Pose commands accept a named `--release` preset or a release JSON/YAML.
When you omit it, native `gem-train pose` checkpoints supply their architecture
and camera conventions through `gem_pose_training.release`. Checkpoints without
that metadata use `body` for body or `hand` for hand. For another Body
architecture, supply a matching release configuration explicitly.

## Body framewise video

Omit `--boxes` to detect and track one primary person automatically with YOLOX
and ByteTrack, then run Body Pose independently on each crop:

```bash
gem-infer pose-video --task body --input video.mp4 \
  --release body --checkpoint weights/body_s240000.ckpt \
  --soma-assets assets/soma --device cuda --output body_video.pt
```

This uses the same detector path and camera options as [Body image](#body-image).
The primary track has the largest cumulative bounding-box area over the clip.
Tracked boxes use gap interpolation, smoothing and square-crop preparation.
The source video's original frame indices and frame rate are preserved.
Body Pose produces camera-space geometry; it needs no ViTPose or Body Motion
weights. For learned temporal prediction and world-space motion, use
[Body Motion](#automatic-body-video).

To supply your own tracks, save frame-indexed boxes as `body_tracks.json` and
add `--boxes body_tracks.json`:

```json
{
  "intrinsics": [[800, 0, 320], [0, 800, 240], [0, 0, 1]],
  "frames": {
    "0": {"instances": [{"track_id": "person0", "bbox": [320, 240, 360]}]},
    "1": {"instances": [{"track_id": "person0", "bbox": [322, 240, 360]}]}
  }
}
```

```bash
gem-infer pose-video --task body --input video.mp4 --boxes body_tracks.json \
  --release body --checkpoint weights/body_s240000.ckpt \
  --soma-assets assets/soma --device cuda --output body_video.pt
```

Frame IDs are zero-based decoded source-video frames. Include every frame for
complete coverage, or annotate only the frames you need; unlisted frames remain
absent. Supplied tracks bypass the detector and preserve your frame coverage.
Multiple people in one frame need distinct `track_id` values.

Per-frame `intrinsics` overrides the shared matrix. `--intrinsics camera.json`
overrides intrinsics for the entire invocation. Output geometry is in camera
coordinates.

## Hand image or framewise video

Use `--task hand` with `weights/egohand_s060000.ckpt`, downloaded from the official
[nvidia/EgoHand-v1 repository](https://huggingface.co/nvidia/EgoHand-v1)
with an account authorized to access it. See
[Hand weights](PRETRAINED_MODELS.md#hand-pose-s060000) for the download command.
The `hand` release preset is selected automatically when no native checkpoint
metadata or explicit `--release` overrides it.

Choose one input provider:

| Provider | Option | Track behavior |
| --- | --- | --- |
| JSON/YAML annotations | `--boxes tracks.json` | Caller-supplied IDs and frame coverage |
| Per-hand box files | `--boxes boxes/` | Saved left/right boxes; missing frames stay absent |
| RT-DETR ONNX detector | `--hand-detector hand_detector.onnx` | Per-frame detection by default; wearer-view tracking with `--hand-view ego` |

For video annotations, include `side` on every instance. The same caller
`track_id` can appear on both sides:

```json
{
  "intrinsics": [[800, 0, 320], [0, 800, 240], [0, 0, 1]],
  "frames": {
    "0": {"instances": [{"track_id": "person0", "side": "left", "bbox": [220, 250, 150]}]},
    "2": {"instances": [
      {"track_id": "person0", "side": "left", "bbox": [225, 250, 150]},
      {"track_id": "person0", "side": "right", "bbox": [410, 250, 150]}
    ]}
  }
}
```

Save this as `tracks.json`. It predicts frames 0 and 2, leaving frame 1 absent.
Zero-padded keys such as `000002` and empty `instances` lists are accepted.
Per-frame intrinsics and `--intrinsics` work as in Body Pose.

```bash
gem-infer pose-video --task hand --input video.mp4 --boxes tracks.json \
  --checkpoint weights/egohand_s060000.ckpt \
  --soma-assets /path/to/soma-assets --device cuda --output hands.pt
```

For an image, use the [Body image](#body-image) structure with `instances`
directly, adding `"side": "left"` or `"side": "right"` to each instance:

```bash
gem-infer pose-image --task hand --input image.jpg --boxes hand_boxes.json \
  --checkpoint weights/egohand_s060000.ckpt \
  --soma-assets /path/to/soma-assets --device cuda --output hand.pt
```

The pinhole Hand pipeline mirrors right-hand pixels with `W−1−x`, transforms
intrinsics accordingly, and reconstructs the requested physical hand side.
Saved geometry already includes this reconstruction.

### Choose a hand processing profile

`--hand-view exo` is the default. Select `--hand-view ego` for the camera wearer's
own hands. Both profiles use the same checkpoint and architecture.

| Input/profile | Automatic boxes | Pixel-aspect correction | Tracked-box smoothing | Parameter smoothing |
| --- | --- | --- | --- | --- |
| Image, `exo` | Per-frame anatomical labels | Off | Off | Unavailable |
| Image, `ego` | Per-frame anatomical labels | On* | Off | Unavailable |
| Video, `exo` | Per-frame anatomical labels | Off | Off | Off |
| Video, `ego` | Native wearer-view tracker | On* | On with native detector tracking | On |

\* `--perspective-crop` handles camera geometry directly and disables pixel-aspect
resizing. Automatic boxes and tracked-box smoothing apply only to
`--hand-detector`; supplied boxes preserve the caller's tracks and gaps.

Override these defaults with `--square-pixels` / `--no-square-pixels`,
`--smooth-hand-boxes` / `--no-smooth-hand-boxes`, and
`--smooth-hand` / `--no-smooth-hand`.

### Automatic hand boxes

Detector weights are distributed at
[nvidia/RT-DETR-Hand-Detector-v1](https://huggingface.co/nvidia/RT-DETR-Hand-Detector-v1).

Install the `hand` extra and download the detector with an account that has
access to its model repository:

```bash
python -m pip install -e '.[hand]'
hf auth login
hf download nvidia/RT-DETR-Hand-Detector-v1 hand_detector.onnx \
  --local-dir checkpoints/hand_detector
gem-infer pose-video --task hand --input video.mp4 \
  --hand-detector checkpoints/hand_detector/hand_detector.onnx --detector-device cpu \
  --detector-batch-size 8 --hand-view ego \
  --checkpoint weights/egohand_s060000.ckpt --soma-assets /path/to/soma-assets \
  --device cuda --output hands.pt
```

The detector also works with `pose-image --task hand`. Its device defaults to
CPU independently of Pose's `--device`, and its batch size defaults to 8.
Pass local ONNX weights; inference does not download them automatically.

For ego video, the default `--hand-tracker native` associates one left/right
pair over the complete clip and enforces wearer-view screen order when both
hands appear. It fills internal gaps and holds endpoint boxes for any side
observed in the clip. A never-observed side produces no predictions.
`bbox_observed` distinguishes detections from filled boxes, including predictions
made while a hand is outside the visible image.

The exo profile selects the highest-scoring detection per anatomical side on
each frame, expands its square box by 1.3, and leaves missing hands absent.
Images and `--hand-view ego --hand-tracker none` use this per-frame policy too.
For multiple people, supply boxes with your own track IDs. Native tracking is
available only for ego video.

Use `--hands left`, `--hands right`, or `--hands left right` to select output
sides for the detector or saved-directory provider. Native tracking processes
both classes before selecting sides.

### Optional tracked-box smoothing

`--smooth-hand-boxes` filters crops **before** Hand inference. It defaults on for
ego video with `--hand-detector` and `--hand-tracker native`; explicit activation
requires that same combination. Use `--no-smooth-hand-boxes` to retain the native
tracker's boxes. Python detector adapters default to `smooth_boxes=False`.

The filter rejects isolated spikes, adapts center smoothing to motion, and
limits size changes in log space. Its tighter crop retains a margin around the
native box's central area. Associations, observed/filled labels, and side
selection stay intact. Filtering uses the complete clip, including future
frames for size limiting; metadata records `bbox_smoothing` settings.

### Saved per-hand box directories

Organize existing boxes by input filename stem:

```text
boxes/
└── video/
    ├── video_left_bbox.json
    └── video_right_bbox.json
```

For `video.mp4`, each file maps decoded frame IDs to `[cx, cy, size]`, for example
`{"000000": [220,250,150], "000002": [225,250,150]}`. An image uses its own stem
and frame `0`. At least one requested side file must exist. Missing sides and
frame entries remain absent; these boxes bypass tracking, padding, and box
smoothing.

```bash
gem-infer pose-video --task hand --input video.mp4 --boxes boxes/ \
  --checkpoint weights/egohand_s060000.ckpt --soma-assets /path/to/soma-assets \
  --device cuda --output hands.pt
```

For the detector and saved-directory providers, optionally supply
`--intrinsics camera.json` containing
`{"intrinsics": [[fx,0,cx],[0,fy,cy],[0,0,1]]}`. Otherwise, Pose estimates focal
length as `max(width,height)` and principal point as `(width/2,height/2)`.
Use calibrated values when available. For distorted source-image boxes, use
[`--camera` with `--perspective-crop`](#calibrated-ego-hand-cameras).

### Optional hand smoothing

`--smooth-hand` filters predicted SOMA parameters **after** inference. It defaults
on for ego hand video and off for exo video. `--no-smooth-hand` saves only raw
predictions.

The filter groups each `(track_id, side)` on the original decoded timeline,
using `frame_id / source_fps` in seconds. It smooths camera translation, wrist
and finger rotations on SO(3), and identity/scale parameters, then reconstructs
the physical hand and projects its joints through the source camera.

Filtering uses the complete clip; geometry batches do not split temporal
context. Missing predictions, nonfinite parameter rows, and adjacent timestamps
more than 0.15 seconds apart split tracks. Missing rows stay absent, and invalid
rows retain their raw view. Filter widths adapt to local motion; defaults are
experimental.

Raw fields and query tokens remain available alongside a complete `smoothed`
variant. See [Outputs](#outputs) for its fields and reconstruction contract.

### Pixel-aspect correction for Body and Hand Pose

Use `--square-pixels` with either Pose command and task when calibrated focal
lengths differ. Ego Hand enables it by default except with perspective crops;
other profiles require an explicit option. Distorted-camera inputs are
undistorted first.

```bash
gem-infer pose-video --task body --input video.mp4 --boxes body_tracks.json \
  --intrinsics camera.json --square-pixels \
  --checkpoint weights/body_s240000.ckpt \
  --soma-assets /path/to/soma-assets --device cuda --output body_video.pt
```

The adapter resizes the axis with the smaller focal length, updating intrinsics
and boxes together. A relative focal difference of at most `0.001` leaves the
image unchanged. Resizing accounts for integer output dimensions and OpenCV's
half-pixel sampling. Detection runs on corrected images; supplied boxes are
transformed in memory.

Saved geometry remains in the original camera frame; saved full-image
keypoints, boxes, and intrinsics refer to source pixels. Raw detector candidates
retain their corrected-image coordinates in provider metadata. Input files are
unchanged. This option applies to Pose; Motion uses its trained input convention.

<a id="optional-ego-hand-fisheye-camera-adapter"></a>
<a id="optional-ego-hand-perspective-camera-adapter"></a>

### Calibrated ego-hand cameras

Use `--camera camera.json` with `--task hand --hand-view ego`. Supported camera
models are **PINHOLE**, **RATIONAL_POLYNOMIAL**, **KB4**, and **FISHEYE624**.

| Camera | Default | With `--perspective-crop` |
| --- | --- | --- |
| PINHOLE | Ordinary pinhole detection and inference | Detect on the source image; rectify each hand region |
| RATIONAL_POLYNOMIAL, KB4, FISHEYE624 | Undistort the full image; detect and infer there | Detect on the source image; rectify each hand region |

Full-image undistortion retains raster dimensions and intrinsics, followed by
the ego profile's pixel-aspect correction. Its pinhole field of view can exclude
wide-angle source content. **Distorted cameras require `--hand-detector` for
this default route. Use `--perspective-crop` with supplied source-image boxes.**

Perspective crops aim a virtual pinhole camera at each hand and mirror right
hands in that view. They disable square-pixel resizing, so omit
`--square-pixels`. Use `--no-perspective-crop` to select the default route.
The aliases `--perspective-camera FILE` and `--fisheye-camera FILE` remain
shortcuts for `--camera FILE --perspective-crop`.

Calibration supplies the intrinsics; omit `--intrinsics` when using `--camera`.
Dimensions and orientation must match the decoded input. GEM's format uses
`model`, `width`, `height`, and a zero-skew `intrinsics` matrix:

```json
{
  "model": "pinhole",
  "width": 1280,
  "height": 960,
  "intrinsics": [[610, 0, 639.5], [0, 610, 479.5], [0, 0, 1]],
  "distortion_coefficients": []
}
```

Use your actual calibration. Model names are case-insensitive; coefficients
follow these conventions:

| Model | Coefficients in GEM's format |
| --- | --- |
| `pinhole` | `distortion_coefficients: []` |
| `rational_polynomial` | Fourteen `distortion_coefficients` in OpenCV order: `[k1,k2,p1,p2,k3,k4,k5,k6,s1,s2,s3,s4,tau_x,tau_y]` |
| `kb4` | Four `radial` coefficients |
| `fisheye624` | Six `radial` coefficients and optional six `tangential_prism` coefficients in `[p0,p1,s0,s1,s2,s3]` order; omitted terms are zero |

GEM also accepts the shared camera format with `image_width`, `image_height`,
`camera_matrix.matrix_3x3`, and a single `distortion_coefficients` vector:

```json
{
  "model": "KB4",
  "image_width": 1280,
  "image_height": 960,
  "camera_matrix": {
    "matrix_3x3": [[610, 0, 639.5], [0, 610, 479.5], [0, 0, 1]]
  },
  "distortion_coefficients": [0.01, -0.002, 0.0001, 0.0]
}
```

In this format, coefficient counts are 0, 14, 4, and 12 for PINHOLE,
RATIONAL_POLYNOMIAL, KB4, and FISHEYE624. FISHEYE624 packs six radial terms
followed by six tangential/prism terms. Convert application-specific calibration
fields to either supported format before inference.

```bash
gem-infer pose-video --task hand --hand-view ego --input video.mp4 \
  --hand-detector checkpoints/hand_detector/hand_detector.onnx \
  --camera camera.json --checkpoint weights/egohand_s060000.ckpt \
  --soma-assets /path/to/soma-assets --device cuda --output calibrated_hands.pt
```

Add `--perspective-crop` for per-region rectification. Both routes save 3D
geometry in the original camera axes and boxes/keypoints in source pixels.
Full-image undistortion filters parameters before restoring raw and smoothed
projections to the source image. Perspective crops restore translation,
geometry, and wrist orientation from the virtual camera to the source camera.
A crop outside the calibrated forward camera domain skips that hand/frame;
`skipped_instances` records its original box and reason.

Rendering these distorted-camera or explicit perspective-crop results uses a
source-image **skeleton overlay**, including for PINHOLE perspective crops.
Direct PINHOLE results use mesh overlays. See [Outputs](#outputs) for the saved
calibration and adapter metadata.

## Automatic body video

The selected Motion release is **native77 final775**, using native77 2D keypoint
observations and Body Pose `s240000` features. Its matched inference detector is
the original `vitpose.pth`, with identifier
`vitpose77-dinov3-huge-metrosim-256x192-v1`. Native29 is an alternate layout
requiring its own matching weights.

| Observation layout | Configuration | Network and condition schema |
| --- | --- | --- |
| Native77: selected final775 release | [body_video_native77.yaml](configs/body_video_native77.yaml) | `obs_num_joints: 77`, `soma77_native_v1` |
| Native29: alternate checkpoint | [body_video_native29.yaml](configs/body_video_native29.yaml) | `obs_num_joints: 29`, `soma29_nofinger_v1` |

Prepare the configuration before running:

1. Install the `video` extra and [FFmpeg](INSTALL.md#optional-dependencies), then
   follow [checkpoint downloads](PRETRAINED_MODELS.md#download-checkpoints) and
   [YOLOX extraction](PRETRAINED_MODELS.md#person-detector-for-body-motion).
   Checkpoint and statistics paths resolve relative to the YAML file.
2. Keep the original ViTPose identifier in both native77 detector-ID fields.
   For the separate native29 template, replace its `YOUR_NATIVE77_RELEASE_ID`
   placeholders with the identifier for that checkpoint's matched detector.
3. Keep the Body feature release, observation schema, and Motion checkpoint
   compatible. Changing `obs_num_joints` does not convert checkpoint layouts.

Both templates use YOLOX ONNX person detection, native77 2D keypoint observations,
and Body Pose features. The download instructions place the extracted detector
at `weights/yolox_humanart.onnx`. For Contact/IK postprocessing and mesh rendering,
also download the [SOMA v0.3.3 assets](INSTALL.md#soma-code-and-assets) into
`assets/soma/` and pass that directory with `--soma-assets`.
Native29 selects its 29-joint subset inside Motion. Another
Body feature release needs its matching configuration and feature ID.
`motion-video` uses these explicit settings rather than Pose's automatic
checkpoint release discovery.

```bash
gem-infer motion-video --task body --input video.mp4 \
  --config docs/configs/body_video_native77.yaml \
  --start 0 --duration 5 --device cuda --output motion.pt
```

To apply Contact/IK and render the result, prepare the SOMA assets and
[rendering dependencies](INSTALL.md#rendering-dependencies), then run:

```bash
gem-infer motion-video --task body --input video.mp4 \
  --config docs/configs/body_video_native77.yaml \
  --start 0 --duration 5 --device cuda --postprocess \
  --soma-assets assets/soma --output motion.pt
gem-render --task body --input motion.pt --soma-assets assets/soma \
  --output-dir rendered_motion
```

`--start` and `--duration` are in seconds; clips can be at most 60 seconds.
The default duration covers the remaining input when it fits within that limit.
Longer inputs require an explicit duration.

The pipeline uses FFmpeg to normalize the selected interval to **30 fps**,
tracks the person with the largest cumulative detected area, extracts native77
2D keypoint observations and Body query tokens, estimates camera rotation, and
predicts temporal SOMA parameters. Tracking interpolates internal gaps and holds
endpoint boxes; observation flags identify filled entries. A clip needs at least
one detected person.

Frame IDs and timestamps refer to the normalized 30 Hz clip. Original source
metadata and the requested interval are stored separately. Inference processes
the whole sequence: `network.max_len: 120` controls local attention while
identity/scale averaging and world integration span the sequence. RGB is read
in batches, and models load by stage.

Choose camera rotation handling in the configuration:

| `camera.mode` | Use when | Required input |
| --- | --- | --- |
| `background` (default) | Background features support an approximate SIFT/homography rotation estimate | Video; the primary person's confident keypoint hull is masked |
| `static` | The camera is fixed | No rotations file |
| `provided` | You have camera rotations | `rotations: rotations.npy`, shaped `[T,3,3]` on the normalized timeline |

The background mode reports an error if background evidence is insufficient.
Optional `camera.intrinsics` is a constant 3×3 matrix for the normalized image;
otherwise both feature extraction and Motion use the diagonal-based focal
estimate.

The output saves camera/global parameters, contact logits, tracking flags,
camera diagnostics, and replayable `conditions`. Pass the whole result or its
saved conditions to [cached inference](#cached-temporal-motion). Parameter-only
inference needs no SOMA geometry assets. Add `--postprocess --soma-assets ...`
for [contact/IK](#contactik-postprocessing), or use
[rendering](#render-saved-motion) to reconstruct meshes. The temporary normalized
video is removed after inference; rendering recreates it from the source.

## Cached temporal Motion

Extract the `motion` section of the matching video configuration as a standalone
`motion.yaml`. First fill its detector release IDs. This retains the network
architecture and expected condition specification:

```bash
python - <<'PY'
from omegaconf import OmegaConf

release = OmegaConf.load("docs/configs/body_video_native77.yaml")
motion = OmegaConf.to_container(release.motion, resolve=True)
motion.pop("checkpoint")
motion.pop("stats", None)
OmegaConf.save(OmegaConf.create(motion), "motion.yaml")
PY
```

For a native29 checkpoint, extract from `body_video_native29.yaml` instead.
The supported observation schemas are `soma29_nofinger_v1` and
`soma77_native_v1`; select the one used to train your checkpoint.

The input `.pt` is a sequence dictionary with `condition_spec`, `length`,
`f_imgseq`, `bbx_xys`, `K_fullimg`, `cam_angvel`, and either normalized `obs` or
full-image `kp2d_native77`. The expected conditions in `motion.yaml` must match
the cache. Automatic video inference supplies explicit 30 Hz timestamps. Other
caches must preserve the physical-time settings and source-rate mapping of their
training recipe. Optional `frame_times` are in seconds, `sampling_fps` is in
frames per second, and `time_positions` uses reference-frame units. Tensor
length alone does not define frame rate.

```bash
gem-infer motion-cache --task body --input conditions.pt --config motion.yaml \
  --checkpoint weights/motion_native77_e774_s292175.ckpt \
  --stats weights/motion_stats_raw.npz --device cuda --output motion.pt
```

The selected final775 weights require their matching training normalization NPZ
through `--stats`. Native `gem-train motion` checkpoints embed codec statistics;
omit `--stats` for those checkpoints. Here *native checkpoint* describes the
saved training format and is independent of the native29/native77 joint layout.

## Contact/IK postprocessing

Add `--postprocess` to either Motion command to adjust world translation and
body pose using the six ankle/foot/wrist contact logits:

```bash
gem-infer motion-video --task body --input video.mp4 \
  --config docs/configs/body_video_native77.yaml \
  --postprocess --soma-assets /path/to/soma-assets --device cuda \
  --output processed.pt

gem-infer motion-cache --task body --input conditions.pt --config motion.yaml \
  --checkpoint weights/motion_native77_e774_s292175.ckpt \
  --stats weights/motion_stats_raw.npz \
  --postprocess --soma-assets /path/to/soma-assets \
  --device cuda --output processed.pt
```

For an existing prediction, omit both `--config` and `--checkpoint` to use saved
parameters and contact logits directly:

```bash
gem-infer motion-cache --task body --input motion.pt --postprocess \
  --soma-assets /path/to/soma-assets --device cuda --output processed.pt
```

Providing both options runs the Motion network again from embedded conditions.
A conditions-only input requires the network. As above, omit `--stats` when the
checkpoint embeds codec statistics.

Postprocessing requires **one complete, unpadded sequence with at least two
frames and continuous, uniformly spaced 30 Hz timestamps**. Automatic video
inference supplies this timeline. Other rates, skipped or variable intervals,
and multi-sequence batches are unsupported.

The main `pred_soma_params_incam` and `pred_soma_params_global` fields become the
processed prediction. `raw_predictions` preserves the original parameters and
coordinate description; `postprocessing` records the method and diagnostics.
Repeated processing starts from the raw prediction, so corrections do not
accumulate.

World translation and the shared camera/world `body_pose` change. Root
orientations, identity, scales, camera-space translation, and contact logits
remain unchanged. The translation stage smooths X/Z and aligns the lowest
public-joint Y with the ground even when no contact logit is positive.
The method uses predicted contacts and does not correct camera depth or RGB
alignment.

Contact/IK uses the fixed-GS1 SOMA0.3.3 XLO/Warp profile with procedural
transforms and bind-pose reposing, without pose correctives. Temporal
identity/scale averaging covers the whole sequence. FK/IK disables autocast
and TF32 matrix products, restoring the caller's precision settings afterward.
Rendering dependencies are unnecessary for saving processed parameters.

The Python API exposes the same operation and raw-result selection:

```python
from gem.inference.postprocess import postprocess_motion_result, select_motion_result

processed = postprocess_motion_result(
    result, soma_assets="/path/to/soma-assets", device="cuda"
)
raw = select_motion_result(processed, "raw")
```

## Render saved Pose

Pose results already contain camera-space geometry and topology. Install the
[render dependencies](INSTALL.md#rendering-dependencies) for pinhole mesh
overlays; no checkpoint or SOMA assets are needed:

```bash
gem-render --task hand --input hands.pt --output-dir hand_render
gem-render --task body --input body.pt --output-dir body_render
```

Use the supported x86/Open3D 0.19 rendering environment. For predictions made
on GB300/ARM64, copy the result and source media to that environment; native
Open3D 0.20 rendering is unsupported. A moved source can be specified with
`--source /new/path/to/source`.

Image results produce `overlay.png`. Video results produce `overlay.mp4` at
the original FPS, retaining every source frame; frames without predictions
show the source image. Pose supports the overlay view only.

| Saved Pose camera path | Overlay |
| --- | --- |
| Direct pinhole | Meshes with per-instance intrinsics and shared depth ordering |
| Distorted camera or explicit perspective crops | Skeletons projected through source calibration; no Open3D mesh renderer needed |

The output retains source width unless you set `--width`. Hand results with
saved smoothing display that variant by default. Use `--variant raw` or
`--variant smoothed` to choose explicitly. Rendering uses saved geometry and
never runs smoothing.

## Render saved Motion

Install the [render dependencies](INSTALL.md#rendering-dependencies), then
render camera and world views from saved SOMA parameters:

```bash
gem-render --task body --input motion.pt --soma-assets /path/to/soma-assets \
  --device cuda --output-dir render
```

Use the supported x86/Open3D 0.19 environment described above, with matching
SOMA assets and source media. Rendering reconstructs geometry without loading
Pose/Motion checkpoints or rerunning neural inference.

The active parameter fields are rendered by default. A processed bundle also
supports explicit selection of either saved variant:

```bash
gem-render --task body --input processed.pt --variant raw \
  --soma-assets /path/to/soma-assets --output-dir render_raw
gem-render --task body --input processed.pt --variant postprocessed \
  --soma-assets /path/to/soma-assets --output-dir render_processed
```

`postprocessed` must already exist in the result. Render options are:

| Option | Behavior |
| --- | --- |
| `--views overlay world` | Default: write `overlay.mp4`, `world.mp4`, and `render.json` |
| `--views overlay` or `--views world` | Render one view |
| `--width N` | Override source-size overlay and 1280-pixel-wide world output; use an even width of at least 64 |
| `--backend software` | Default Mesa EGL backend |
| `--backend egl` | Use the caller's EGL configuration |
| `--device` | Select SOMA geometry compute device, independently of the renderer |

Overlays use saved camera-space parameters and `conditions.K_fullimg` with an
opaque green surface. Output dimensions are made even for H.264. The world
view uses a light-blue material, ground grid, and projected Hips trajectory.

A `motion-video` result stores its source path and selected interval so rendering
can recreate the normalized 30 Hz clip. If the source moved, pass
`--source-video /new/path/video.mp4`; the saved start and duration still apply.
For a bare cached result, supply an already aligned video whose frame count,
FPS, and intrinsics match the Motion timeline. Rendering requires one unpadded
sequence with uniformly spaced timestamps.

A world-only render needs neither source video nor camera intrinsics:

```bash
gem-render --task body --input cached_motion.pt --soma-assets /path/to/soma-assets \
  --views world --output-dir render_world
```

Mesh reconstruction uses SOMA0.3.3 MID with pose correctives and leaves saved
parameters unchanged. A single display translation places the first Hips X/Z
at the horizontal origin and the sequence's lowest vertex Y on the floor. A
fixed camera frames the complete motion. These display transforms do not
apply contact correction or IK. Processed world meshes use the same IEEE
float32 matrix policy as contact/IK; camera overlays use ordinary rendering
precision. `render.json` records timeline, geometry profile, backend, output
paths, and world display transform.

To render immediately after inference, add `--render-dir` and `--soma-assets`:

```bash
gem-infer motion-video --task body --input video.mp4 \
  --config docs/configs/body_video_native77.yaml \
  --device cuda --output motion.pt --render-dir render \
  --soma-assets /path/to/soma-assets

gem-infer motion-cache --task body --input conditions.pt --config motion.yaml \
  --checkpoint weights/motion_native77_e774_s292175.ckpt \
  --stats weights/motion_stats_raw.npz \
  --output motion.pt --render-dir render_world \
  --soma-assets /path/to/soma-assets --render-views world
```

Use `--render-views`, `--render-width`, and `--render-backend` for the corresponding
options. `--render-variant raw|postprocessed` selects the saved variant and defaults
to the active result. Add `--postprocess` to process before saving and rendering;
otherwise Motion outputs remain raw.

## Outputs

GEM saves `.pt` tensor dictionaries:

| Format | Contents |
| --- | --- |
| `gem_pose_predictions_v1` | Frame IDs, source image size/intrinsics, caller track IDs, physical-side parameters, camera-space joints/vertices, translation, full-image and crop-normalized keypoints, validity masks, and query tokens |
| `gem_body_video_predictions_v1` | Motion results, replayable conditions, normalized timeline metadata, primary tracking, and camera diagnostics |
| `gem_motion_predictions_v1` | Condition specification, sequence length, camera/global SOMA parameters, and static-contact logits |

Pose stores faces once per body/left/right topology. `source_frame_count`
retains the complete input grid, including unannotated frames. Each pinhole
instance records its effective intrinsics.

Full-image undistortion and perspective-crop results store source calibration
in each frame's `camera`, with `intrinsics: null`. Perspective-crop rows also
retain their virtual `perspective_crop` camera. `camera_adapter` identifies
`full_image_undistortion`, `perspective_crops`, and `output_raster: source`.
Skeleton render metadata identifies this output as `source_camera_keypoints`.
Hand providers add box provenance and optional `box_provider.bbox_smoothing`.

With hand smoothing, `hand_smoothing` describes per-row `smoothed` groups
containing `parameters`, `camera_translation`, `joints_camera`,
`vertices_camera`, `keypoints_2d`, `keypoints_crop`, and `projection_valid`.
`shape_parameters` exposes filtered identity and packed scales. To reconstruct
a smoothed hand, pass `smoothed.parameters` to `HandGeometry.forward_physical`
with the saved side, then apply `smoothed.camera_translation` once.

**Saved Pose geometry is already translated into the original camera, in
meters.** Apply `camera_translation` only when reconstructing from parameters,
not to the saved joints or vertices. Raw hand query tokens remain in the
canonical model frame even when geometry describes the physical right hand.

Motion is raw by default. After `--postprocess`, main parameter fields hold the
processed prediction, `raw_predictions` retains the original parameters and
coordinates, and `postprocessing` records diagnostics.
