# Evaluation

[Overview](../README.md) · [Training](TRAINING.md) · [Pretrained models](PRETRAINED_MODELS.md) · [Demos](DEMO.md)

Evaluate Body Pose and Hand Pose on prepared image samples with `gem-eval`.
Evaluate Body Motion on complete sequences with `gem-train motion --validate`.
The entry points share metric utilities but use different data contracts:

| Model | Evaluation path | Evaluator |
| --- | --- | --- |
| Body Pose | `gem-eval pose --task body` | Caller-provided [Body evaluator](#body-evaluation) |
| Hand Pose | `gem-eval pose --task hand` | Built-in [native SOMA25 evaluator](#native-hand-evaluation) |
| Body Motion | `gem-train motion --validate` | Caller-provided [sequence evaluator](TRAINING.md#motion-validation) |

Pose video inference applies the image model frame by frame; its benchmark
interface consumes individual frames.
Prepare evaluation samples, weights and SOMA assets before running these commands.

## Release benchmark results

These results use the [selected release checkpoints](PRETRAINED_MODELS.md).
GEM-pose (Body Pose) estimates each image independently; GEM-motion (Body
Motion) reconstructs temporal body motion. Contact/IK postprocessing is enabled
for the reported Motion results. It does not apply to single-frame Pose.

### GEM-pose: Body Pose

The single-frame Body Pose checkpoint is `body_s240000.ckpt`. Errors are in
millimeters; lower is better. All three metrics are computed in the same pass
on the same evaluation samples.

| Dataset | MPJPE ↓ | PA-MPJPE ↓ | PVE ↓ |
| --- | ---: | ---: | ---: |
| 3DPW | 62.20 | 39.56 | 72.89 |
| EMDB-1 | 67.83 | 43.14 | 80.27 |

The 3DPW test set contains 35,515 person-image samples across 24 video sequences
(37 person tracks), using its 14-joint benchmark layout. EMDB-1 contains 24,117
valid frames across all 17 sequences, using its 24-joint benchmark layout.

Evaluation treats each image independently, using the benchmark boxes and
camera intrinsics, FP32 inference and no test-time augmentation. MPJPE measures
joint error after pelvis translation alignment. PA-MPJPE additionally fits a
similarity transform. PVE measures corresponding mesh-vertex error after pelvis
translation alignment, without fitting rotation or scale. Each evaluation
sample has equal weight; batch means are not averaged together.

### GEM-motion: EMDB-2

The selected model is native77 final775, paired with Body Pose `s240000` features
and the **original `vitpose.pth`**, with **Contact/IK postprocessing enabled**.
The results cover all 25 EMDB-2 sequences: 43,205 input frames and 43,129 valid
scored frames. Lower is better for every metric.

| Model | W-MPJPE₁₀₀ (mm) ↓ | WA-MPJPE₁₀₀ (mm) ↓ | RTE (%) ↓ | Jitter score ↓ | Foot sliding (mm) ↓ |
| --- | ---: | ---: | ---: | ---: | ---: |
| GEM-motion + Contact PP | 267.20 | 112.76 | 2.45 | 34.21 | 4.22 |

**Evaluation settings.** The benchmark uses dataset-provided camera intrinsics
for feature extraction and Motion conditioning, and dataset-provided camera
rotations. Inference uses FP32, without Motion flip testing; the original
ViTPose observation extractor uses its heatmap flip test. These are calibrated
benchmark results; the automatic RGB-video pipeline can instead estimate camera
inputs. This table reports the original ViTPose detector, not blur e10.

**Metric definitions.** The world-space metrics follow the
[reference evaluation definitions](https://github.com/zju3dv/GVHMR/blob/main/hmr4d/utils/eval/eval_utils.py):

- **W-MPJPE₁₀₀:** joint error over 100-frame segments, using a similarity
  alignment fitted to the first two frames of each segment.
- **WA-MPJPE₁₀₀:** joint error after fitting a similarity alignment to all
  frames of each 100-frame segment.
- **RTE:** root trajectory error after whole-sequence rigid alignment with
  fixed scale, expressed as a percentage of the reference path length.
- **Jitter score:** mean third-difference joint jerk at 30 Hz, divided by 10,
  following the reference implementation's reporting scale.
- **Foot sliding:** predicted foot displacement over consecutive retained
  frames where reference foot displacement is below 1 cm, in millimeters.

Invalid frames are excluded before the reference temporal calculations.
Metrics pool their individual samples across sequences: 43,129 for the two
world joint errors and RTE, 43,054 for jitter, and 54,720 foot-contact samples
for foot sliding. The table uses postprocessed predictions throughout.

### Benchmark results and evaluator interfaces

The tables report the selected checkpoints under the stated benchmark settings.
Dataset adapters, reference-label preparation and benchmark-specific scoring
implementations are not included in the public package. The commands below
run caller-configured evaluators; they are not bundled EMDB or 3DPW runners.
The downloaded Motion inference weights are also not a native training
checkpoint for `--resume`; see [checkpoint loading modes](PRETRAINED_MODELS.md#initialization-inference-and-full-resume).

## Commands

```bash
python -m pip install -e '.[eval]'
gem-eval pose --task body --config body_evaluation.yaml --output body_metrics.json
gem-eval pose --task hand --config hand_evaluation.yaml --output hand_metrics.json
```

Use `--checkpoint` to override the configured checkpoint and `--device` to choose
the execution device. Native checkpoints written by the GEM Pose trainer carry
their release configuration. Weights without that metadata require an explicit
matching `release` in the predictor configuration. The
[Hand SOMA25 example](configs/eval_hand_soma25.yaml) uses the built-in native
Hand evaluator.
Body Pose evaluation requires a caller-provided evaluator, described below.

To run training-objective validation, configure a prepared validation dataset
and enable validation batches:

```bash
python -m pip install -e '.[train]'
gem-train pose --task body --config body_validation.yaml --validate --resume body_training.ckpt
gem-train pose --task hand --config hand_validation.yaml --validate --resume hand_training.ckpt
gem-train motion --config motion_validation.yaml --validate --resume motion_training.ckpt
```

The Pose commands report their configured training objectives. Benchmark
evaluation during Pose training requires a separate `PoseEvaluationCallback`. The
Motion command runs its configured sequence validation and metric callbacks. Here
`--resume` selects a native training checkpoint for validation without continuing
optimization. An inference-only checkpoint instead belongs in the Pose benchmark
predictor's `checkpoint` field or its `--checkpoint` override. See
[Training and validation](TRAINING.md) for initialization, full recovery and
example data contracts.

## Prepared benchmark data

`PreparedPoseEvaluationDataset` reads `torch.save({"samples": rows}, path)`.
Every row contains a stable string `sample_id`, normalized CHW RGB `img`,
square `bbx_xys[3]`, full-image `K_fullimg[3,3]` and the ground-truth (GT)
fields for the selected protocol. IDs must distinguish frames, people and hand sides. Benchmark
split selection, calibrated crop construction and any lens rectification happen
in the dataset producer. The consumer applies no training augmentation.

Every Hand row includes full-image `img_size=[width,height]` for direct 2D
translation solving. Hand crops use the canonical-left image convention. For
right-hand rows, also set `right_hand=True` and retain `original_bbx_xys` and
`original_K_fullimg`, as produced by the Pose input API. The predictor
reconstructs the physical side; GT must describe that same side and camera frame.
A dataset that deliberately evaluates both sides in the canonical-left frame
instead uses left predictions and correspondingly canonical-left GT. Do not mix
these two conventions within a row.

Prepare benchmark GT independently of the training objective's targets derived
from forward kinematics (FK). A training batch does not automatically supply
benchmark GT.

## Body evaluation

Body Pose supports a caller-provided evaluator in the standalone runner and
`PoseEvaluationCallback`. Configure its import path under `evaluator`:

```yaml
evaluator:
  _target_: my_project.evaluation.BodyEvaluator
```

The evaluator exposes a `protocol` name and implements
`evaluate(prediction, batch)`. It returns a mapping from metric name to a tensor
with one scalar per sample, shaped `[B]`. The caller owns GT labels, coordinate
conventions, metric units and any joint or mesh selection. Shared alignment and
joint-error functions are available in `gem.common.evaluation`.

The runner deduplicates samples by `sample_id` before computing means. For metrics
that need aggregation across batches, an evaluator can additionally implement
`protocol_values(prediction, batch)` and `summarize(payload)`; the payload is
merged across the retained samples before summarization. An optional `to(device)`
method moves evaluator-owned tensors to the prediction device.

## Motion validation

Body Motion uses `gem-train motion --validate` with a caller-provided sequence
evaluator. `MotionEvaluationCallback` collects its per-sequence metric arrays,
removes distributed sampler duplicates and logs dataset-wide means. The public
package supplies callback aggregation and shared alignment math; metric definitions,
benchmark selection and GT preparation belong to the caller's integration.
See [Motion validation](TRAINING.md#motion-validation) for configuration and the
evaluator's input/output contract.

For raw versus contact-postprocessed (PP) comparisons, retain each complete
sequence's raw parameters and separately apply `SomaContactPostprocessor` before
scoring the processed parameters. Match the checkpoint, observations, camera
calibration, precision and frame mask across both variants. See
[Contact/IK](DEMO.md#contactik-postprocessing) for the 30 Hz postprocessing
interface.

## Native Hand evaluation

`HandEvaluator` supports the `soma25_online` protocol using native SOMA25 joints,
including the wrist and five fingertips. It reports the following metrics in
millimeters:

| Metric | Definition |
| --- | --- |
| MPJPE | Mean per-joint position error after subtracting the wrist position. |
| PA-MPJPE | Mean per-joint position error after wrist centering and Procrustes alignment. |
| RTE | Error in absolute camera translation, without trajectory alignment. |

Prepared samples provide `gt_joints_camera[25,3]` and
`gt_camera_translation[3]` in meters, with the same physical hand side and camera
axes as the prediction. Optional boolean `valid` excludes an entire sample.
Each retained sample has equal weight and all 25 joints enter the joint mean.
There is no implicit per-joint visibility weighting or skeleton conversion.

## Aggregation and online evaluation

The standalone command runs on one configured device. It reports protocol,
release, sample count, metric values and per-metric sample counts. Missing
protocol measurements are JSON null, so missing measurements remain
distinguishable from zero error.

`PoseEvaluationCallback` runs the same evaluator on an independent prepared
benchmark during training. Configure it under the existing `callbacks` list
with `dataset`, `evaluator`, `name`, `every_n_steps`, and loader options.
The callback uses the current model and geometry; it restores all original
module train/eval flags and model randomness after evaluation. Benchmark GT
never passes through the training loss interface.

An explicit `steps` list can replace `every_n_steps` when a training recipe has
multiple evaluation phases. `evaluate_on_resume: true` evaluates the restored
checkpoint once at training start, after model/geometry initialization. The
optional train-end evaluation runs only at natural step/epoch completion.
Metrics are written to each configured logger at the actual evaluation step;
`output_dir` additionally saves one JSON result per benchmark and step, including
evaluated sample counts and coverage.

Distributed online evaluation shards the dataset and removes sampler padding
by `sample_id` before aggregation. Metrics use the merged samples; epoch or
batch means are not substituted for the full protocol.

## Training previews

`PoseVisualizationCallback` produces crop, keypoint and mesh previews
from caller-provided unaugmented samples. It saves PNG grids and sends those
grids to a configured W&B logger. Install the render extra alongside training:

```bash
python -m pip install -e '.[train,render]'
```

Add a callback to a training configuration, using your own prepared preview file:

```yaml
callbacks:
  previews:
    _target_: gem.pose.evaluation.PoseVisualizationCallback
    name: preview
    dataset:
      _target_: gem.pose.data.evaluation.PreparedPoseEvaluationDataset
      path: /path/to/preview_samples.pt
    num_samples: 4
    seed: 12345
    every_n_steps: 1000
    forward_batch_size: 16
    precision: 32-true
```

Rows supply normalized `img`, full-image `bbx_xys` and `K_fullimg`, following
the prepared evaluation contract above. Native body GT enables a GT mesh;
matching landmark labels and an objective projector enable dense overlays.
An optional evaluator supplies benchmark joint projection. `full_image: true`
additionally reads each row's caller-provided `imgname` for a full-image overlay.

The callback handles rendering, scheduling and seeded sampling using the
provided dataset. The default output is `<training-output>/visualizations/<name>/step-<step>/`;
`output_dir` can select another root. Only rank zero renders, while all ranks
participate in the callback and restore their prior model modes and RNG state.
