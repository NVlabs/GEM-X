# Training and validation

[Overview](../README.md) · [Models](MODEL_OVERVIEW.md) · [Evaluation](EVAL.md) · [Pretrained models](PRETRAINED_MODELS.md)

GEM provides training entry points for **Body Pose**, **Hand Pose**, and **Body
Motion**. Pose estimates a camera-frame body or hand from one image crop.
Body Motion uses a sequence of cached Body Pose features, observations and camera
information to estimate camera-frame and world-frame body motion.

| Model | Training command | Training inputs | Validation and benchmark evaluation |
| --- | --- | --- | --- |
| Body Pose | `gem-train pose --task body --config training.yaml` | Prepared RGB crops and body labels | Objective validation; [caller-provided evaluator](EVAL.md#body-evaluation) |
| Hand Pose | `gem-train pose --task hand --config training.yaml` | Prepared canonical-left RGB crops and hand labels | Objective validation; [native SOMA25 evaluation](EVAL.md#native-hand-evaluation) |
| Body Motion | `gem-train motion --config training.yaml` | Mixed temporal video batches and single-image/T1 batches with cached Pose features | [Full-sequence validation](#motion-validation) with Motion evaluation callbacks |

Complete validation and benchmark commands appear below and in
[Evaluation](EVAL.md). See [pretrained models](PRETRAINED_MODELS.md) for the
matching model configurations and weights, and [demos](DEMO.md) for inference.

Install the training dependencies:

```bash
python -m pip install -e '.[train]'
```

Training configuration supplies `model`, `data`, `trainer`, `checkpoint`, optional
`callbacks`, and an optional `seed`. The Pose quick starts generate synthetic
samples to exercise the training API. The Motion example consumes batches you
prepare. For training on real data, implement the corresponding sample contract
and choose settings for your dataset. See [Recovery](#recovery) for initialization
and resume behavior.

## Body Pose

Start with the public [body example](configs/pose_body_example.yaml):

```bash
export GEM_SOMA_ASSETS=/path/to/soma-assets
export GEM_POSE_TRAIN_SAMPLES=/tmp/pose_body_example.pt
python scripts/make_pose_example.py --output "$GEM_POSE_TRAIN_SAMPLES" --count 8
gem-train pose --task body --config docs/configs/pose_body_example.yaml
```

The generator produces RGB gradients and artificial finite parameter/camera
targets. Keypoint confidence is zero because the generator performs no SOMA
projection. The configuration disables dense landmark supervision and runs two
single-GPU BF16 training steps. Replace `GEM_POSE_TRAIN_SAMPLES` with a real
prepared dataset to train on labeled images.

The default `body_native_config()` predicts 56 local scales with global scale
fixed to 1. The model factory accepts `initialization` for compatible model
weights or `backbone_initialization` for the image backbone alone. To resume or
validate a native training checkpoint:

```bash
gem-train pose --task body --config training.yaml --resume full_pose_checkpoint.ckpt
gem-train pose --task body --config validation.yaml --validate --resume full_pose_checkpoint.ckpt
```

Validation needs a prepared validation dataset and enabled validation batches;
the quick-start configuration trains only. This command evaluates the configured
Body training objective.

A caller-provided Body evaluator runs through `gem-eval pose` or a configured
`PoseEvaluationCallback` over an independent benchmark dataset. See
[Evaluation](EVAL.md#body-evaluation). Training-loss validation and benchmark
metrics retain separate inputs and names.

Native Pose training checkpoints carry their release configuration and learned
weights. Inference can recover that configuration when `--release` is omitted:

```bash
gem-infer pose-image --task body --input image.jpg --boxes boxes.json \
  --checkpoint full_pose_checkpoint.ckpt --soma-assets "$GEM_SOMA_ASSETS" \
  --device cuda --output body.pt
```

Inference still requires the external SOMA assets. See
[Body Pose training](POSE_TRAINING.md) for the prepared schema, dense308
projector, normalization precision and data recovery contract.

## Hand Pose

Hand uses the same command with its own model factory and prepared-data schema:

```bash
export GEM_SOMA_ASSETS=/path/to/soma-assets
export GEM_HAND_TRAIN_SAMPLES=/tmp/hand_pose_example.pt
python scripts/make_hand_pose_example.py --output "$GEM_HAND_TRAIN_SAMPLES" --count 8
gem-train pose --task hand --config docs/configs/pose_hand_example.yaml
```

The example uses `hand_training_config()` and the full Hand architecture. It
runs two FP16 mixed-precision steps with AdamW and gradient clipping after
unscale. The parameter head, separate 2D head and learned residual log-likelihood
estimation (RLE) density train alongside the image backbone. Global hand scale
is learned; the 26 local MHR controls are signed. Every prepared training sample
must use the canonical-left camera/pose convention and include the full-image
`img_size=[width,height]` required by the direct 2D branch.

The objective rebuilds target joint coordinates through forward kinematics (FK).
GT `pred_cam_gt` remains `[s,tx,ty]` and is converted to log-lambda inside the
loss. See [Hand Pose training](HAND_TRAINING.md) for the complete schema, direct
2D/translation losses and preparation convention.

To fine-tune the selected Hand `s060000` weights on real prepared samples, use
[hand_finetune.yaml](configs/hand_finetune.yaml). Set `GEM_HAND_CHECKPOINT` to the
complete local checkpoint (for example `weights/egohand_s060000.ckpt`),
`GEM_HAND_TRAIN_SAMPLES` to the prepared dataset, and
`GEM_SOMA_ASSETS` to the hand geometry assets:

```bash
gem-train pose --task hand --config docs/configs/hand_finetune.yaml
```

This initializes the backbone, parameter head, separate 2D head and learned RLE
density for a new run. See
[EgoHand fine-tuning](HAND_TRAINING.md#egohand-fine-tuning) for the environment
setup, editable training settings, checkpoint locations and resume commands.

Resume or validate a native Hand training checkpoint:

```bash
gem-train pose --task hand --config training.yaml --resume full_hand_checkpoint.ckpt
gem-train pose --task hand --config validation.yaml --validate --resume full_hand_checkpoint.ckpt
```

The validation configuration must supply a prepared hand validation dataset;
native SOMA25 metrics are documented under
[Hand evaluation](EVAL.md#native-hand-evaluation).
A nonfinite forward consumes the batch and completes
backward before dropping the update; an AMP gradient overflow follows
GradScaler's skip/backoff. Applied Adam updates can therefore differ from
Lightning's optimizer-attempt count.

For multi-GPU training, configure `strategy: ddp_find_unused_parameters_true`
because the image model's masked-token parameter is unused for these RGB inputs.

The training entry point also accepts ranks, node count and devices supplied
by a torchrun launcher. Configure the launcher and storage for your environment.

<a id="motion"></a>

## Body Motion

Start with the public [mixed Motion example](configs/motion_mixed_example.yaml):

```bash
export GEM_SOMA_ASSETS=/path/to/soma-assets
export GEM_MOTION_STATS=/path/to/motion_stats_raw.npz
export GEM_LANDMARK_PROJECTOR=/path/to/landmark_projector.pt
export GEM_POSE_FEATURE_RELEASE=my-body-pose-release:pose_tokens
export GEM_MOTION_BATCH_PACK=/path/to/mixed_batches.pt
export GEM_MOTION_RUN_DIR=outputs/motion_mixed_example
gem-train motion --config docs/configs/motion_mixed_example.yaml
```

The example starts a fresh 20-attempt run with the full native29 Body Motion
network, 655 output coordinates and cached 1024-dimensional Body Pose tokens.
Four video batches and one image/T1 batch form each shuffled block. The 80:20
mixture illustrates the data interface; choose sources and proportions for your
training dataset.

This **native29 example is an alternate configuration** to the pretrained native77
Body Motion model. Native29 requires its own matching weights, observations and
normalization statistics. See [Body Motion weights](PRETRAINED_MODELS.md#body-motion-choose-a-matched-set)
for the release configuration.

Provide the matching 655D fixed-GS1 normalizer, a `.pt` containing
`M_soma[308,18056]`, and [SOMA assets](INSTALL.md). All cached features must match
`GEM_POSE_FEATURE_RELEASE`, including the feature format. Checkpoints record
this identifier and normalization buffers. The factory still needs explicit
assets when constructing a model for resume or validation.

This configuration consumes a finite prepared batch pack; it does not generate
training data. The existing public `PreparedBatchPack` reader accepts:

```python
torch.save({
    "schema": "gem_motion_real_batch_pack_v1",
    "metadata": {"feature_version": "my-body-pose-release:pose_tokens"},
    "batches": {
        "video": [video_batch_0, video_batch_1],
        "images": [image_batch_0, image_batch_1],
    },
}, "mixed_batches.pt")
```

Each item already has a batch axis and precedes model-side FK, observation
augmentation and modality dropout. The pack reader preserves metadata; the
producer owns feature/label compatibility. Use the public
`gem.motion.data.collate.collate_video_rows` and `collate_image_rows` to build
these batches from individual rows. The image collator adds T=1 and retains
image labels without world/contact targets.

| Input | Video batch | Image batch |
| --- | --- | --- |
| `length` | `int64[B]`, valid frames per padded clip | `int64[B]`, all 1 |
| `f_imgseq` | `[B,T,1024]` | `[B,1,1024]` |
| `bbx_xys`, `K_fullimg` | `[B,T,3]`, `[B,T,3,3]` | `[B,1,3]`, `[B,1,3,3]` |
| Observations | `kp2d[B,T,29,3]`, full-image XY/confidence | Constructed from reliable native GT |
| Camera/time | `cam_angvel[B,T,6]`, `R_c2gv[B,T,3,3]`; `frame_times[B,T]` in seconds | Image camera/labels, no temporal target |
| Supervision | `soma_params_c`, `soma_params_w`, `mask`, `meta` | `image_labels` without T axis |

Video parameters use axis-angle `body_pose[B,T,76,3]`,
`global_orient[B,T,3]`, `transl[B,T,3]`, identity128 and scales57
(`[GS1,local56]`, positive local scales). Identity/scales can also be static
`[B,D]`. Masks include `valid[B,T]` and per-row `bbx_xys`, `f_imgseq`, `vitpose`,
`spv_incam_only`; sparse rows also provide `label_valid`, `kp2d` and `imgfeat`
frame masks. Preserve separate `motion_targets` and root masks when supplied.
Use the source sampling clock in `frame_times`; the example defaults to 30 Hz
when no explicit clock is supplied.

Image labels include `bbx_xys`, `K_fullimg`, axis-angle `soma_body_pose` and
`soma_global_orient`, `soma_identity_coeffs[B,128]`,
`soma_scale_params[B,57]`, `soma_transl_m[B,3]`, `has_soma_params` and
`has_cam_gt`. Supply `native_joint_confidence[B,77]` or the confidence column of
`keypoints_3d[B,77,4]`. The dense308 supervision field is
`keypoints_2d_sapiens[B,308,3]`, with centered-crop XY and confidence; unavailable
landmarks use zero confidence. This retained field name describes supervision,
not the inference detector. The current factory requires the matched
projector and label layout even when the corresponding loss weight is zero.
Motion image targets use packed fixed-GS1 scales57, whereas the Pose-body example
uses raw local56.

For larger datasets, existing public components include `ImageFeatureCacheDataset`
for partitioned image-feature columns, `NativeVideoObservations` for joining
native77 detector rows and selecting native29, and `BudgetedVideoDataset` for
per-source video budgets. A caller dataset supplies the same row contract;
raw-source preparation stays outside the trainer. The Pose feature producer is
fixed input to this Motion factory.

```bash
gem-train motion --config training.yaml --resume native_motion_checkpoint.ckpt
gem-train motion --config validation.yaml --validate --resume native_motion_checkpoint.ckpt
```

Set the desired total `trainer.max_steps`/`max_epochs` when extending a completed
run. Scheduled checkpoints use names such as `attempt-000000005.ckpt`; consumed
attempts include AMP-skipped updates. For multi-GPU mixed training configure
`strategy: ddp_find_unused_parameters_true`. Every source needs enough records
for a global batch. With the prepared pack, sampler batch size stays 1 and each
rank reads a distinct complete batch record.

The compatibility shorthand `gem-train --config training.yaml` and
`python scripts/train_motion.py` invoke the same Motion trainer.

## Configuration responsibilities

- `model`: network, schema, losses and explicit SOMA/normalization/projector assets.
- `data`: datasets, batch sizes and optional validation datasets. Body and Hand
  Pose consume their respective prepared image samples; Motion additionally
  configures temporal source ordering, epoch budgets and its image/video schedule.
- `trainer`: device/precision, cadence and optimization-loop settings.
- `checkpoint`: complete-state checkpoint location and cadence.
- `callbacks`: configured evaluation or application callbacks.

`trainer.logger` accepts Lightning's usual boolean setting or a Hydra logger
configuration with `_target_`, such as
`lightning.pytorch.loggers.CSVLogger`. A list of configured loggers is also
supported. Install the selected logger's optional dependencies when needed.

Prepared-data consumers live in `gem.pose.data.training`,
`gem.pose.data.hand_training` and `gem.motion.data`. Your data integration handles
source reading, image/label processing, annotation, fitting and benchmark
ground-truth preparation, then emits the documented sample or batch contract.
Install that integration separately when your configuration requires it.

## Recovery

**Initialization** loads learned weights to start a new run. Body supports full
model or backbone initialization; Hand initialization from the selected
`egohand_s060000.ckpt` also loads the learned RLE density. Existing Motion
inference weights require a caller-supplied initialization path.

**Resume** requires a native training checkpoint: a complete checkpoint written
by the GEM trainer, including learned weights, model/objective configuration,
optimizer, scheduler when configured, precision state, loop progress, consumed
data and each rank's random state. Keep the configured model and objective
consistent with the saved checkpoint. Hand checkpoints also retain the RLE density
and GradScaler. Historical or inference-only weights do not contain this full
recovery state. Inference from a native Pose checkpoint uses its saved release
configuration and weights without restoring the training loop.

Pose checkpoints commit the data cursor before checkpoint callbacks and restore
per-rank randomness at the first resumed batch. Motion records consumed attempts
rather than worker prefetch position. Keep the dataset, ordering settings and
distributed topology consistent when resuming either trainer. See
[Pose recovery](POSE_TRAINING.md#finite-epochs-and-recovery) for the finite-epoch
loader contract.

Validation-only execution does not restore training RNG topology, so it can
evaluate a checkpoint on a different rank count. It still checks model observation
schema and feature compatibility.

## Motion validation

Use `gem-train motion --config validation.yaml --validate --resume checkpoint.ckpt`.
The `gem-eval` command selects Pose protocols only; Motion validates complete
sequences with its own model outputs and a caller-provided evaluator.

Native validation runs raw inference without training augmentation or contact
postprocessing. B1 full-sequence loaders retain the caller's GT mask and optional
clock information. `MotionEvaluationCallback` deduplicates distributed sampler
padding and aggregates per-frame metric arrays. It preserves the training RNG
state around lazy evaluation setup.

To configure validation, copy the mixed example to `validation.yaml`, retain its
model/training-data entries, and merge these entries into the existing sections.
Replace `my_project.evaluation.SequenceEvaluator` with your own implementation:

```yaml
data:
  validation_sources:
    prepared_sequences:
      _target_: torch.load
      f: ${oc.env:GEM_MOTION_VALIDATION_ROWS}
      map_location: cpu
      weights_only: true
trainer:
  limit_val_batches: 1.0
  inference_mode: false
callbacks:
  - _target_: gem.motion.evaluation.MotionEvaluationCallback
    evaluator:
      _target_: my_project.evaluation.SequenceEvaluator
```

Save `GEM_MOTION_VALIDATION_ROWS` as `torch.save(sequence_rows, path)`, a list of
**unbatched complete sequence dictionaries**. For the native29 example, each row
contains scalar `length`, `kp2d[T,29,3]`, `f_imgseq[T,1024]`, `bbx_xys[T,3]`,
`K_fullimg[T,3,3]`, `cam_angvel[T,6]`, boolean GT `mask[T]`, and
`meta={"dataset_id": "my_validation", "vid": "unique-sequence-id"}`. Supply
physical `frame_times[T]` when needed. The loader adds B=1. Observation order,
feature identity and dimensions must match the model's condition configuration.
Provide any additional GT fields required by your evaluator in each row.

The evaluator implements `evaluate(outputs, batch)` and returns a mapping with
`dataset_id`, `vid` and `metrics`. `metrics` maps each metric name to a finite,
nonempty array of individual measurements, so the callback can concatenate
measurements across sequences before taking their mean. The existing `fs` metric
may contain no contact samples; it is omitted if no sequence contributes any.
Choose stable sequence IDs to let the callback remove distributed sampler copies.
The evaluator owns joint/mesh conversion, coordinate conventions, GT selection
and metric units. Shared alignment and joint-error functions are available in
`gem.common.evaluation`.

Metrics are logged under `val_metric_<dataset_id>/<metric>`. The training example
disables validation until data and callback settings are supplied. Dataset readers,
benchmark protocols and their concrete configurations remain in the caller's
integration.

## Testing

```bash
python -m pip install -e '.[train,eval,dev]'
pytest tests
```

Default tests use small local fixtures. Tests that need real checkpoints or
additional runtime assets document their opt-in environment variables in the
test modules.
