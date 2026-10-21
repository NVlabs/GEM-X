# Body Pose Training

[Training overview](TRAINING.md) · [Models](MODEL_OVERVIEW.md) · [Evaluation](EVAL.md#body-evaluation) · [Pretrained models](PRETRAINED_MODELS.md)

Body Pose training uses the image model in `gem/pose`, a frozen differentiable
SOMA 0.3.3 geometry adapter, and the body objective in `gem/pose/training`.
Geometry parameters and the external landmark projector are not learned model
weights. The default `body_native_config()` preset predicts 56 raw local scales
with global scale fixed to 1. This guide covers prepared samples, loss inputs
and checkpoint recovery; see [Models](MODEL_OVERVIEW.md) for the architecture.

```bash
python -m pip install -e '.[train]'
gem-train pose --task body --config training.yaml
gem-train pose --task body --config training.yaml --resume full_pose_checkpoint.ckpt
gem-train pose --task body --config training.yaml --validate --resume full_pose_checkpoint.ckpt
```

The Pose configuration supplies a model factory, a data module, trainer options
and checkpoint settings. `gem.pose.training.factory.build_body_training_model`
accepts explicit SOMA assets, optional model or backbone initialization, and an
optional landmark projector. Its model initialization can reuse compatible Body Pose
learned tensors with the native local56 output convention. Initialization starts
a new run. `--resume` loads a complete native checkpoint written by the GEM
trainer and restores learned weights, optimizer, precision and loop state,
consumed-data progress and per-rank randomness. The configured model and
objective must match the configuration saved in that checkpoint.

Body training keeps constant learning rates unless `model.scheduler` supplies
`constant_steps`, `decay_steps` and `eta_min`. This optional schedule holds each
optimizer group's initial rate, then applies a one-way cosine decay to the shared
absolute `eta_min` floor. Full checkpoints restore its progress together with
Adam. The BF16 body recipe advances it once per optimizer update; `max_steps`
includes both the constant and decay phases.

The public [example configuration](configs/pose_body_example.yaml) reads
`GEM_SOMA_ASSETS` and `GEM_POSE_TRAIN_SAMPLES`. See
[the training quick start](TRAINING.md#body-pose) for generating synthetic samples
and running that configuration. The generated file is marked `synthetic: true`
and contains RGB gradients and artificial finite parameter/camera targets. Its
keypoint confidences are zero because the generator performs no SOMA projection.
The configuration exercises the training API with dense landmark supervision
disabled; use real prepared samples for model training. Validation requires a
separate validation dataset and enabled validation batches.

A native Pose training checkpoint also supplies its release configuration to
inference. Pass it to `gem-infer pose-image --task body --checkpoint ...` and omit
`--release` to use the saved architecture/camera convention. This loads learned
model weights for prediction.

## Prepared data boundary

`gem.pose.data.training.PreparedPoseDataset` reads a finite PyTorch file:

```python
{"samples": [sample, ...]}
```

Each sample is already cropped, augmented if desired, and normalized. There is
no image decoding, augmentation, detector execution or target fitting in this
consumer. Rows use the following tensors; batching prepends dimension `B`.

| Field | Per-row shape | Meaning |
| --- | --- | --- |
| `img` | `[3,256,256]` | Normalized floating-point RGB in CHW order. |
| `bbx_xys` | `[3]` | Full-image crop center and square side, in pixels. |
| `K_fullimg` | `[3,3]` | Camera matrix corresponding to the prepared image/labels; not a crop-resized K. |
| `soma_body_pose` | `[76,3]` | Augmented native body joint-local axis angles, in radians. |
| `soma_global_orient` | `[3]` | Augmented camera-frame root axis angle, in radians. |
| `soma_identity_coeffs` | `[128]` | Native identity coefficients. |
| `soma_scale_params` | `[56]` | Positive raw local scales; exclude a packed global-scale entry. |
| `keypoints_2d` | `[77,3]` | Crop-centered XY plus confidence; XY is normally within `[-0.5,0.5]` for visible points. |
| `keypoints_3d` | `[77,4]` | The objective uses the last confidence column; GT XYZ is rebuilt by FK from the body labels. |
| `pred_cam_gt` | `[3]` | Legacy crop camera `[s,tx,ty]` using the supplied bbox and K. |
| `has_soma_params` | scalar | Availability of the parameter labels. |
| `has_cam_gt` | scalar | Availability of the camera target. |

Optional `has_body_pose` and `has_betas` override the corresponding availability
independently; otherwise the objective uses `has_soma_params`. Parameter and
camera targets must remain finite even where a flag is zero. A masked row needs
finite placeholder values, not NaN parameters. Confidence values are preserved,
including values above 1; the data consumer does not clamp confidence.

Optional `aug_rot` is a scalar angle in **degrees**, defaulting to zero. Pose and
2D targets must already follow the corresponding augmentation convention. The
objective applies the reference full-perspective principal-point correction; it
does not perform a second crop rotation. `img_size`, when provided, is `[width,height]`.
An optional `soma_global_scale` must equal 1; fixed GS1 is part of the architecture.

Dense supervision uses optional `landmarks_2d` shaped `[308,3]`, with the same
crop-centered XY/confidence convention. Supply its matching `[308,V]` projector
through the model factory's `landmark_projector`, either as a tensor or a file
containing a tensor or the `M_soma` entry. For the selected SOMA MID topology,
`V=18056`. Mapping source-specific labels to these 308 landmarks belongs to the
external prepared-data adapter; it is not a native77 or 70-point substitution.
Optional fields should be present consistently across rows so standard PyTorch
collation can stack the batch.

## Crop and normalization precision

The reference unaugmented image path solves a three-point OpenCV affine in
float32 pixel coordinates, samples with `INTER_LINEAR` and constant border,
then applies ImageNet normalization. The public
`prepare_pose_batch(..., crop_convention="motion_training")` uses that affine
sampling convention. The default inference crop convention is a separate
compatibility choice and should not replace training crops silently.

The training normalization convention uses NumPy float64 ImageNet constants after
converting pixels to float32 and dividing by 255, then stores the normalized image
as float32. The public inference crop uses float32 constants. To reproduce
training preprocessing, retain this normalization precision and sampling grid when
generating prepared files.

## Finite epochs and recovery

Construct `PoseTrainingDataModule(train_dataset=..., batch_size=..., seed=...)`.
Its sampler uses deterministic `seed + epoch` shuffling and the standard
DistributedSampler rank/tail policy. Each batch carries one `data_cursor` mapping
with collated `epoch`, `batch_index` and `sample_index` values. Prefetch never
advances the committed cursor.

The training-state callback calls `mark_batch_consumed(batch)` before checkpoint
callbacks, including when a consumed batch does not apply an optimizer update.
The data module saves its epoch, consumed-batch cursor and ordering configuration.
At an epoch's last consumed batch, it advances to the next epoch. On resume it
yields only the unconsumed suffix while reporting the full epoch length for
Lightning's restored loop progress. Keep the dataset contents and ordering
configuration consistent; the reader checks dataset size, batch size, seed,
shuffle/tail policy and rank count. It does not infer a replacement dataset.

The Pose CLI disables Lightning's automatic sampler replacement because this
data module already owns distributed partitioning. Full checkpoints separately
restore model/optimizer state and per-rank randomness through the training layer.
A different data plugin owns its own consumption and recovery policy.

GPU floating-point results can vary after recovery even when the saved cursor
and random state are restored. Epoch loss logs after a mid-epoch restart average
the batches processed by the resumed process; they exclude the earlier process's
partial metric accumulator. Compare matching step logs when inspecting recovery.

<a id="public-framework-and-private-preparation"></a>

## Data preparation responsibilities

GEM provides the parameter model, differentiable geometry, objective,
optimization and recovery interfaces, and prepared-sample consumer. Your data
integration handles dataset selection, raw-storage reading, label association,
calibrated image processing, annotation and fitting.

Emit the prepared schema while preserving the source sample identity, camera,
crop, confidence and augmentation semantics. Image-backbone training requires
RGB `img` tensors; feature-only caches cannot replace them.
