# Hand Pose Training

[Training overview](TRAINING.md) · [Models](MODEL_OVERVIEW.md) · [Hand evaluation](EVAL.md#native-hand-evaluation) · [Pretrained models](PRETRAINED_MODELS.md#hand-pose-s060000)

Hand Pose uses a single-image model for image and framewise video inference.
`hand_config()` describes inference; `hand_training_config()` uses the same
architecture with training stochastic depth. It combines a full 256 × 256 crop, calibrated perspective camera, 8-band RayPE
in the parameter head, log-lambda depth and a separate 2D keypoint decoder.
The geometry uses a canonical-left hand with 5 identity coefficients and a
27-entry scale output containing global scale followed by 26 local MHR controls.

Training uses differentiable SOMA 0.3.3 hand geometry and a learned residual
log-likelihood estimation (RLE) density model alongside the parameter and 2D
heads. SOMA assets remain frozen and external to the checkpoint; the learned density is saved and restored.

```bash
python -m pip install -e '.[train]'
python scripts/make_hand_pose_example.py \
  --output outputs/hand_example_samples.pt --count 8
export GEM_SOMA_ASSETS=/path/to/soma-assets
export GEM_HAND_TRAIN_SAMPLES="$PWD/outputs/hand_example_samples.pt"
gem-train pose --task hand --config docs/configs/pose_hand_example.yaml
```

The generator creates **synthetic RGB gradients and artificial valid targets** to
exercise the training API. Use real prepared RGB and ground-truth samples for
training on your dataset. The [example
configuration](configs/pose_hand_example.yaml) runs two FP16 mixed-precision steps
with separate backbone/head learning rates and gradient clipping. Its default
model has the full production architecture; the example therefore still requires
suitable GPU memory and hand assets.

`gem.pose.training.factory.build_hand_training_model` accepts caller-owned
assets and optional initialization from a complete matching Hand checkpoint,
including its learned RLE density. Initialization starts a new run without
restoring optimizer, scaler, step, data cursor or random state. Full recovery
uses a native checkpoint written by the GEM trainer, containing those states
alongside the model, density and release/objective configuration:

```bash
gem-train pose --task hand --config training.yaml --resume full_hand_checkpoint.ckpt
gem-train pose --task hand --config training.yaml --validate --resume full_hand_checkpoint.ckpt
```

Validation requires an explicit validation dataset and enabled validation
batches. The short example disables validation. Native Pose training
checkpoints contain the release configuration, so inference can use
`gem-infer pose-image --task hand --checkpoint ...` without a separate
`--release` file.

## EgoHand fine-tuning

Use [hand_finetune.yaml](configs/hand_finetune.yaml) to start from the complete
`egohand_s060000.ckpt` checkpoint. It initializes the image backbone, parameter
head, separate 2D head and learned RLE density, then trains them with
`hand_training_config()`. A model-only export that
omits density weights is not a complete initialization for this objective. Supply
your own prepared RGB/GT samples in the [format below](#prepared-sample-format),
including canonical-left camera/pose labels and full-image size.

Download the matching GA `s060000` checkpoint from the official
[nvidia/EgoHand-v1 repository](https://huggingface.co/nvidia/EgoHand-v1)
with an account authorized to access it. Hand weights are not distributed in the
GEM-X bundle. See [checkpoint downloads](PRETRAINED_MODELS.md#download-checkpoints)
and [Hand pretrained weights](PRETRAINED_MODELS.md#hand-pose-s060000).
Use this upstream checkpoint through `GEM_HAND_CHECKPOINT` below to initialize
a new run. Use `--resume` only for a checkpoint saved by the GEM trainer.

```bash
python -m pip install -e '.[train]'
export GEM_SOMA_ASSETS=/path/to/soma-assets
export GEM_HAND_TRAIN_SAMPLES=/path/to/prepared_hand_train.pt
export GEM_HAND_CHECKPOINT=weights/egohand_s060000.ckpt
gem-train pose --task hand --config docs/configs/hand_finetune.yaml
```

The default output directory is `outputs/hand_finetune`. Set
`GEM_HAND_FINETUNE_RUN_DIR` to change it. The configuration uses FP16 mixed
precision, batch size 1, zero data-loader workers, head learning rate `1e-5` and
backbone learning rate `5e-6`. The head group includes the separate 2D decoder
and RLE density. It runs up to 10,000 optimizer attempts without an
epoch limit. Every 500 attempts it saves the most recent step checkpoint and
`checkpoints/last.ckpt`; AMP-skipped updates still count as attempts. Validation
is disabled until you configure validation data and enable validation batches.

These defaults are a starting point, not an optimized fine-tuning recipe. Copy
the YAML to adjust learning rates, batch size, run length or validation for your
data. Data preparation is described below.

Resume from a checkpoint saved by this run:

```bash
gem-train pose --task hand --config docs/configs/hand_finetune.yaml \
  --resume "${GEM_HAND_FINETUNE_RUN_DIR:-outputs/hand_finetune}/checkpoints/last.ckpt"
```

On resume, the CLI clears the initialization setting before constructing the
model. The original EgoHand file and `GEM_HAND_CHECKPOINT` environment variable
are no longer needed. Keep the matching SOMA assets, prepared dataset, release,
objective and data-ordering settings; full recovery also requires the saved distributed
topology. `max_steps` is the total target, not additional steps after resuming.

The saved checkpoint also works directly with the Pose inference entry point:

```bash
gem-infer pose-image --task hand --input image.jpg --boxes hand_boxes.json \
  --checkpoint "${GEM_HAND_FINETUNE_RUN_DIR:-outputs/hand_finetune}/checkpoints/last.ckpt" \
  --soma-assets "$GEM_SOMA_ASSETS" --device cuda --output hands.pt
```

Omit `--release` to use the checkpoint's saved hand architecture and camera
convention. See [Hand inputs](DEMO.md#hand-image-or-framewise-video) for the box,
side and intrinsics format.

## Prepared sample format

`gem.pose.data.hand_training.PreparedHandPoseDataset` reads a finite PyTorch
file containing `{"samples": [sample, ...]}` or an in-memory sequence of rows.
Each row already contains the normalized crop and canonical-left labels.
Batching prepends a dimension `B`. **Every prepared training row requires
`img_size`**, for both left and right hands, so the direct 2D branch can construct
image-boundary masks.

| Field | Per-row shape | Meaning |
| --- | --- | --- |
| `img` | `[3,256,256]` | Floating-point ImageNet-normalized RGB, CHW order. |
| `bbx_xys` | `[3]` | Crop center X/Y and square side in the prepared full-image pixel frame. |
| `K_fullimg` | `[3,3]` | Full camera matrix in that same prepared frame. |
| `img_size` | `[2]` | Finite positive full-image `[width,height]`, not the 256 crop dimensions. |
| `hand_poses` | `[25,3,3]` | Canonical-left SOMA rotation matrices: wrist, then 24 finger joints. |
| `hand_identity_coeffs` | `[5]` | MHR hand identity coefficients. |
| `hand_global_scale` | `[1]` | Positive global hand scale. |
| `hand_scale_params` | `[26]` | Signed MHR controls, with zero as neutral; exclude global scale. |
| `pred_cam_gt` | `[3]` | Calibrated crop camera target `[s,tx,ty]`; the objective converts its first channel to log-lambda. |
| `keypoints_2d` | `[25,3]` | Only the final confidence column is consumed; XY may be placeholders. |
| `keypoints_3d` | `[25,4]` | Only the final confidence column is consumed; XYZ may be placeholders. |
| `has_hand_params` | scalar | Availability of hand parameter labels and their FK supervision. |
| `has_cam_gt` | scalar | Availability of the direct camera target. |

The 25-joint order is wrist, four thumb joints, then five joints each for index,
middle, ring and pinky. Fingertips are already indices 4, 9, 14, 19 and 24. Do not append
another tip set. Local MHR controls may be negative; they do not use the body
model's positive local56 convention. The head packs global and local scale
values into 27 entries, while prepared labels retain the two original fields.

Parameter and camera tensors must be finite even on masked rows. Use finite
placeholder values for unavailable targets. Confidence must be finite and is
preserved, including values greater than 1. The consumer neither clips confidence
nor interprets optional visibility metadata as a replacement loss mask.

The objective rebuilds **both 2D and 3D target coordinates** from the hand GT
parameters. It converts `pred_cam_gt` through the supplied bbox and full K,
projects the resulting camera joints, and measures 2D coordinates relative to
the crop center and side. The source convention is crop-centered XY, normally
within `[-0.5,0.5]` for in-crop points. Placeholder keypoint coordinates in a
prepared file are not independent annotated 2D/3D supervision.

Keep prepared camera values in the `[s,tx,ty]` encoding. The loss forms
`log(lambda) = log(2/s)`, with `lambda = box_size * Z / focal_x`.

## Loss and translation contract

The parameter head predicts raw log-lambda plus lateral offsets. The preset
uses a straight-through decode bound of `[-3.5,3.0]`; camera supervision acts on
the raw scalar with Smooth L1 after division by `0.3735116511306386`, together
with MSE on the two offset channels. Ground-truth values are not clipped to the
decode bounds. The release owns camera decoding, RayPE and solver configuration;
conflicting structural overrides in the objective are rejected.

The separate 25-point decoder predicts direct crop coordinates, uncertainty,
presence and evidence maps. RLE supervises every labeled coordinate, including
out-of-crop joints. Presence distinguishes points inside both the crop and full
image; evidence-map CRPS uses only those observed joints. The RLE normalizing
flow is trainable, participates in the head optimizer group and gradient clipping,
and is included in complete training checkpoints. Inference loads the prediction
model without running this training density.

The lateral solver fixes root depth from the decoded camera and solves X/Y
translation through the full camera matrix. Training uses GT label availability,
crop/image membership and positive GT depth to select correspondences; predicted
presence and uncertainty do not gate this training solve. Translation loss uses
valid solves. Projected-joint loss additionally masks invalid predicted depths.
At inference, correspondence selection instead uses predicted presence and
coordinates, with a regressed-camera fallback when the solve is invalid.

The default weights are 0.01 for projected 2D joints, 0.05 for root-relative 3D
joints, 0.001 each for finger rotations, wrist rotation and camera, 0.0005 each
for identity and scale, 1 each for RLE and presence, 0.1 for evidence and 0.05
for translation. The parameter/joint terms use the configured sum reduction;
RLE divides by batch size, presence/evidence use their valid-joint means, and
translation averages squared metric error over valid samples. These distinct
reductions are retained from the checkpoint's recipe.

## Canonical-left camera and crop

Left-hand rows use left geometry in the prepared camera frame. Right-hand rows
must already be mirrored to canonical-left using the source hand rig's FK
mirror and parameter decomposition. Flipping an axis-angle sign or only flipping
the RGB crop is insufficient to establish this contract.

For the calibrated full-image convention, mirror right-hand pixels by
`u' = (W - 1) - u`. With pixel reflection `F` and camera reflection
`M = diag(-1,1,1)`, the matching intrinsic matrix is `K' = F K M`; the bbox
center follows the same pixel reflection. This also preserves cameras with
skew. The source right-hand identity, global scale and local MHR controls are
used with the canonical-left rig.

Camera roll augmentation is applied upstream to the image, crop, wrist rotation
and camera translation together. The prepared row's `aug_rot` must be absent or
zero because that roll is already part of its camera frame. An optional
`full_image_roll_deg` may record the sampled angle as metadata. Optional original
side fields such as `is_flipped` are preserved but never trigger another mirror.

The unrotated training crop uses a three-point OpenCV affine, `INTER_LINEAR`
sampling and constant border. To reproduce training preprocessing, convert pixels
to float32, divide by 255, apply the reference NumPy float64 ImageNet mean/std,
then store float32 CHW. Inference preprocessing uses float32 constants; retain the
training precision when preparing training samples.

## Loader recovery and external preparation

Hand training uses `gem.pose.data.training.PoseTrainingDataModule`. Deterministic
seed-plus-epoch shuffling and the committed `data_cursor` follow the
[Pose recovery contract](POSE_TRAINING.md#finite-epochs-and-recovery).
Checkpoint callbacks record consumed batches rather than worker prefetch.
Resume requires the same dataset ordering, batch settings and rank count.
Optional fields must be consistent across rows for standard collation.

GEM provides the model, geometry, objective, optimization, recovery and prepared
sample reader. Your data integration handles dataset discovery, raw media
reading, calibration, canonicalization, augmentation, annotation and fitting.
Preserve source sample identity and camera/label provenance when emitting the
prepared schema. Detection boxes and inferred hand poses alone do not provide
ground-truth training labels.
