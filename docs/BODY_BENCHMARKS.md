# Body Pose benchmarks

[Overview](../README.md) · [Evaluation](EVAL.md) · [Motion benchmarks](MOTION_BENCHMARKS.md) · [Pretrained models](PRETRAINED_MODELS.md)

Evaluate `body_s240000.ckpt` on the official 3DPW test split and EMDB-1 with
the supplied preparation command, fixed configurations and SOMA joint projectors.
The runner reports MPJPE, PA-MPJPE and scale-MPJPE in millimeters. This guide
covers single-frame Body Pose. The existing [release table](EVAL.md#gem-pose-body-pose)
also reports PVE, but this runner does not compute PVE or temporal Motion metrics.

> **Asset publication status:** the two joint projector files listed below are
> prepared for review and have not been uploaded to Hugging Face. The projector
> download command will work after they are published. Each evaluation run needs
> the matching local projector at the documented path.

## Install and download model assets

Install GEM and the evaluation dependencies from this checkout:

```bash
python -m pip install -e '.[eval,benchmark]'
hf download nvidia/GEM-X body_s240000.ckpt --local-dir weights
```

Obtain [SOMA 0.3.3 assets](INSTALL.md#soma-code-and-assets) and place them in
`assets/soma/`. The release uses the built-in `body_config()` explicitly;
`body_s240000.ckpt` is a weights export rather than a complete training-resume
checkpoint. Evaluation uses the supplied benchmark boxes and intrinsics, so no
person detector, ViTPose detector or Motion checkpoint is needed.

The planned projector paths in [nvidia/GEM-X](https://huggingface.co/nvidia/GEM-X/tree/main) are:

| File | Prediction joint order | Scored joints |
| --- | --- | --- |
| `evaluation/body_pose_v1/3dpw_projector.pt` | Complete 44-joint order used by the 3DPW preparation and evaluator | Indices 25–37 and 43; 14 joints |
| `evaluation/body_pose_v1/emdb_projector.pt` | SMPL24 order | All 24 joints |

After publication, download them with:

```bash
hf download nvidia/GEM-X \
  evaluation/body_pose_v1/3dpw_projector.pt \
  evaluation/body_pose_v1/emdb_projector.pt \
  --local-dir weights
```

`--local-dir weights` preserves the `evaluation/body_pose_v1/` subdirectory.
Set `GEM_EVAL_ASSETS` to that subdirectory, not to its parent `weights/`.
Each projector expects the SOMA **MID topology with 18,056 vertices**, in meters.
Any face-point inputs are encoded in the asset and handled by the projector.
Pass the matching full joint order to the evaluator; selecting joints happens
inside the protocol. Native SOMA rig joints are not a substitute for these
benchmark joints.

## Obtain the official datasets and SMPL models

Download the datasets and SMPL models directly from their providers and follow
their access and use terms. GEM does not distribute dataset images, annotations,
SMPL model files or prepared benchmark labels.

| Required files | Official source | Local layout used below |
| --- | --- | --- |
| 3DPW `imageFiles.zip` and `sequenceFiles.zip` | [3DPW download and terms](https://virtualhumans.mpi-inf.mpg.de/3DPW/license.html) | `data/3dpw/imageFiles/` and `data/3dpw/sequenceFiles/` |
| EMDB sequences and annotations | [EMDB access application](https://emdb.ait.ethz.ch/); approval is required | `data/emdb/P0/`, `data/emdb/P1/`, etc. |
| SMPL male, female and neutral models | [SMPL registration and downloads](https://smpl.is.tue.mpg.de/) | `data/smpl/SMPL_MALE.pkl`, `SMPL_FEMALE.pkl`, `SMPL_NEUTRAL.pkl` |
| `SMPL_to_J19.pkl` for 3DPW ground truth | [4D-Humans / HMR2 support assets](https://github.com/shubham-goel/4D-Humans#installation) | `data/hmr2-support/data/SMPL_to_J19.pkl` |

For 3DPW, extract both archives under the same root. The preparation command
uses `sequenceFiles/test/` and resolves images in `imageFiles/<sequence>/`.
For EMDB, retain the official participant and sequence directories. The
[EMDB repository](https://github.com/eth-ait/emdb#dataset) documents access and
the dataset layout.

Use the official SMPL model files, with the three filenames shown above in one
directory. Male/female models reconstruct the dataset's gender-specific body
geometry; the neutral model supplies the common joint regressor. These files
are inputs to local ground-truth preparation, separate from the SOMA assets
used for GEM predictions.
The preparation command loads them through `smplx`. For older model downloads
containing Chumpy objects, follow the upstream
[SMPL model setup instructions](https://github.com/vchoutas/smplx#smpl-and-smplh-setup)
to prepare files supported by that loader.

The [official HMR2 downloader](https://github.com/shubham-goel/4D-Humans/blob/main/hmr2/models/__init__.py)
uses the following support archive, and its
[model configuration](https://github.com/shubham-goel/4D-Humans/blob/main/hmr2/configs_hydra/experiment/default.yaml)
selects `data/SMPL_to_J19.pkl`. Obtain it from that upstream distribution:

```bash
mkdir -p data/hmr2-support
curl -L https://www.cs.utexas.edu/~pavlakos/4dhumans/hmr2_data.tar.gz \
  -o data/hmr2-support/hmr2_data.tar.gz
tar -xzf data/hmr2-support/hmr2_data.tar.gz -C data/hmr2-support
```

If it is already available in your HMR2 installation, pass its existing local
path to `--joint-regressor`. This support file is not bundled with GEM.

## Prepare local benchmark labels

Run from the GEM checkout, with the dataset roots and model paths adjusted to
your downloads:

```bash
python -m gem.pose.evaluation.prepare_body \
  --dataset 3dpw --data-root data/3dpw \
  --smpl-model-dir data/smpl \
  --joint-regressor data/hmr2-support/data/SMPL_to_J19.pkl \
  --output data/prepared/body_3dpw_test.npz

python -m gem.pose.evaluation.prepare_body \
  --dataset emdb --data-root data/emdb \
  --smpl-model-dir data/smpl \
  --output data/prepared/body_emdb1.npz
```

Preparation writes local NPZ files containing benchmark labels and relative
image paths. It does not copy the images into the prepared file. Keep each
prepared file paired with the dataset root from which it was created.
The runtime dataset reads images and constructs unaugmented calibrated crops.
Ground-truth joint coordinates are in meters in the camera frame.

3DPW preparation selects the test split and camera-valid frames. Its boxes use
annotated 2D joints with positive confidence and a 1.2 expansion. EMDB preparation
selects EMDB-1 sequences and their good-frame masks, using the official boxes
with the longer side as the initial square. Both paths expand to the 192:256
aspect convention and create 256-pixel square crops. Images are not undistorted.

The prepared format contains `protocol`, `imgname`, `center`, `box_size`,
`intrinsics`, `keypoints_3d`, `sequence_id`, `person_id` and `frame_id`.
The final square `box_size` includes the benchmark's 192:256 aspect-ratio
expansion; the dataset must not expand it again. The stable sequence/person/frame
identity distinguishes samples during result aggregation. Prepared labels
remain local to the dataset user.

## Run the fixed release configurations

Set paths from the checkout root:

```bash
export GEM_POSE_CHECKPOINT="$PWD/weights/body_s240000.ckpt"
export GEM_SOMA_ASSETS="$PWD/assets/soma"
export GEM_EVAL_ASSETS="$PWD/weights/evaluation/body_pose_v1"
export GEM_3DPW_ROOT="$PWD/data/3dpw"
export GEM_3DPW_PREPARED="$PWD/data/prepared/body_3dpw_test.npz"
export GEM_EMDB_ROOT="$PWD/data/emdb"
export GEM_EMDB_PREPARED="$PWD/data/prepared/body_emdb1.npz"

gem-eval pose --task body --config docs/configs/eval_body_3dpw.yaml \
  --output outputs/evaluation/body_3dpw_test.json
gem-eval pose --task body --config docs/configs/eval_body_emdb.yaml \
  --output outputs/evaluation/body_emdb1.json
```

The [3DPW](configs/eval_body_3dpw.yaml) and [EMDB-1](configs/eval_body_emdb.yaml)
configurations select `body_config()`, CUDA, FP32 (`32-true`), batch size 64 and
8 data-loader workers. They use no test-time augmentation or temporal
postprocessing. The result JSON records the protocol, release identifier,
precision, input rows, retained sample count, per-metric counts and metric means.

## Joint protocols and interpretation

The 3DPW protocol uses the complete 44-joint input order, subtracts joint 39
(pelvis), then scores joints 25–37 and 43. The EMDB protocol uses SMPL24 order,
subtracts the midpoint of hips 1 and 2, and scores all 24 joints. Both prediction
and ground truth use the same centering rule.

| JSON metric | Definition |
| --- | --- |
| `mpjpe` | Mean Euclidean joint error after the protocol's pelvis alignment. |
| `pa_mpjpe` | Joint error after fitting a proper rotation, uniform scale and translation to the scored joints. |
| `scale_mpjpe` | Joint error after fitting nonnegative uniform scale and translation, without rotation. |

All three outputs are in millimeters. Each retained person-image sample has
equal weight; the runner removes duplicate sample IDs and aggregates samples
across batches. Nonfinite selected ground-truth joints exclude that sample from
the metric mean. No 2D visibility weighting is applied to these 3D joint metrics.

For the released benchmark population, expect 35,515 3DPW person-image samples
across 24 test sequences and 24,117 valid EMDB-1 frames across 17 sequences.
Check coverage alongside errors before comparing with the
[published results](EVAL.md#gem-pose-body-pose). Scale-MPJPE is an additional
reported joint metric; it has no entry in that historical table. PVE remains a
historical mesh metric and is outside this runner's supported outputs.
