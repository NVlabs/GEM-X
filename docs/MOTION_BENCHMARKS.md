# Body Motion benchmarks

[Overview](../README.md) · [Evaluation](EVAL.md) · [Body Pose benchmarks](BODY_BENCHMARKS.md) · [Pretrained models](PRETRAINED_MODELS.md)

Use `gem-eval motion` to evaluate the released native77 final775 Motion
checkpoint on full benchmark sequences. The workflow prepares ground truth from
official dataset downloads, extracts Body Pose features and original ViTPose
observations, then reports raw and/or Contact/IK-postprocessed predictions.

| Dataset | Evaluation coordinates | Joint protocol |
| --- | --- | --- |
| EMDB-1 | Camera | SMPL24, centered on the midpoint of hips 1 and 2 |
| EMDB-2 | Camera and world | SMPL24 with additional world and trajectory alignment metrics |
| 3DPW test | Camera | Motion J14, centered on the midpoint of joints 2 and 3 |

**Motion J14 is a separate protocol from the single-frame Body Pose 3DPW
protocol.** Use the Motion `3dpw_j14.pt` projector and its corresponding
ground-truth regressor. Do not substitute the Body Pose 44-joint projector or
`SMPL_to_J19.pkl`.

> **Asset publication status:** the Motion projectors listed below are prepared
> for review and have not been uploaded to Hugging Face. Their download commands
> become usable after publication; evaluation requires matching local assets.

## Install and obtain weights

From this checkout:

```bash
python -m pip install -e '.[eval,benchmark]'
hf download nvidia/GEM-X \
  body_s240000.ckpt vitpose.pth \
  motion_native77_e774_s292175.ckpt motion_stats_raw.npz \
  --local-dir weights
```

Obtain [SOMA 0.3.3 assets](INSTALL.md#soma-code-and-assets) and place them in
`assets/soma/`. The fixed [evaluation configuration](configs/eval_motion_native77.yaml)
matches [the native77 video release](configs/body_video_native77.yaml): final775
Motion weights, Body Pose s240000 query features and the original ViTPose
`vitpose77-dinov3-huge-metrosim-256x192-v1` observations. Keep the statistics
sidecar for this inference checkpoint.

The planned [nvidia/GEM-X](https://huggingface.co/nvidia/GEM-X/tree/main) assets are:

| Path under `evaluation/motion_v1/` | Use |
| --- | --- |
| `smpl24.pt` | EMDB prediction joints in SMPL24 order |
| `3dpw_j14.pt` | 3DPW Motion prediction joints in the 14-joint order |
| `mesh_projector/face_vertex_indices.pt` | Mesh projector input topology |
| `mesh_projector/tet_vertex_indices.pt` | Mesh projector point indexing |
| `mesh_projector/smpl_vertex_face_ids.pt` | Mesh projector output indexing |
| `mesh_projector/smpl_vertex_barycentric.pt` | Mesh projector interpolation weights |

The projectors consume SOMA MID meshes with 18,056 vertices in meters. Any
face-point inputs are encoded in the assets and handled internally. Keep all
four mesh projector files in one directory. The mesh output uses 6,890-vertex
SMPL ordering to match the locally prepared ground truth. After publication:

```bash
hf download nvidia/GEM-X \
  evaluation/motion_v1/smpl24.pt \
  evaluation/motion_v1/3dpw_j14.pt \
  evaluation/motion_v1/mesh_projector/face_vertex_indices.pt \
  evaluation/motion_v1/mesh_projector/tet_vertex_indices.pt \
  evaluation/motion_v1/mesh_projector/smpl_vertex_face_ids.pt \
  evaluation/motion_v1/mesh_projector/smpl_vertex_barycentric.pt \
  --local-dir weights
```

This preserves `weights/evaluation/motion_v1/` and its `mesh_projector/`
subdirectory. The joint projectors are required for the respective benchmark.
PVE and foot sliding additionally require the mesh projector and locally
prepared ground-truth vertices. To run without those mesh metrics, set
`evaluation.mesh_projector: null` in your configuration and omit
`--include-mesh` during preparation.

## Download official data and ground-truth model files

Follow the [official dataset and SMPL download instructions](BODY_BENCHMARKS.md#obtain-the-official-datasets-and-smpl-models).
Use the following local roots:

- **3DPW:** extract the official `imageFiles.zip` and `sequenceFiles.zip` under
  `data/3dpw/`, preserving `imageFiles/<sequence>/` and `sequenceFiles/test/`.
- **EMDB:** apply through the [official access page](https://emdb.ait.ethz.ch/),
  then extract the approved download under `data/emdb/`, preserving the
  participant/sequence directories, images and annotation pickle files.
- **SMPL:** obtain male, female and neutral models from the
  [official model site](https://smpl.is.tue.mpg.de/) and prepare
  `data/smpl/SMPL_MALE.pkl`, `SMPL_FEMALE.pkl` and `SMPL_NEUTRAL.pkl` for the
  `smplx` loader.

Motion 3DPW additionally requires the J14 ground-truth regressor distributed
in the [official GVHMR repository](https://github.com/zju3dv/GVHMR/blob/main/hmr4d/utils/body_model/smpl_3dpw14_J_regressor_sparse.pt).
Its [3DPW evaluation callback](https://github.com/zju3dv/GVHMR/blob/main/hmr4d/model/gvhmr/callbacks/metric_3dpw.py)
specifies this regressor and pelvis indices 2/3. Download it directly upstream:

```bash
mkdir -p data/gvhmr-support
curl -L \
  https://raw.githubusercontent.com/zju3dv/GVHMR/main/hmr4d/utils/body_model/smpl_3dpw14_J_regressor_sparse.pt \
  -o data/gvhmr-support/smpl_3dpw14_J_regressor_sparse.pt
```

Dataset images, annotations, SMPL model files and prepared ground truth remain
local. GEM does not distribute them. Follow the providers' access and use terms.

## Set the evaluation paths

The example configuration uses environment variables, so all model and asset
paths below are absolute. Run from the checkout root:

```bash
export GEM_POSE_CHECKPOINT="$PWD/weights/body_s240000.ckpt"
export GEM_VITPOSE_CHECKPOINT="$PWD/weights/vitpose.pth"
export GEM_MOTION_CHECKPOINT="$PWD/weights/motion_native77_e774_s292175.ckpt"
export GEM_MOTION_STATS="$PWD/weights/motion_stats_raw.npz"
export GEM_SOMA_ASSETS="$PWD/assets/soma"
export GEM_MOTION_EVAL_ASSETS="$PWD/weights/evaluation/motion_v1"
```

## Prepare full sequences and extract features

Each output directory receives an `index.json` and per-sequence files. First
prepare targets, masks, benchmark boxes and camera inputs:

```bash
python -m gem.motion.evaluation.prepare --stage targets \
  --dataset emdb1 --data-root data/emdb --smpl-model-dir data/smpl \
  --output-dir data/prepared/motion_emdb1 --include-mesh

python -m gem.motion.evaluation.prepare --stage targets \
  --dataset emdb2 --data-root data/emdb --smpl-model-dir data/smpl \
  --output-dir data/prepared/motion_emdb2 --include-mesh

python -m gem.motion.evaluation.prepare --stage targets \
  --dataset 3dpw --data-root data/3dpw --smpl-model-dir data/smpl \
  --joint-regressor data/gvhmr-support/smpl_3dpw14_J_regressor_sparse.pt \
  --output-dir data/prepared/motion_3dpw
```

Then extract the frozen Body Pose query features and original native77 ViTPose
observations for each directory:

```bash
python -m gem.motion.evaluation.prepare --stage features \
  --output-dir data/prepared/motion_emdb1 \
  --config docs/configs/eval_motion_native77.yaml --device cuda

python -m gem.motion.evaluation.prepare --stage features \
  --output-dir data/prepared/motion_emdb2 \
  --config docs/configs/eval_motion_native77.yaml --device cuda

python -m gem.motion.evaluation.prepare --stage features \
  --output-dir data/prepared/motion_3dpw \
  --config docs/configs/eval_motion_native77.yaml --device cuda
```

`--stage all` runs both stages with the combined arguments. The preparation
command's `--batch-size` controls ground-truth model evaluation; feature batch
sizes are `pose.batch_size` and `observations.batch_size` in the YAML.
Use `--sequences` followed by exact sequence IDs to prepare a subset.

The index format is `gem_motion_benchmark_index_v1`. Each entry supplies
`dataset_id`, `sequence_id`, `path`, `frame_count` and `valid_frame_count`.
Per-sequence paths resolve relative to the index directory. A sequence contains
the complete time axis, model conditions and local `eval_targets`; GT validity
is kept as a mask rather than deleting frames before inference.

## Evaluate raw and Contact/IK predictions

```bash
gem-eval motion --config docs/configs/eval_motion_native77.yaml \
  --data-index data/prepared/motion_emdb1/index.json \
  --output outputs/evaluation/motion_emdb1.json --postprocess both --device cuda

gem-eval motion --config docs/configs/eval_motion_native77.yaml \
  --data-index data/prepared/motion_emdb2/index.json \
  --output outputs/evaluation/motion_emdb2.json --postprocess both --device cuda

gem-eval motion --config docs/configs/eval_motion_native77.yaml \
  --data-index data/prepared/motion_3dpw/index.json \
  --output outputs/evaluation/motion_3dpw.json --postprocess both --device cuda
```

`--postprocess both` is the default. Select `raw` or `pp` to score only one
variant. Inference runs on each complete sequence; Contact/IK then processes
the full continuous 30 Hz prediction before applying the evaluation mask.
Raw and PP use the same inputs, checkpoint, cameras and scored frames.
`--sequences` accepts space-separated exact names, or `dataset/sequence`
identifiers when needed to distinguish entries.

The JSON report separates `results.raw` and `results.pp`. Each contains
`metrics` and `metric_sample_counts` grouped by `EMDB_1`, `EMDB_2` or `3DPW`.
The `sequences` list records input/valid frame counts, input protocols and each
sequence's raw/PP statistics. Inference and geometry scoring use IEEE FP32.

## Metrics and masks

Camera evaluation reports pelvis-aligned MPJPE and similarity-aligned PA-MPJPE
in millimeters. EMDB uses 24 joints with hips 1/2 as the pelvis; Motion 3DPW
uses its dedicated J14 order with hips 2/3. With mesh inputs, EMDB-1 additionally
reports pelvis-aligned PVE in millimeters and acceleration error in m/s² at
30 Hz. This release does not add those mesh-path outputs to 3DPW.
EMDB-1 also reports `camera_root_translation_error_mm`, the unaligned absolute
joint-0 camera-position error. It is distinct from the pelvis-aligned pose
metrics and the world trajectory RTE percentage.

EMDB-2 also reports world metrics, with the following JSON names:

| JSON metric | Meaning |
| --- | --- |
| `mpjpe`, `pa_mpjpe` | Camera-coordinate joint errors, in millimeters |
| `wa2_mpjpe` | W-MPJPE100, in millimeters |
| `waa_mpjpe` | WA-MPJPE100, in millimeters |
| `rte` | Root trajectory error, as a percentage of the GT root path length |
| `jitter` | Mean joint jerk at 30 Hz divided by 10 |
| `fs` | Foot sliding in millimeters per retained GT contact sample; requires mesh inputs |

The
[metric definitions](EVAL.md#gem-motion-emdb-2) describe alignment windows,
units and the jitter reporting scale. These follow the
[GVHMR reference](https://github.com/zju3dv/GVHMR/blob/main/hmr4d/utils/eval/eval_utils.py).
Results pool metric samples across sequences rather than averaging sequence
means equally. W-MPJPE100 fits its similarity transform to the first two frames
of each window; WA-MPJPE100 fits the whole window. The last partial window is
also scored. World metrics require at least four valid frames, and RTE requires
a nonzero GT joint-0 trajectory length. Foot sliding measures each of four foot
vertices at transitions where that GT vertex moves less than 1 cm; no contact
samples means no foot-sliding measurement, not zero error.

Invalid GT frames are removed **after** full-sequence inference and
postprocessing, **before** reference temporal metrics. Temporal differences
and 100-frame windows therefore operate on the compressed valid-frame sequence,
using the reference 30 Hz convention. This is distinct from computing temporal
differences over the original timeline and masking them afterward.

## Input protocol and comparison with published results

The published EMDB-2 table uses dataset-provided intrinsics and camera rotations,
original ViTPose observations, native77 final775 and enabled Contact/IK.
This benchmark preparation also uses dataset cameras. It is not a test of
automatic camera-motion estimation or SLAM.

The public raw-data preparation uses official EMDB boxes and 3DPW positive-
confidence 2D joints to form person boxes, followed by the 3:4 crop convention
and a 1.2 expansion. Missing boxes are interpolated with endpoint holding to
retain the complete sequence. These boxes differ from the detector caches used
for historical feature extraction, so newly extracted features and observations
must not be presented as numerically identical to those historical inputs.

3DPW uses the official camera-valid mask, covering 35,515 person-image samples.
Historical Motion evaluations using a WHAM mask scored 34,561 samples; that mask
is not substituted into this raw-data workflow. Keep the mask protocol and
coverage with every result. A different mask or box source changes the benchmark
inputs even when checkpoint and metric names match.

To evaluate a caller-owned 3DPW subset mask, pass `--valid-masks local_masks.pt`
to target preparation. The file is a tensor dictionary from exact sequence IDs
(`<sequence>_<person-index>`) to Boolean arrays on each full timeline. The masks
must be subsets of official camera-valid frames. Supplying a mask selects that
protocol explicitly; it does not restore historical detector boxes or features.

If you have matching local boxes, pass `--boxes local_boxes.pt` during target
preparation, then extract features using those boxes. This is a tensor dictionary
from exact sequence IDs to `[T,3]` arrays of final `[center_x, center_y,
square_size]` in full-image pixels. Supplied boxes are used directly, without
another crop expansion or smoothing pass. Both optional dictionaries only need
entries for the sequences selected by `--sequences`.

The [single-frame Body Pose runner](BODY_BENCHMARKS.md) remains joint-only.
Optional mesh metrics in this Motion workflow do not add PVE support to it.
