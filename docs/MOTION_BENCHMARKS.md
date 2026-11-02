# Body Motion benchmarks

[Overview](../README.md) · [Evaluation](EVAL.md) · [Body Pose benchmarks](BODY_BENCHMARKS.md)

Evaluate the released **native77 final775** checkpoint on complete EMDB-1,
EMDB-2 or 3DPW sequences. Each dataset needs one preparation command and one
`gem-eval motion` command; results include raw and Contact/IK variants.
Run commands from the checkout root.

> **Projectors await publication on Hugging Face.** Their download command below
> becomes usable after publication; matching local assets are required to run now.

## 1. Prepare inputs

Install dependencies and download the matching weights and projector bundle:

```bash
python -m pip install -e '.[eval,benchmark]'
hf download nvidia/GEM-X body_s240000.ckpt vitpose.pth \
  motion_native77_e774_s292175.ckpt motion_stats_raw.npz --local-dir weights
hf download nvidia/GEM-X --include 'evaluation/motion_v1/*' --local-dir weights
```

Use the [shared dataset and SMPL download table](BODY_BENCHMARKS.md#obtain-the-official-datasets-and-smpl-models):
SOMA assets go in `assets/soma/`, SMPL models in `data/smpl/`, and official data
in `data/emdb/` or `data/3dpw/`. SOMA generates predictions; local SMPL models
generate GT. Data, models and generated GT are not bundled with this code.

**For Motion 3DPW, use its J14 regressor below, not Body Pose's
`SMPL_to_J19.pkl`.** Download it from the
[upstream GVHMR repository](https://github.com/zju3dv/GVHMR/blob/main/hmr4d/utils/body_model/smpl_3dpw14_J_regressor_sparse.pt):

```bash
mkdir -p data/gvhmr-support
curl -L \
  https://raw.githubusercontent.com/zju3dv/GVHMR/main/hmr4d/utils/body_model/smpl_3dpw14_J_regressor_sparse.pt \
  -o data/gvhmr-support/smpl_3dpw14_J_regressor_sparse.pt
```

Set these paths from the checkout root:

```bash
export GEM_POSE_CHECKPOINT="$PWD/weights/body_s240000.ckpt"
export GEM_VITPOSE_CHECKPOINT="$PWD/weights/vitpose.pth"
export GEM_MOTION_CHECKPOINT="$PWD/weights/motion_native77_e774_s292175.ckpt"
export GEM_MOTION_STATS="$PWD/weights/motion_stats_raw.npz"
export GEM_SOMA_ASSETS="$PWD/assets/soma"
export GEM_MOTION_EVAL_ASSETS="$PWD/weights/evaluation/motion_v1"
```

<a id="input-protocol-and-comparison-with-published-results"></a>

**Before comparing scores:** this workflow uses dataset-provided camera
intrinsics and rotations. Default boxes come from official annotations, which
can differ from historical detector caches. Motion 3DPW defaults to 35,515
camera-valid samples, rather than the historical 34,561-sample WHAM subset.
Therefore, default results are not claimed to reproduce the old cached-input
table verbatim. Matching local boxes and masks can be supplied as described below.

<a id="prepare-full-sequences-and-extract-features"></a>

## 2. Prepare the dataset and features

Run the command for your dataset. `--stage all` generates GT and extracts the
original ViTPose observations and Body Pose features. It preserves complete
sequences and writes a reusable `index.json` plus per-sequence caches.

```bash
# EMDB-1: camera metrics, including PVE and acceleration
python -m gem.motion.evaluation.prepare --stage all \
  --dataset emdb1 --data-root data/emdb --smpl-model-dir data/smpl \
  --output-dir data/prepared/motion_emdb1 --include-mesh \
  --config docs/configs/eval_motion_native77.yaml --device cuda

# EMDB-2: camera and world metrics, including foot sliding
python -m gem.motion.evaluation.prepare --stage all \
  --dataset emdb2 --data-root data/emdb --smpl-model-dir data/smpl \
  --output-dir data/prepared/motion_emdb2 --include-mesh \
  --config docs/configs/eval_motion_native77.yaml --device cuda

# 3DPW: Motion J14 joint metrics
python -m gem.motion.evaluation.prepare --stage all \
  --dataset 3dpw --data-root data/3dpw --smpl-model-dir data/smpl \
  --joint-regressor data/gvhmr-support/smpl_3dpw14_J_regressor_sparse.pt \
  --output-dir data/prepared/motion_3dpw \
  --config docs/configs/eval_motion_native77.yaml --device cuda
```

## 3. Evaluate and read the results

```bash
gem-eval motion --config docs/configs/eval_motion_native77.yaml \
  --data-index data/prepared/motion_emdb1/index.json \
  --output outputs/evaluation/motion_emdb1.json --postprocess both

gem-eval motion --config docs/configs/eval_motion_native77.yaml \
  --data-index data/prepared/motion_emdb2/index.json \
  --output outputs/evaluation/motion_emdb2.json --postprocess both

gem-eval motion --config docs/configs/eval_motion_native77.yaml \
  --data-index data/prepared/motion_3dpw/index.json \
  --output outputs/evaluation/motion_3dpw.json --postprocess both
```

The fixed config uses CUDA and IEEE FP32. Read `results.raw` and `results.pp`
in the JSON; each contains dataset-level `metrics` and `metric_sample_counts`.
The `sequences` section records frame counts and input protocols.

| Dataset | Reported metrics |
| --- | --- |
| EMDB-1 | MPJPE, PA-MPJPE, PVE, acceleration, absolute camera-root error |
| EMDB-2 | MPJPE, PA-MPJPE, W/WA-MPJPE100, RTE, jitter, foot sliding |
| 3DPW | Motion J14 MPJPE and PA-MPJPE |

Expected valid-frame coverage is **24,117 for EMDB-1**, **43,129 for EMDB-2**
and **35,515 for default 3DPW**. Use `raw` or `pp` instead of `both` to score
one variant. Inference and Contact/IK operate on the full continuous 30 Hz
sequence; GT masks are applied afterward. Metric samples are pooled across
sequences. See [metric definitions](EVAL.md#gem-motion-emdb-2).

## Optional settings

<details>
<summary>Reuse preparation, select sequences, or reduce memory use</summary>

- `--stage targets` prepares GT without neural weights. Later run
  `--stage features --output-dir <prepared-directory> --config
  docs/configs/eval_motion_native77.yaml --device cuda` to extract features.
  Already prepared caches can be evaluated repeatedly without repeating either stage.
- `--sequences` accepts space-separated exact sequence names. The evaluation
  command also accepts `DATASET/sequence` names to disambiguate entries.
- Preparation's `--batch-size` controls GT generation. Set `pose.batch_size`
  and `observations.batch_size` in the YAML to control feature-extraction memory.
- For joint-only scoring, set `evaluation.mesh_projector: null` and omit
  `--include-mesh` during preparation. PVE and foot sliding need the complete
  mesh projector directory and locally generated GT vertices.

</details>

<details>
<summary>Use matching local boxes or a 3DPW subset mask</summary>

Pass `--boxes local_boxes.pt` during preparation to use a tensor dictionary
mapping sequence names to final `[T,3]` center-x, center-y, square-size arrays
in source pixels. Supplied boxes receive no further expansion or smoothing.
Default annotation boxes use a 3:4 crop and 1.2 expansion; missing boxes are
interpolated without dropping frames.

For 3DPW, `--valid-masks local_masks.pt` accepts a dictionary of Boolean `[T]`
arrays. Each must be a subset of the official camera-valid mask. Names use
`<sequence>_<person-index>`; only selected sequences need dictionary entries.
A mask override alone does not restore historical boxes or features. Extract
features again after changing boxes or rerunning target preparation.

</details>

<details>
<summary>Joint conventions and temporal metric details</summary>

EMDB uses SMPL24 with hips 1/2 as the pelvis; Motion 3DPW uses its separate J14
order with hips 2/3. Both use SOMA MID18056 geometry without pose correctives.
Do not substitute the Body Pose projector. The Motion bundle includes
`smpl24.pt`, `3dpw_j14.pt`, and the complete `mesh_projector/` directory.

JSON names `wa2_mpjpe`, `waa_mpjpe` and `fs` mean W-MPJPE100, WA-MPJPE100 and
foot sliding. Distances are in mm, `rte` in percent, `accel` in m/s², and `jitter`
uses the reference jerk/10 scale. `camera_root_translation_error_mm` is an
unaligned joint-0 camera error, distinct from world trajectory RTE.

Invalid frames are compressed before temporal differences and 100-frame windows,
including the last partial window. World metrics need at least four valid frames;
RTE needs a nonzero GT root path. No GT foot-contact samples means an unavailable
foot-sliding measurement, not zero error. These conventions follow the
[reference metric definitions](EVAL.md#gem-motion-emdb-2).

</details>
