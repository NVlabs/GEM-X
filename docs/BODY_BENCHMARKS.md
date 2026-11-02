# Body Pose benchmarks

[Overview](../README.md) · [Evaluation](EVAL.md) · [Motion benchmarks](MOTION_BENCHMARKS.md)

Run `body_s240000.ckpt` on **3DPW test** or **EMDB-1**. This guide reports
MPJPE, PA-MPJPE and scale-MPJPE in millimeters; Body Pose PVE is not included.
Run commands from the checkout root. Once inputs are ready, each dataset needs
one preparation command and one evaluation command.

Body Pose and Motion share the [two-file projector bundle](https://huggingface.co/nvidia/GEM-X/tree/main/evaluation/projectors_v1):
`emdb_projector.pt` and `3dpw_projector.pt`. Each evaluator selects its protocol.

<a id="obtain-the-official-datasets-and-smpl-models"></a>

## 1. Prepare inputs

```bash
python -m pip install -e '.[eval,benchmark]'
hf download nvidia/GEM-X body_s240000.ckpt --local-dir weights
hf download nvidia/GEM-X --include 'evaluation/projectors_v1/*' --local-dir weights
```

Download only the datasets you want to evaluate. **SOMA is used for predictions;
SMPL is used to prepare ground truth.** Obtain data and SMPL models directly
from their providers under their terms; generated labels remain local.

| Input | Official instructions | Expected local path |
| --- | --- | --- |
| SOMA 0.3.3 assets | [SOMA setup](INSTALL.md#soma-code-and-assets) | `assets/soma/` |
| 3DPW images and annotations | [Download imageFiles and sequenceFiles](https://virtualhumans.mpi-inf.mpg.de/3DPW/license.html) | `data/3dpw/imageFiles/`, `data/3dpw/sequenceFiles/` |
| EMDB images and annotations | [Request access](https://emdb.ait.ethz.ch/) and retain the [official layout](https://github.com/eth-ait/emdb#dataset) | `data/emdb/P0/`, `P1/`, etc. |
| SMPL male, female and neutral models | [Register and download](https://smpl.is.tue.mpg.de/) | `data/smpl/SMPL_MALE.pkl`, `SMPL_FEMALE.pkl`, `SMPL_NEUTRAL.pkl` |
| 3DPW ground-truth regressor | [HMR2 support assets](https://github.com/shubham-goel/4D-Humans#installation) | `data/hmr2-support/data/SMPL_to_J19.pkl` |

<details>
<summary>Get the 3DPW regressor / prepare older SMPL downloads</summary>

If you already have `SMPL_to_J19.pkl`, pass its path to the preparation command.
Otherwise, the [official HMR2 downloader](https://github.com/shubham-goel/4D-Humans/blob/main/hmr2/models/__init__.py)
provides this support archive:

```bash
mkdir -p data/hmr2-support
curl -L https://www.cs.utexas.edu/~pavlakos/4dhumans/hmr2_data.tar.gz \
  -o data/hmr2-support/hmr2_data.tar.gz
tar -xzf data/hmr2-support/hmr2_data.tar.gz -C data/hmr2-support
```

For older SMPL files containing Chumpy objects, follow the upstream
[SMPL setup instructions](https://github.com/vchoutas/smplx#smpl-and-smplh-setup)
before using the `smplx` loader.

</details>

Set the paths used by the evaluation configurations; adjust them to your downloads:

```bash
export GEM_POSE_CHECKPOINT="$PWD/weights/body_s240000.ckpt"
export GEM_SOMA_ASSETS="$PWD/assets/soma"
export GEM_EVAL_ASSETS="$PWD/weights/evaluation/projectors_v1"
export GEM_3DPW_ROOT="$PWD/data/3dpw"
export GEM_3DPW_PREPARED="$PWD/data/prepared/body_3dpw_test.npz"
export GEM_EMDB_ROOT="$PWD/data/emdb"
export GEM_EMDB_PREPARED="$PWD/data/prepared/body_emdb1.npz"
```

`hf download --local-dir weights` preserves the projector subdirectory;
`GEM_EVAL_ASSETS` must point to `weights/evaluation/projectors_v1/` for both tasks.

<a id="prepare-local-benchmark-labels"></a>

## 2. Prepare the dataset

Run the command for your dataset once. It writes local labels and relative
image paths; the original images stay in their download directory.

```bash
# 3DPW test
python -m gem.pose.evaluation.prepare_body \
  --dataset 3dpw --data-root "$GEM_3DPW_ROOT" --smpl-model-dir data/smpl \
  --joint-regressor data/hmr2-support/data/SMPL_to_J19.pkl \
  --output "$GEM_3DPW_PREPARED"

# EMDB-1
python -m gem.pose.evaluation.prepare_body \
  --dataset emdb --data-root "$GEM_EMDB_ROOT" --smpl-model-dir data/smpl \
  --output "$GEM_EMDB_PREPARED"
```

<a id="run-the-fixed-release-configurations"></a>

## 3. Evaluate and read the results

```bash
gem-eval pose --task body --config docs/configs/eval_body_3dpw.yaml \
  --output outputs/evaluation/body_3dpw_test.json

gem-eval pose --task body --config docs/configs/eval_body_emdb.yaml \
  --output outputs/evaluation/body_emdb1.json
```

The configurations use CUDA, FP32, batch size 64 and 8 loader workers, without
test-time augmentation. Check `metrics` and `metric_sample_counts` in the JSON:
expected coverage is **35,515 person-images for 3DPW** and **24,117 frames for
EMDB-1**. Each retained person-image has equal weight. See the
[release results and metric definitions](EVAL.md#gem-pose-body-pose) for comparison.

<a id="joint-protocols-and-interpretation"></a>

<details>
<summary>Protocol reference</summary>

- **3DPW:** `3dpw_projector.pt` supplies Body Pose's 44-joint matrix. Score
  indices 25–37 and 43, centered on joint 39, for the 14-joint Body Pose protocol.
- **EMDB:** `emdb_projector.pt` produces SMPL24; score all joints, centered on
  the midpoint of hips 1 and 2. Indices start at zero.
- Projectors expect SOMA MID meshes with 18,056 vertices in meters. The supplied
  evaluator handles projection and joint selection; native SOMA rig joints are
  not interchangeable with benchmark joints.
- `mpjpe` uses pelvis alignment; `pa_mpjpe` additionally fits rotation, uniform
  scale and translation; `scale_mpjpe` fits nonnegative scale and translation
  without rotation. All outputs are in millimeters.
- Preparation preserves benchmark camera intrinsics and crop rules. 3DPW uses
  positive-confidence 2D joints and a 1.2 box expansion; EMDB uses official boxes
  with the longer side as the initial square. Both use the 192:256 aspect
  convention and 256-pixel square crops, without image undistortion.

</details>
