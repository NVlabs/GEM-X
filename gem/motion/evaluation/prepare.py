# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare complete calibrated Motion benchmark sequences from local raw data.

``--stage targets`` builds GT and image/camera metadata without neural weights.
``--stage features`` adds original native77 observations and Body Pose tokens.
The default ``all`` runs both stages. Dataset images, SMPL models, regressors,
generated targets and feature caches remain on the user's machine.
"""

import argparse
import json
import pickle
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np
import torch

from gem.common.detectors.tracks import _interpolate_gaps, square_crops
from gem.common.pose2d import load_native77_detector
from gem.inference.body_video import body_release, load_body_video_config
from gem.inference.camera_motion import camera_angular_velocity
from gem.motion.api import MotionConditionSpec
from gem.pose.api import load_pose_checkpoint
from gem.pose.data.crop import prepare_pose_batch


def _load_pickle(path):
    with Path(path).open("rb") as handle:
        return pickle.load(handle, encoding="latin1")


def _filled_boxes(boxes, observed):
    """Use existing track interpolation while retaining every source frame."""
    boxes = np.asarray(boxes, dtype=np.float32).copy()
    observed = np.asarray(observed, dtype=bool) & ((boxes[:, 2:] - boxes[:, :2]) > 0).all(-1)
    if not observed.any():
        raise ValueError("A benchmark track needs at least one positive annotated box")
    ids = np.where(observed, 0, -1)
    _interpolate_gaps(boxes, ids, 0)
    return square_crops(boxes, enlarge=1.2), observed


def raw_sequences(dataset, data_root, valid_masks=None, sequences=None):
    """Read original annotations without shortening their camera/image timeline.

    3DPW defaults to the official ``campose_valid`` mask, not the historical
    WHAM subset. Optional user-provided masks must be subsets of that mask.
    Annotation-based crops are recomputed, not claimed identical to historical
    detector and feature caches.
    """
    data_root = Path(data_root)
    wanted = None if sequences is None else set(sequences)
    if dataset in {"emdb1", "emdb2"}:
        for path in sorted(data_root.glob("P[0-9]/*/*_data.pkl")):
            data = _load_pickle(path)
            if not data[dataset]:
                continue
            if wanted is not None and str(data["name"]) not in wanted:
                continue
            length = int(data["n_frames"])
            boxes = np.asarray(data["bboxes"]["bboxes"], dtype=np.float32)
            observed = np.ones(length, dtype=bool)
            observed[np.asarray(data["bboxes"]["invalid_idxs"], dtype=int)] = False
            crops, observed = _filled_boxes(boxes, observed)
            smpl = data["smpl"]
            yield {
                "dataset_id": "EMDB_1" if dataset == "emdb1" else "EMDB_2",
                "sequence_id": str(data["name"]),
                "gender": data["gender"],
                "mask": np.asarray(data["good_frames_mask"], dtype=bool),
                "mask_protocol": "emdb_good_frames_mask",
                "image_paths": [
                    (path.parent.relative_to(data_root) / "images" / f"{i:05d}.jpg").as_posix()
                    for i in range(length)
                ],
                "smpl_params": {
                    "global_orient": smpl["poses_root"],
                    "body_pose": smpl["poses_body"],
                    "transl": smpl["trans"],
                    "betas": np.broadcast_to(smpl["betas"], (length, 10)).copy(),
                },
                "T_w2c": np.asarray(data["camera"]["extrinsics"]),
                "K_fullimg": np.broadcast_to(data["camera"]["intrinsics"], (length, 3, 3)).copy(),
                "bbx_xys": crops,
                "box_observed": observed,
            }
        return
    if dataset != "3dpw":
        raise ValueError("Motion benchmark dataset must be emdb1, emdb2, or 3dpw")
    for path in sorted((data_root / "sequenceFiles" / "test").glob("*.pkl")):
        data = _load_pickle(path)
        video = str(data["sequence"])
        length = len(data["cam_poses"])
        for person, gender in enumerate(data["genders"]):
            name = f"{video}_{person}"
            if wanted is not None and name not in wanted:
                continue
            mask = np.asarray(data["campose_valid"][person], dtype=bool)
            mask_protocol = "3dpw_campose_valid"
            if valid_masks is not None:
                selected = torch.as_tensor(valid_masks[name])
                if selected.dtype != torch.bool or tuple(selected.shape) != (length,):
                    raise ValueError(f"{name}: supplied mask must be bool[{length}]")
                selected = selected.cpu().numpy()
                if (selected & ~mask).any():
                    raise ValueError(f"{name}: supplied mask selects frames without valid cameras")
                mask = selected
                mask_protocol = "3dpw_user_supplied_subset"
            boxes = np.zeros((length, 4), dtype=np.float32)
            observed = np.zeros(length, dtype=bool)
            for i, points in enumerate(data["poses2d"][person]):
                points = np.asarray(points).T
                visible = points[points[:, 2] > 0, :2]
                if len(visible):
                    boxes[i] = np.concatenate((visible.min(0), visible.max(0)))
                    observed[i] = True
            crops, observed = _filled_boxes(boxes, observed)
            pose = np.asarray(data["poses"][person])
            yield {
                "dataset_id": "3DPW",
                "sequence_id": name,
                "gender": {"m": "male", "f": "female"}[gender],
                "mask": mask,
                "mask_protocol": mask_protocol,
                "image_paths": [f"imageFiles/{video}/image_{i:05d}.jpg" for i in range(length)],
                "smpl_params": {
                    "global_orient": pose[:, :3],
                    "body_pose": pose[:, 3:],
                    "transl": np.asarray(data["trans"][person]),
                    "betas": np.broadcast_to(
                        np.asarray(data["betas"][person])[:10], (length, 10)
                    ).copy(),
                },
                "T_w2c": np.asarray(data["cam_poses"]),
                "K_fullimg": np.broadcast_to(data["cam_intrinsics"], (length, 3, 3)).copy(),
                "bbx_xys": crops,
                "box_observed": observed,
            }


@torch.no_grad()
def build_targets(record, model, regressor, *, device="cpu", batch_size=100, include_mesh=False):
    """Regress gendered world vertices before applying world-to-camera transforms.

    Device mesh allocations are bounded by ``batch_size``. Optional complete
    vertex targets are allocated once on CPU for the per-sequence cache.
    """
    regressor = torch.as_tensor(regressor, dtype=torch.float32, device=device)
    length = len(record["mask"])
    targets = {"incam": torch.empty(length, regressor.shape[0], 3)}
    if record["dataset_id"] != "3DPW":
        targets["global"] = torch.empty_like(targets["incam"])
    for start in range(0, length, batch_size):
        stop = min(start + batch_size, length)
        parameters = {
            key: torch.as_tensor(value[start:stop], dtype=torch.float32, device=device)
            for key, value in record["smpl_params"].items()
        }
        with torch.autocast(device_type=torch.device(device).type, enabled=False):
            vertices = model(**parameters).vertices
            joints = torch.matmul(regressor, vertices)
            camera = torch.as_tensor(
                record["T_w2c"][start:stop], dtype=torch.float32, device=device
            )
            incam = joints @ camera[:, :3, :3].transpose(1, 2) + camera[:, None, :3, 3]
            targets["incam"][start:stop] = incam.cpu()
            if "global" in targets:
                targets["global"][start:stop] = joints.cpu()
            if include_mesh:
                if start == 0:
                    targets["verts_incam"] = torch.empty(length, vertices.shape[1], 3)
                    if "global" in targets:
                        targets["verts_global"] = torch.empty_like(targets["verts_incam"])
                targets["verts_incam"][start:stop] = (
                    vertices @ camera[:, :3, :3].transpose(1, 2) + camera[:, None, :3, 3]
                ).cpu()
                if "verts_global" in targets:
                    targets["verts_global"][start:stop] = vertices.cpu()
    return targets


def _index(output_dir):
    path = Path(output_dir) / "index.json"
    if path.exists():
        return json.loads(path.read_text())
    return {"format": "gem_motion_benchmark_index_v1", "sequences": []}


def _write_index(output_dir, index):
    (Path(output_dir) / "index.json").write_text(json.dumps(index, indent=2) + "\n")


def prepare_targets(
    dataset,
    data_root,
    output_dir,
    models,
    *,
    joint_regressor=None,
    valid_masks=None,
    boxes=None,
    sequences=None,
    device="cpu",
    batch_size=100,
    include_mesh=False,
):
    """Write one full tensor cache per track and an index with relative paths."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    index = _index(output_dir)
    entries = {(entry["dataset_id"], entry["sequence_id"]): entry for entry in index["sequences"]}
    wanted = None if sequences is None else set(sequences)
    found = set()
    regressor = models["neutral"].J_regressor if dataset != "3dpw" else joint_regressor
    for record in raw_sequences(dataset, data_root, valid_masks=valid_masks, sequences=sequences):
        name = record["sequence_id"]
        if wanted is not None and name not in wanted:
            continue
        found.add(name)
        length = len(record["mask"])
        box_protocol = "raw_annotations_aspect_3_4_enlarge_1_2_interpolated"
        if boxes is not None:
            supplied = torch.as_tensor(boxes[name], dtype=torch.float32)
            if (
                tuple(supplied.shape) != (length, 3)
                or not torch.isfinite(supplied).all()
                or not (supplied[:, 2] > 0).all()
            ):
                raise ValueError(
                    f"{name}: supplied boxes must be finite [T,3] with positive square sizes"
                )
            record["bbx_xys"] = supplied.cpu().numpy()
            record["box_observed"] = np.ones(length, dtype=bool)
            box_protocol = "user_supplied_bbx_xys"
        cameras = torch.as_tensor(record["T_w2c"], dtype=torch.float32)
        targets = build_targets(
            record,
            models[record["gender"]],
            regressor,
            device=device,
            batch_size=batch_size,
            include_mesh=include_mesh,
        )
        payload = {
            "format": "gem_motion_benchmark_v1",
            "dataset_id": record["dataset_id"],
            "sequence_id": name,
            "data_root": str(Path(data_root).resolve()),
            "image_paths": record["image_paths"],
            "features_ready": False,
            "mask": torch.from_numpy(record["mask"].copy()),
            "mask_protocol": record["mask_protocol"],
            "box_protocol": box_protocol,
            "protocol": {
                "mask_protocol": record["mask_protocol"],
                "bbox_source": box_protocol,
                "camera_source": "dataset_intrinsics_and_extrinsics",
                "image_source": "official_raw_images",
                "feature_source": "not_extracted",
            },
            "box_observed": torch.from_numpy(record["box_observed"].copy()),
            "eval_targets": targets,
            "T_w2c": cameras,
            "frame_ids": torch.arange(length),
            "fps": 30.0,
            "conditions": {
                "length": torch.tensor([length]),
                "bbx_xys": torch.as_tensor(record["bbx_xys"], dtype=torch.float32).unsqueeze(0),
                "K_fullimg": torch.as_tensor(record["K_fullimg"], dtype=torch.float32).unsqueeze(0),
                "cam_angvel": camera_angular_velocity(cameras[:, :3, :3]).unsqueeze(0),
                "frame_times": (torch.arange(length, dtype=torch.float64) / 30).unsqueeze(0),
                "sampling_fps": 30.0,
            },
        }
        relative = Path(record["dataset_id"]) / f"{name}.pt"
        path = output_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, path)
        entries[(record["dataset_id"], name)] = {
            "dataset_id": record["dataset_id"],
            "sequence_id": name,
            "path": relative.as_posix(),
            "frame_count": length,
            "valid_frame_count": int(payload["mask"].sum()),
            "mask_protocol": record["mask_protocol"],
            "features_ready": False,
            "box_protocol": box_protocol,
        }
        print(f"Prepared targets: {record['dataset_id']}/{name} ({length} frames)", flush=True)
    if wanted is not None and wanted - found:
        raise ValueError(f"Requested sequences not found: {sorted(wanted - found)}")
    if not found:
        raise ValueError(f"No {dataset} sequences found under {data_root}")
    index["sequences"] = sorted(
        entries.values(), key=lambda entry: (entry["dataset_id"], entry["sequence_id"])
    )
    _write_index(output_dir, index)
    return index


class ImageFrames(Sequence):
    """Decode only the RGB frames requested by a feature-extraction batch."""

    def __init__(self, data_root, image_paths):
        self.root, self.paths = Path(data_root), image_paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        path = self.root / self.paths[index]
        image = cv2.imread(str(path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        if image is None:
            raise OSError(f"Cannot read benchmark image {path}")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


@torch.inference_mode()
def extract_pose_features(frames, boxes, intrinsics, model, *, batch_size, device):
    """Use public Motion-training crops with each frame's calibrated camera."""
    tokens = []
    version = None
    with torch.autocast(device_type=torch.device(device).type, enabled=False):
        for start in range(0, len(frames), batch_size):
            stop = min(start + batch_size, len(frames))
            batch = prepare_pose_batch(
                frames[start:stop],
                boxes[start:stop],
                intrinsics[start:stop],
                crop_convention="motion_training",
                device=device,
            )
            condition = model.encode(batch)
            tokens.append(condition.pose_tokens.cpu())
            version = condition.feature_version
    return torch.cat(tokens), version


def extract_features(output_dir, config, *, device="cpu", sequences=None):
    """Load each frozen model once, then process all selected sequence caches."""
    output_dir = Path(output_dir)
    index = _index(output_dir)
    wanted = None if sequences is None else set(sequences)
    entries = [
        entry for entry in index["sequences"] if wanted is None or entry["sequence_id"] in wanted
    ]
    if not entries or (wanted is not None and wanted - {entry["sequence_id"] for entry in entries}):
        raise ValueError("Requested feature-extraction sequences are missing from index.json")
    release = body_release(config)
    expected = MotionConditionSpec(**config["motion"]["condition_spec"])
    actual = MotionConditionSpec(
        f"{release.pose_release_id}:pose_tokens",
        config["observations"]["release_id"],
        "soma77_native_v1",
    )
    if actual != expected:
        raise ValueError(
            f"Motion benchmark feature configuration differs from release: {actual} != {expected}"
        )
    observations = config["observations"]
    detector = load_native77_detector(
        observations["checkpoint"],
        release_id=observations["release_id"],
        device=device,
        batch_size=observations.get("batch_size", 4),
    )
    for entry in entries:
        path = output_dir / entry["path"]
        payload = torch.load(path, map_location="cpu", weights_only=True)
        frames = ImageFrames(payload["data_root"], payload["image_paths"])
        payload["features_ready"] = False
        payload["conditions"].pop("f_imgseq", None)
        payload["conditions"]["kp2d_native77"] = (
            detector.predict(frames, payload["conditions"]["bbx_xys"][0]).cpu().unsqueeze(0)
        )
        torch.save(payload, path)
        entry["features_ready"] = False
        print(f"Extracted native77: {entry['dataset_id']}/{entry['sequence_id']}", flush=True)
    del detector
    _write_index(output_dir, index)
    model = load_pose_checkpoint(config["pose"]["checkpoint"], release, device=device)
    for entry in entries:
        path = output_dir / entry["path"]
        payload = torch.load(path, map_location="cpu", weights_only=True)
        conditions = payload["conditions"]
        frames = ImageFrames(payload["data_root"], payload["image_paths"])
        features, version = extract_pose_features(
            frames,
            conditions["bbx_xys"][0],
            conditions["K_fullimg"][0],
            model,
            batch_size=config["pose"].get("batch_size", 4),
            device=device,
        )
        actual = MotionConditionSpec(version, observations["release_id"], "soma77_native_v1")
        if actual != expected:
            raise ValueError(
                f"Extracted Pose feature identity differs from Motion release: {actual} != {expected}"
            )
        conditions.update(f_imgseq=features.unsqueeze(0), condition_spec=asdict(actual))
        payload["protocol"]["feature_source"] = asdict(actual)
        payload["features_ready"] = entry["features_ready"] = True
        torch.save(payload, path)
        _write_index(output_dir, index)
        print(f"Completed features: {entry['dataset_id']}/{entry['sequence_id']}", flush=True)
    return index


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("all", "targets", "features"), default="all")
    parser.add_argument("--dataset", choices=("emdb1", "emdb2", "3dpw"))
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--smpl-model-dir", type=Path)
    parser.add_argument(
        "--joint-regressor", type=Path, help="3DPW Motion smpl_3dpw14_J_regressor_sparse.pt"
    )
    parser.add_argument(
        "--valid-masks", type=Path, help="Optional 3DPW sequence_id -> bool[T] tensor dictionary"
    )
    parser.add_argument(
        "--boxes", type=Path, help="Optional sequence_id -> bbx_xys[T,3] tensor dictionary"
    )
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sequences", nargs="+")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=100, help="SMPL target batch size")
    parser.add_argument(
        "--include-mesh", action="store_true", help="Also store full GT meshes for PVE/foot sliding"
    )
    args = parser.parse_args(argv)
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.stage in {"all", "features"} and args.config is None:
        parser.error("--config is required for feature extraction")
    if args.stage in {"all", "targets"}:
        if args.dataset is None or args.data_root is None or args.smpl_model_dir is None:
            parser.error("Target preparation requires --dataset, --data-root, and --smpl-model-dir")
        if args.dataset == "3dpw" and args.joint_regressor is None:
            parser.error("3DPW requires --joint-regressor with the Motion 14-joint regressor")
        if args.valid_masks is not None and args.dataset != "3dpw":
            parser.error("--valid-masks is only available for 3DPW")
        import smplx

        models = {
            gender: smplx.SMPL(str(args.smpl_model_dir), gender=gender, create_transl=False)
            .to(args.device)
            .eval()
            .requires_grad_(False)
            for gender in ("male", "female", "neutral")
        }
        regressor = None
        if args.joint_regressor is not None:
            regressor = torch.load(
                args.joint_regressor, map_location="cpu", weights_only=True
            ).to_dense()
            if tuple(regressor.shape) != (14, 6890):
                parser.error("3DPW Motion joint regressor must have shape [14,6890]")
        masks = (
            None
            if args.valid_masks is None
            else torch.load(args.valid_masks, map_location="cpu", weights_only=True)
        )
        boxes = (
            None
            if args.boxes is None
            else torch.load(args.boxes, map_location="cpu", weights_only=True)
        )
        prepare_targets(
            args.dataset,
            args.data_root,
            args.output_dir,
            models,
            joint_regressor=regressor,
            valid_masks=masks,
            boxes=boxes,
            sequences=args.sequences,
            device=args.device,
            batch_size=args.batch_size,
            include_mesh=args.include_mesh,
        )
        del models
    if args.stage in {"all", "features"}:
        extract_features(
            args.output_dir,
            load_body_video_config(args.config),
            device=args.device,
            sequences=args.sequences,
        )


if __name__ == "__main__":
    main()
