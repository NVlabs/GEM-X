# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare release Body Pose joint benchmarks from user-downloaded datasets.

SMPL models, 3DPW's SMPL-to-J19 regressor and original dataset annotations are
local inputs. This module does not download or distribute any of those assets.
The resulting NPZ is a local annotation cache; images remain in the dataset.
"""

import argparse
import pickle
from pathlib import Path

import cv2
import numpy as np
import torch


def _load_pickle(path):
    with Path(path).open("rb") as handle:
        return pickle.load(handle, encoding="latin1")


def _square_size(width_height):
    """Expand benchmark boxes to 192:256, then use the enclosing square."""
    width, height = np.asarray(width_height).T
    return np.maximum(width * 256 / 192, height)


@torch.no_grad()
def _regress_joints(model, regressor, pose, betas, *, batch_size, device, trans=None, cameras=None):
    """Regress from posed mesh vertices, rather than returning SMPL FK joints."""
    regressor = torch.as_tensor(regressor, dtype=torch.float32, device=device)
    chunks = []
    for start in range(0, len(pose), batch_size):
        stop = start + batch_size
        parameters = {
            "global_orient": torch.as_tensor(
                pose[start:stop, :3], dtype=torch.float32, device=device
            ),
            "body_pose": torch.as_tensor(pose[start:stop, 3:], dtype=torch.float32, device=device),
            "betas": torch.as_tensor(betas[start:stop], dtype=torch.float32, device=device),
        }
        if trans is not None:
            parameters["transl"] = torch.as_tensor(
                trans[start:stop], dtype=torch.float32, device=device
            )
        vertices = model(**parameters).vertices
        if cameras is not None:
            camera = torch.as_tensor(cameras[start:stop], dtype=torch.float32, device=device)
            vertices = vertices @ camera[:, :3, :3].transpose(1, 2) + camera[:, None, :3, 3]
        chunks.append(torch.einsum("jv,bvc->bjc", regressor, vertices).cpu().numpy())
    return np.concatenate(chunks)


def prepare_3dpw(data_root, models, joint_regressor, *, batch_size=128, device="cpu"):
    """Use test person-frames with valid cameras and the release's J19 targets.

    COCO detections with positive confidence define the box, enlarged by 1.2.
    Camera rotation is composed into the SMPL root; absolute translation is
    omitted, matching the release archive and its J19[14] pelvis alignment.
    The first 25 joint slots are unused; slots 25:44 contain J19.
    """
    data_root = Path(data_root)
    files = sorted((data_root / "sequenceFiles" / "test").glob("*.pkl"))
    if not files:
        raise ValueError(f"No 3DPW test annotations under {data_root / 'sequenceFiles/test'}")
    records = []
    for path in files:
        data = _load_pickle(path)
        sequence = str(data["sequence"])
        for person, gender in enumerate(data["genders"]):
            frames = np.flatnonzero(data["campose_valid"][person])
            if not len(frames):
                continue
            pose = np.asarray(data["poses"][person])[frames].copy()
            cameras = np.asarray(data["cam_poses"])[frames]
            for i in range(len(frames)):
                root_rotation = cv2.Rodrigues(pose[i, :3])[0]
                pose[i, :3] = cv2.Rodrigues(cameras[i, :3, :3] @ root_rotation)[0].ravel()
            betas = np.broadcast_to(
                np.asarray(data["betas"][person])[:10], (len(frames), 10)
            ).copy()
            joints = _regress_joints(
                models[{"m": "male", "f": "female"}[gender]],
                joint_regressor,
                pose,
                betas,
                batch_size=batch_size,
                device=device,
            )
            for i, frame in enumerate(frames):
                detections = np.asarray(data["poses2d"][person][frame]).T
                visible = detections[detections[:, 2] > 0, :2]
                minimum, maximum = visible.min(axis=0), visible.max(axis=0)
                center = (minimum + maximum) / 2
                size = _square_size((maximum - minimum) * 1.2)
                target = np.zeros((44, 3), dtype=np.float32)
                target[25:] = joints[i]
                records.append(
                    (
                        f"imageFiles/{sequence}/image_{frame:05d}.jpg",
                        center,
                        size,
                        data["cam_intrinsics"],
                        target,
                        sequence,
                        person,
                        frame,
                    )
                )
    return _pack("3dpw", records)


def prepare_emdb(data_root, models, *, batch_size=128, device="cpu"):
    """Use EMDB-1 good frames and neutral-regressor joints on gendered meshes.

    World-space SMPL vertices are transformed by the annotation's world-to-
    camera extrinsics. Official boxes become square before aspect expansion,
    retaining the published GEM crop, including its additional 4/3 expansion.
    """
    data_root = Path(data_root)
    records = []
    regressor = models["neutral"].J_regressor
    for path in sorted(data_root.glob("P[0-9]/*/*_data.pkl")):
        data = _load_pickle(path)
        if not data["emdb1"]:
            continue
        frames = np.flatnonzero(data["good_frames_mask"])
        if not len(frames):
            continue
        smpl = data["smpl"]
        pose = np.concatenate((smpl["poses_root"], smpl["poses_body"]), axis=-1)[frames]
        betas = np.broadcast_to(np.asarray(smpl["betas"]), (len(frames), 10)).copy()
        joints = _regress_joints(
            models[data["gender"]],
            regressor,
            pose,
            betas,
            trans=np.asarray(smpl["trans"])[frames],
            cameras=np.asarray(data["camera"]["extrinsics"])[frames],
            batch_size=batch_size,
            device=device,
        )
        # Float32 scale / 200 * 200 reproduces the release dataset's box path.
        boxes = np.asarray(data["bboxes"]["bboxes"], dtype=np.float32)[frames]
        centers = ((boxes[:, :2] + boxes[:, 2:]) / 2).astype(np.float32)
        scale = (boxes[:, 2:] - boxes[:, :2]).max(axis=1).astype(np.float32)
        sides = (scale / 200) * 200
        sizes = _square_size(np.stack((sides, sides), axis=-1))
        for i, frame in enumerate(frames):
            image = path.parent.relative_to(data_root) / "images" / f"{frame:05d}.jpg"
            records.append(
                (
                    image.as_posix(),
                    centers[i],
                    sizes[i],
                    data["camera"]["intrinsics"],
                    joints[i],
                    str(data["name"]),
                    0,
                    frame,
                )
            )
    return _pack("emdb", records)


def _pack(protocol, records):
    if not records:
        raise ValueError(f"No valid {protocol} benchmark samples found")
    columns = list(zip(*records))
    names = (
        "imgname",
        "center",
        "box_size",
        "intrinsics",
        "keypoints_3d",
        "sequence_id",
        "person_id",
        "frame_id",
    )
    data = {name: np.asarray(values) for name, values in zip(names, columns)}
    data["protocol"] = np.asarray(protocol)
    data["intrinsics"] = data["intrinsics"].astype(np.float32)
    return data


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("3dpw", "emdb"), required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--smpl-model-dir",
        type=Path,
        required=True,
        help="Local directory containing SMPL_MALE/FEMALE/NEUTRAL.pkl",
    )
    parser.add_argument(
        "--joint-regressor", type=Path, help="Local SMPL_to_J19.pkl; required for 3DPW"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    if args.dataset == "3dpw" and args.joint_regressor is None:
        parser.error("3DPW requires --joint-regressor pointing to SMPL_to_J19.pkl")
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    import smplx

    genders = ("male", "female", "neutral") if args.dataset == "emdb" else ("male", "female")
    models = {
        gender: smplx.SMPL(model_path=str(args.smpl_model_dir), gender=gender, create_transl=False)
        .to(args.device)
        .eval()
        .requires_grad_(False)
        for gender in genders
    }
    options = {"batch_size": args.batch_size, "device": args.device}
    if args.dataset == "3dpw":
        data = prepare_3dpw(args.data_root, models, _load_pickle(args.joint_regressor), **options)
    else:
        data = prepare_emdb(args.data_root, models, **options)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, **data)
    print(f"Prepared {len(data['imgname'])} {args.dataset} person-frames in {args.output}")


if __name__ == "__main__":
    main()
