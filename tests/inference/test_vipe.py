# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import subprocess
import sys
from pathlib import Path
from textwrap import dedent

import numpy as np
import pytest
import torch

from gem.inference.vipe import run_vipe_camera


@pytest.fixture
def vipe_stub(tmp_path):
    """A real subprocess that emits small official-format artifacts, without models."""
    video = tmp_path / "normalized clip.mp4"
    video.touch()
    artifacts = tmp_path / "artifacts"
    (artifacts / "pose").mkdir(parents=True)
    (artifacts / "intrinsics").mkdir()
    poses = np.repeat(np.eye(4)[None], 3, axis=0)
    poses[1, :3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    poses[2, :3, :3] = [[0, 0, 1], [0, 1, 0], [-1, 0, 0]]
    poses[:, :3, 3] = [[1, 2, 3], [4, 5, 6], [10, 20, 30]]
    intrinsics = np.array([[600, 610, 320, 240], [620, 630, 321, 241], [640, 650, 322, 242]])
    pose_order = [1, 2, 0]
    intrinsics_order = [2, 0, 1]
    np.savez(artifacts / "pose" / f"{video.stem}.npz", data=poses[pose_order], inds=pose_order)
    np.savez(
        artifacts / "intrinsics" / f"{video.stem}.npz",
        data=intrinsics[intrinsics_order],
        inds=intrinsics_order,
    )
    arguments = tmp_path / "arguments.json"
    exit_code = tmp_path / "exit_code.txt"
    exit_code.write_text("0")
    executable = tmp_path / "vipe stub"
    executable.write_text(
        f"#!{sys.executable}\n"
        + dedent(
            f"""\
            import json
            import shutil
            import sys
            from pathlib import Path

            args = sys.argv[1:]
            Path({str(arguments)!r}).write_text(json.dumps(args))
            code = int(Path({str(exit_code)!r}).read_text())
            if code:
                sys.exit(code)
            output = args[args.index('-o') + 1]
            shutil.copytree({str(artifacts)!r}, output, dirs_exist_ok=True)
            """
        )
    )
    executable.chmod(0o755)
    return {
        "video": video,
        "artifacts": artifacts,
        "executable": executable,
        "arguments": arguments,
        "exit_code": exit_code,
        "poses": poses,
        "intrinsics": intrinsics,
    }


def test_cli_camera_conversion_preserves_per_frame_intrinsics_and_cleans_up(vipe_stub):
    result = run_vipe_camera(
        vipe_stub["video"],
        frame_count=3,
        executable=vipe_stub["executable"],
        pipeline="custom_pose_pipeline",
    )
    expected_poses = torch.tensor(vipe_stub["poses"], dtype=torch.float32)
    torch.testing.assert_close(result["C2W"], expected_poses)
    torch.testing.assert_close(result["R_w2c"], expected_poses[:, :3, :3].transpose(-1, -2))
    torch.testing.assert_close(
        result["cam_angvel"],
        torch.tensor([[0, 1, 0, -1, 0, 0], [0, 0, -1, 1, 0, 0], [0, 0, -1, 1, 0, 0.0]]),
    )
    expected_intrinsics = torch.tensor(
        [
            [[600, 0, 320], [0, 610, 240], [0, 0, 1]],
            [[620, 0, 321], [0, 630, 241], [0, 0, 1]],
            [[640, 0, 322], [0, 650, 242], [0, 0, 1]],
        ],
        dtype=torch.float32,
    )
    torch.testing.assert_close(result["K_fullimg"], expected_intrinsics)
    assert result["diagnostics"]["method"] == "vipe"
    assert result["diagnostics"]["pipeline"] == "custom_pose_pipeline"
    assert result["diagnostics"]["translation_used"] is False
    arguments = json.loads(vipe_stub["arguments"].read_text())
    assert arguments[:5] == ["infer", str(vipe_stub["video"]), "-p", "custom_pose_pipeline", "-o"]
    assert len(arguments) == 6
    assert not Path(arguments[-1]).exists()


def test_single_frame_accepts_explicit_pinhole_camera(vipe_stub):
    stem = vipe_stub["video"].stem
    for kind in ("pose", "intrinsics"):
        values = vipe_stub["poses" if kind == "pose" else "intrinsics"][:1]
        np.savez(vipe_stub["artifacts"] / kind / f"{stem}.npz", data=values, inds=[0])
    (vipe_stub["artifacts"] / "intrinsics" / f"{stem}_camera.txt").write_text("0: PINHOLE\n")
    result = run_vipe_camera(vipe_stub["video"], frame_count=1, executable=vipe_stub["executable"])
    torch.testing.assert_close(result["cam_angvel"], torch.tensor([[1, 0, 0, 0, 1, 0.0]]))
    assert result["diagnostics"]["pipeline"] == "pose_only_long"


@pytest.mark.parametrize("kind", ["pose", "intrinsics"])
@pytest.mark.parametrize("indices", [[0, 0, 2], [1, 2, 3], [0, 2, 4], [[0], [1], [2]]])
def test_rejects_wrong_frame_timeline(vipe_stub, kind, indices):
    values = vipe_stub["poses" if kind == "pose" else "intrinsics"]
    path = vipe_stub["artifacts"] / kind / f"{vipe_stub['video'].stem}.npz"
    np.savez(path, data=values, inds=indices)
    with pytest.raises(ValueError, match="indices"):
        run_vipe_camera(vipe_stub["video"], frame_count=3, executable=vipe_stub["executable"])


@pytest.mark.parametrize("kind", ["pose", "intrinsics"])
def test_rejects_sparse_artifacts(vipe_stub, kind):
    values = vipe_stub["poses" if kind == "pose" else "intrinsics"]
    path = vipe_stub["artifacts"] / kind / f"{vipe_stub['video'].stem}.npz"
    np.savez(path, data=values[[0, 2]], inds=[0, 2])
    with pytest.raises(ValueError, match="shape"):
        run_vipe_camera(vipe_stub["video"], frame_count=3, executable=vipe_stub["executable"])


@pytest.mark.parametrize("kind", ["pose", "intrinsics"])
def test_rejects_missing_artifacts(vipe_stub, kind):
    (vipe_stub["artifacts"] / kind / f"{vipe_stub['video'].stem}.npz").unlink()
    with pytest.raises(ValueError, match="did not produce.*camera artifact"):
        run_vipe_camera(vipe_stub["video"], frame_count=3, executable=vipe_stub["executable"])


@pytest.mark.parametrize(
    "value, message", [(np.nan, "finite"), (np.inf, "finite"), (0, "positive"), (-1, "positive")]
)
def test_rejects_invalid_focal_length(vipe_stub, value, message):
    intrinsics = vipe_stub["intrinsics"].astype(float)
    intrinsics[1, 0] = value
    np.savez(
        vipe_stub["artifacts"] / "intrinsics" / f"{vipe_stub['video'].stem}.npz",
        data=intrinsics,
        inds=[0, 1, 2],
    )
    with pytest.raises(ValueError, match=message):
        run_vipe_camera(vipe_stub["video"], frame_count=3, executable=vipe_stub["executable"])


def test_rejects_non_pinhole_camera(vipe_stub):
    camera_path = vipe_stub["artifacts"] / "intrinsics" / f"{vipe_stub['video'].stem}_camera.txt"
    camera_path.write_text("0: PINHOLE\n1: OPENCV_FISHEYE\n2: PINHOLE\n")
    with pytest.raises(ValueError, match="requires PINHOLE.*OPENCV_FISHEYE"):
        run_vipe_camera(vipe_stub["video"], frame_count=3, executable=vipe_stub["executable"])


def test_propagates_subprocess_failure_and_cleans_up(vipe_stub):
    vipe_stub["exit_code"].write_text("7")
    with pytest.raises(subprocess.CalledProcessError) as caught:
        run_vipe_camera(vipe_stub["video"], frame_count=3, executable=vipe_stub["executable"])
    assert caught.value.returncode == 7
    arguments = json.loads(vipe_stub["arguments"].read_text())
    assert not Path(arguments[-1]).exists()


def test_missing_cli_explains_separate_environment(tmp_path):
    with pytest.raises(FileNotFoundError, match="Install ViPE in a separate environment"):
        run_vipe_camera(tmp_path / "video.mp4", frame_count=3, executable=tmp_path / "absent-vipe")
