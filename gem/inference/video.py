# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Disk-backed RGB frames and explicit, bounded 30 FPS video normalization."""

import math
import operator
import subprocess
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np


class VideoFrames(Sequence[np.ndarray]):
    """Read RGB uint8 frames without retaining the decoded video in memory.

    Increasing indices decode sequentially; rewinding seeks to the requested
    frame. A slice returns only its selected frames as a list. Use as a context
    manager, or call ``close`` when the sequence is no longer needed.
    """

    def __init__(self, path):
        self.path = str(Path(path).resolve())
        self._capture = cv2.VideoCapture(self.path)
        if not self._capture.isOpened():
            self.close()
            raise OSError(f"Cannot open video: {self.path}")
        self.frame_count = int(self._capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = float(self._capture.get(cv2.CAP_PROP_FPS))
        if (
            self.frame_count <= 0
            or self.width <= 0
            or self.height <= 0
            or not math.isfinite(self.fps)
            or self.fps <= 0
        ):
            self.close()
            raise OSError(f"Video has no usable frame count, dimensions or frame rate: {self.path}")
        self._next_index = 0

    def __len__(self):
        return self.frame_count

    def __getitem__(self, index):
        if self._capture is None:
            raise ValueError("VideoFrames is closed")
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        index = operator.index(index)
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError("video frame index out of range")
        if index < self._next_index:
            self._capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            self._next_index = index
        while self._next_index < index:
            if not self._capture.grab():
                raise OSError(f"Cannot decode video frame {self._next_index}: {self.path}")
            self._next_index += 1
        valid, frame = self._capture.read()
        if not valid:
            raise OSError(f"Cannot decode video frame {index}: {self.path}")
        self._next_index = index + 1
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def close(self):
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


def _metadata(path):
    with VideoFrames(path) as frames:
        return {
            "path": frames.path,
            "frame_count": frames.frame_count,
            "fps": frames.fps,
            "width": frames.width,
            "height": frames.height,
            "duration": frames.frame_count / frames.fps,
        }


def normalize_video(input_path, output_path, *, start=0.0, duration=None):
    """Convert an explicit source interval to 30 FPS H.264/YUV420P.

    Times are in seconds. A source longer than 60 seconds requires ``duration``;
    no interval is silently shortened. Frames are padded at the right/bottom to
    even dimensions. The caller owns the output path and its cleanup.

    Return ``source`` and ``normalized`` metadata dictionaries with absolute
    paths, frame count, FPS, width, height and duration. ``source`` also records
    ``requested_start``, ``requested_duration`` (None if omitted), and the
    effective ``selected_duration`` before frame-rate conversion.
    """
    source_path, target_path = Path(input_path).resolve(), Path(output_path).resolve()
    if source_path == target_path:
        raise ValueError("Normalized output must have a different path from the source video")
    start = float(start)
    if not math.isfinite(start) or start < 0:
        raise ValueError("Video start must be finite and non-negative")
    if duration is not None:
        duration = float(duration)
        if not math.isfinite(duration) or not 0 < duration <= 60:
            raise ValueError("Video duration must be finite and within (0, 60] seconds")
    source = _metadata(source_path)
    if start >= source["duration"]:
        raise ValueError("Video start must be before the end of the source")
    if duration is None and source["duration"] > 60:
        raise ValueError("Source video exceeds 60 seconds; specify --duration explicitly")
    selected_duration = source["duration"] - start if duration is None else duration
    if start + selected_duration > source["duration"] + 1e-9:
        raise ValueError("Requested video interval extends beyond the end of the source")
    target_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-y",
        "-ss",
        f"{start:.9f}",
        "-i",
        str(source_path),
        "-t",
        f"{selected_duration:.9f}",
        "-map",
        "0:v:0",
        "-vf",
        "fps=30,pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-c:v",
        "libx264",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-an",
        "-movflags",
        "+faststart",
        str(target_path),
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)
    normalized = _metadata(target_path)
    source.update(
        requested_start=start,
        requested_duration=duration,
        selected_duration=selected_duration,
    )
    return {"source": source, "normalized": normalized}
