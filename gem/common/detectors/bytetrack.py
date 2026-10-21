# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Retained GEM implementation of ByteTrack/Kalman association."""

from __future__ import annotations

import numpy as np


def _iou_batch(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between (M, 4) and (N, 4) xyxy boxes → (M, N)."""
    x1 = np.maximum(boxes_a[:, None, 0], boxes_b[None, :, 0])
    y1 = np.maximum(boxes_a[:, None, 1], boxes_b[None, :, 1])
    x2 = np.minimum(boxes_a[:, None, 2], boxes_b[None, :, 2])
    y2 = np.minimum(boxes_a[:, None, 3], boxes_b[None, :, 3])
    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-7)


class _KalmanBoxTracker:
    """Tracks a single object with a Kalman filter over [x, y, a, h] state.

    State vector: [cx, cy, aspect_ratio, height, vx, vy, va, vh].
    Measurement:  [cx, cy, aspect_ratio, height].
    """

    _count = 0

    def __init__(self, bbox: np.ndarray, score: float):
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        a = w / (h + 1e-7)

        self.x = np.array([cx, cy, a, h, 0, 0, 0, 0], dtype=np.float64)
        self.P = np.eye(8) * 10.0
        self.P[4:, 4:] *= 100.0  # high uncertainty on velocities

        # Process noise
        self.Q = np.eye(8) * 1.0
        self.Q[4:, 4:] *= 0.01

        # Measurement noise
        self.R = np.diag([1.0, 1.0, 10.0, 1.0])

        # Transition and observation matrices
        self.F = np.eye(8)
        self.F[:4, 4:] = np.eye(4)  # position += velocity
        self.H = np.zeros((4, 8))
        self.H[:4, :4] = np.eye(4)

        self.id = _KalmanBoxTracker._count
        _KalmanBoxTracker._count += 1

        self.hits = 1
        self.time_since_update = 0
        self.score = score

    def predict(self) -> np.ndarray:
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.time_since_update += 1
        return self._state_to_xyxy()

    def update(self, bbox: np.ndarray, score: float):
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        z = np.array([cx, cy, w / (h + 1e-7), h])

        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(8) - K @ self.H) @ self.P

        self.hits += 1
        self.time_since_update = 0
        self.score = score

    def _state_to_xyxy(self) -> np.ndarray:
        cx, cy, a, h = self.x[:4]
        w = a * h
        return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])


class ByteTracker:
    """ByteTrack multi-object tracker (MIT license).

    Two-stage association:
      1. Match high-score detections to existing tracks (IoU).
      2. Match remaining low-score detections to unmatched tracks (IoU).
    No appearance model — pure motion-based, very fast.

    Parameters
    ----------
    high_thr : float
        Score threshold to split detections into high / low groups.
    match_thr : float
        IoU threshold for matching.
    max_lost : int
        Frames to keep a lost track before removal.
    """

    def __init__(
        self,
        high_thr: float = 0.5,
        match_thr: float = 0.3,
        max_lost: int = 30,
    ):
        self.high_thr = high_thr
        self.match_thr = match_thr
        self.max_lost = max_lost
        self.tracks: list[_KalmanBoxTracker] = []
        _KalmanBoxTracker._count = 0

    def update(self, boxes: np.ndarray, scores: np.ndarray) -> list[tuple[np.ndarray, int, float]]:
        """Process one frame of detections.

        Parameters
        ----------
        boxes : (N, 4) xyxy
        scores : (N,)

        Returns
        -------
        results : list of (box_xyxy, track_id, score)
        """
        # Predict all existing tracks
        for t in self.tracks:
            t.predict()

        # Split detections into high and low score groups
        high_mask = scores >= self.high_thr
        low_mask = ~high_mask & (scores > 0.1)

        high_boxes, high_scores = boxes[high_mask], scores[high_mask]
        low_boxes, low_scores = boxes[low_mask], scores[low_mask]

        # --- Stage 1: match high-score dets to tracks ---
        matched_t, matched_d, unmatched_tracks, unmatched_dets = self._match(
            self.tracks, high_boxes, self.match_thr
        )

        for t_idx, d_idx in zip(matched_t, matched_d):
            self.tracks[t_idx].update(high_boxes[d_idx], high_scores[d_idx])

        # --- Stage 2: match low-score dets to remaining tracks ---
        remaining_tracks = [self.tracks[i] for i in unmatched_tracks]
        if len(remaining_tracks) > 0 and len(low_boxes) > 0:
            mt2, md2, still_unmatched, _ = self._match(remaining_tracks, low_boxes, self.match_thr)
            for t_idx, d_idx in zip(mt2, md2):
                remaining_tracks[t_idx].update(low_boxes[d_idx], low_scores[d_idx])
            unmatched_tracks = [unmatched_tracks[i] for i in still_unmatched]

        # --- Create new tracks from unmatched high-score detections ---
        for d_idx in unmatched_dets:
            self.tracks.append(_KalmanBoxTracker(high_boxes[d_idx], high_scores[d_idx]))

        # --- Remove dead tracks ---
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_lost]

        # --- Collect active results ---
        results = []
        for t in self.tracks:
            if t.time_since_update == 0:
                results.append((t._state_to_xyxy(), t.id, t.score))

        return results

    @staticmethod
    def _match(
        tracks: list[_KalmanBoxTracker],
        boxes: np.ndarray,
        thr: float,
    ) -> tuple[list[int], list[int], list[int], list[int]]:
        """Hungarian matching between tracks and detections."""
        if len(tracks) == 0 or len(boxes) == 0:
            return [], [], list(range(len(tracks))), list(range(len(boxes)))

        from scipy.optimize import linear_sum_assignment

        track_boxes = np.array([t._state_to_xyxy() for t in tracks])
        iou_matrix = _iou_batch(track_boxes, boxes)
        cost = 1.0 - iou_matrix

        row_ind, col_ind = linear_sum_assignment(cost)

        matched_t, matched_d = [], []
        unmatched_t = set(range(len(tracks)))
        unmatched_d = set(range(len(boxes)))

        for r, c in zip(row_ind, col_ind):
            if iou_matrix[r, c] >= thr:
                matched_t.append(r)
                matched_d.append(c)
                unmatched_t.discard(r)
                unmatched_d.discard(c)

        return matched_t, matched_d, sorted(unmatched_t), sorted(unmatched_d)
