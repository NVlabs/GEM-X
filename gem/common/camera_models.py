# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-OpenMDW-1.1

"""Calibrated source cameras for optional Hand perspective crops."""

import cv2
import numpy as np
from numpy.polynomial import Polynomial

from .fisheye import FisheyeCamera

_MAX_INVERSE_ITERATIONS = 50


class PerspectiveCamera:
    """Pinhole or rational perspective projection with prism and sensor-tilt terms."""

    def __init__(self, K, coefficients, width, height, model_name):
        """Store validated calibration and find the radial branch connected to the axis.

        Args:
            K: Validated, copied source-image intrinsics from the shared loader.
            coefficients: Empty for PINHOLE, or OpenCV's fourteen distortion terms.
            width: Source-image width.
            height: Source-image height.
            model_name: Registered source-camera model identifier.
        """
        self.K = K
        self.distortion_coefficients = coefficients if len(coefficients) else np.zeros(14)
        self.width, self.height, self.model_name = width, height, model_name
        d = self.distortion_coefficients
        numerator = Polynomial(np.r_[1, d[[0, 1, 4]]])
        denominator = Polynomial(np.r_[1, d[[5, 6, 7]]])
        # For s=r², the first pole or zero of d(r*N(s)/D(s))/dr ends the physical branch.
        derivative = numerator * denominator + Polynomial([0, 2]) * (
            numerator.deriv() * denominator - numerator * denominator.deriv()
        )
        roots = np.r_[denominator.roots(), derivative.roots()]
        physical = (roots.real > 0) & (abs(roots.imag) <= 1e-10 * np.maximum(1, abs(roots.real)))
        self._max_radius_squared = np.min(roots.real[physical], initial=np.inf)

    def project(self, rays: np.ndarray) -> np.ndarray:
        """Project forward rays, rejecting folded or disconnected distortion branches.

        Args:
            rays: Source-camera vectors shaped (..., 3).

        Returns:
            Source pixels shaped (..., 2); invalid or rear-facing rays become NaN.
        """
        rays = np.asarray(rays, dtype=np.float64)
        if rays.ndim == 0 or rays.shape[-1] != 3:
            raise ValueError("Camera rays must have shape (..., 3)")
        flat = rays.reshape(-1, 3)
        pixels = np.full((len(flat), 2), np.nan)
        indices = np.flatnonzero(np.isfinite(flat).all(axis=1) & (flat[:, 2] > 0))
        if np.isfinite(self._max_radius_squared):
            radii = np.sum((flat[indices, :2] / flat[indices, 2, None]) ** 2, axis=1)
            indices = indices[radii < self._max_radius_squared]
        if len(indices):
            if not np.any(self.distortion_coefficients):
                pixels[indices] = (flat[indices, :2] / flat[indices, 2, None]) * self.K.diagonal()[
                    :2
                ] + self.K[:2, 2]
            else:
                projected, jacobian = cv2.projectPoints(
                    flat[indices],
                    np.zeros(3),
                    np.zeros(3),
                    self.K,
                    self.distortion_coefficients,
                )
                projected = projected.reshape(-1, 2)
                determinant = np.linalg.det(jacobian[:, 3:5].reshape(-1, 2, 2))
                projected[~np.isfinite(determinant) | (determinant <= 0)] = np.nan
                pixels[indices] = projected
        pixels[~np.isfinite(pixels).all(axis=1)] = np.nan
        return pixels.reshape(*rays.shape[:-1], 2)

    def unproject(self, pixels: np.ndarray) -> np.ndarray:
        """Invert perspective distortion with a safeguarded Newton fallback.

        Args:
            pixels: Source-image coordinates shaped (..., 2).

        Returns:
            Unit forward rays shaped (..., 3); invalid inversions become NaN.
            Distorted rays must reproject within 1e-6 pixels on the physical branch.
        """
        pixels = np.asarray(pixels, dtype=np.float64)
        if pixels.ndim == 0 or pixels.shape[-1] != 2:
            raise ValueError("Camera pixels must have shape (..., 2)")
        flat = pixels.reshape(-1, 2)
        finite = np.isfinite(flat).all(axis=1)
        target = flat[finite]
        rays = np.full((len(flat), 3), np.nan)
        xy = (target - self.K[:2, 2]) / self.K.diagonal()[:2]
        distorted = np.any(self.distortion_coefficients)
        if distorted and len(target):
            xy = cv2.undistortPointsIter(
                target[:, None],
                self.K,
                self.distortion_coefficients,
                None,
                None,
                (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 50, 1e-10),
            ).reshape(-1, 2)
            error = np.max(abs(self.project(np.c_[xy, np.ones(len(xy))]) - target), axis=1)
            active = np.flatnonzero(~np.isfinite(error) | (error > 1e-6))
            # Restart failed fixed-point inversions near the pinhole ray, inside the branch.
            xy[active] = (target[active] - self.K[:2, 2]) / self.K.diagonal()[:2]
            radius_squared = np.sum(xy[active] ** 2, axis=1)
            outside = radius_squared >= self._max_radius_squared
            xy[active[outside]] *= np.sqrt(
                0.9 * self._max_radius_squared / radius_squared[outside]
            )[:, None]
            for _ in range(_MAX_INVERSE_ITERATIONS):
                if not len(active):
                    break
                projected, jacobian = cv2.projectPoints(
                    np.c_[xy[active], np.ones(len(active))],
                    np.zeros(3),
                    np.zeros(3),
                    self.K,
                    self.distortion_coefficients,
                )
                residual = projected.reshape(-1, 2) - target[active]
                error = np.max(abs(residual), axis=1)
                # At z=1, translation-x/y derivatives equal derivatives with respect to x/y.
                jacobian = jacobian[:, 3:5].reshape(-1, 2, 2)
                solvable = (
                    np.isfinite(residual).all(axis=1)
                    & np.isfinite(jacobian).all(axis=(1, 2))
                    & (np.linalg.det(jacobian) > np.finfo(np.float64).eps)
                    & (error > 1e-6)
                )
                active = active[solvable]
                if not len(active):
                    break
                step = np.linalg.solve(jacobian[solvable], residual[solvable, :, None])[:, :, 0]
                scale = np.ones(len(active))
                for _ in range(20):
                    candidate = xy[active] - step * scale[:, None]
                    trial = self.project(np.c_[candidate, np.ones(len(candidate))])
                    trial_error = np.max(abs(trial - target[active]), axis=1)
                    improved = np.isfinite(trial_error) & (trial_error < error[solvable])
                    if np.all(improved):
                        break
                    scale[~improved] *= 0.5
                xy[active[improved]] = candidate[improved]
                active = active[improved]
        forward = np.c_[xy, np.ones(len(xy))]
        if distorted and len(target):
            error = np.max(abs(self.project(forward) - target), axis=1)
            forward[~np.isfinite(error) | (error > 1e-6)] = np.nan
        forward[~np.isfinite(forward).all(axis=1)] = np.nan
        forward /= np.max(abs(forward), axis=1, keepdims=True)
        forward /= np.linalg.norm(forward, axis=1, keepdims=True)
        rays[finite] = forward
        return rays.reshape(*pixels.shape[:-1], 3)

    def to_dict(self):
        """Serialize calibration in GEM's decoded-source-image convention."""
        return {
            "model": self.model_name.lower(),
            "width": self.width,
            "height": self.height,
            "intrinsics": self.K.tolist(),
            "distortion_coefficients": (
                [] if self.model_name == "PINHOLE" else self.distortion_coefficients.tolist()
            ),
        }


def camera_from_dict(calibration):
    """Read GEM or shared camera JSON in the decoded image orientation.

    GEM fields are model, width, height and intrinsics. Fisheye models use
    radial and optional tangential_prism; perspective models use
    distortion_coefficients (empty for pinhole, fourteen OpenCV terms for
    rational_polynomial). Shared JSON instead uses image_width/image_height,
    camera_matrix.matrix_3x3 and a single distortion_coefficients vector.
    No device-specific rotation, resize or calibration lookup is performed.
    """
    model = calibration["model"].upper()
    coefficient_count = {"PINHOLE": 0, "RATIONAL_POLYNOMIAL": 14, "KB4": 4, "FISHEYE624": 12}
    if model not in coefficient_count:
        raise ValueError(f"Unsupported camera model: {model}")
    shared = "camera_matrix" in calibration
    if not shared and model in ("KB4", "FISHEYE624"):
        return FisheyeCamera.from_dict(calibration)
    width = calibration["image_width" if shared else "width"]
    height = calibration["image_height" if shared else "height"]
    K = np.array(
        calibration["camera_matrix"]["matrix_3x3"] if shared else calibration["intrinsics"],
        dtype=np.float64,
        copy=True,
    )
    coefficients = np.array(calibration["distortion_coefficients"], dtype=np.float64)
    if (
        type(width) is not int
        or type(height) is not int
        or min(width, height) <= 0
        or K.shape != (3, 3)
        or not np.isfinite(K).all()
        or min(K[0, 0], K[1, 1]) <= 0
        or K[0, 1] != 0
        or K[1, 0] != 0
        or not np.array_equal(K[2], [0, 0, 1])
        or coefficients.shape != (coefficient_count[model],)
        or not np.isfinite(coefficients).all()
    ):
        raise ValueError(
            "Camera calibration requires positive raster dimensions, zero-skew intrinsics and matching finite distortion coefficients"
        )
    if model in ("KB4", "FISHEYE624"):
        count = 4 if model == "KB4" else 6
        return FisheyeCamera(
            K,
            coefficients[:count],
            width,
            height,
            coefficients[count:] if model == "FISHEYE624" else None,
            model_name=model,
        )
    return PerspectiveCamera(K, coefficients, width, height, model)
