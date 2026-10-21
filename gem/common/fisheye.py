# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Projection and unprojection for FISHEYE624 and KB4 fisheye cameras.

Both models distort ray angle with an odd polynomial. FISHEYE624 adds tangential
and thin-prism offsets; KB4 uses only four radial coefficients. The FISHEYE624
offset equations follow Project Aria's FisheyeRadTanThinPrism model.

Calibration uses the decoded image orientation; no device-specific rotation is applied.
"""

import cv2
import numpy as np
from scipy.optimize import newton
from scipy.spatial.transform import Rotation

# Iteration limit for inverse distortion. The FISHEYE624 offset tolerance follows
# Project Aria camera_projections/Common.h in normalized image coordinates.
_MAX_INVERSE_ITERATIONS = 50
_FISHEYE624_TOLERANCE = 1e-7
# Numerical tolerance for KB4 inversion, expressed in source-image pixels.
_KB4_MAX_PIXEL_ERROR = 1e-6


class FisheyeCamera:
    """An angular-polynomial camera in the input image's coordinate frame."""

    def __init__(
        self,
        K: np.ndarray,
        radial: np.ndarray,
        width: int,
        height: int,
        tangential_prism: np.ndarray | None = None,
        model_name: str = "KB4",
    ) -> None:
        """Initialize the camera and its projection polynomials.

        Args:
            K: Zero-skew 3x3 intrinsics, preserving independent focal lengths.
            radial: Four KB4 or six FISHEYE624 radial coefficients.
            width: Image width in pixels.
            height: Image height in pixels.
            tangential_prism: Optional [p0, p1, s0, s1, s2, s3] coefficients:
                p0/p1 are tangential; s0/s1 and s2/s3 are the x/y prism terms.
            model_name: Camera model, ``KB4`` or ``FISHEYE624``.
        """
        K = np.array(K, dtype=np.float64, copy=True)
        radial_coeffs = np.array(radial, dtype=np.float64, copy=True)
        tangential_prism_coeffs = (
            np.zeros(6)
            if tangential_prism is None
            else np.array(tangential_prism, dtype=np.float64, copy=True)
        )
        radial_term_count = {"KB4": 4, "FISHEYE624": 6}.get(model_name)
        if (
            K.shape != (3, 3)
            or not np.isfinite(K).all()
            or K[0, 0] <= 0
            or K[1, 1] <= 0
            or K[0, 1] != 0
            or K[1, 0] != 0
            or not np.array_equal(K[2], [0, 0, 1])
            or radial_coeffs.shape != (radial_term_count,)
            or not np.isfinite(radial_coeffs).all()
            or tangential_prism_coeffs.shape != (6,)
            or not np.isfinite(tangential_prism_coeffs).all()
            or (model_name == "KB4" and np.any(tangential_prism_coeffs))
            or width <= 0
            or height <= 0
        ):
            raise ValueError("Invalid camera parameters")
        self.K = K
        self.radial = radial_coeffs
        self.width = width
        self.height = height
        self.tangential_prism = tangential_prism_coeffs
        self.model_name = model_name
        self._radial_polynomial = np.concatenate(([1.0], radial_coeffs))
        self._radial_derivative = self._radial_polynomial * np.arange(
            1, 2 * len(self._radial_polynomial), 2
        )

    @classmethod
    def from_dict(cls, calibration: dict) -> "FisheyeCamera":
        """Read a generic calibration in the decoded RGB image coordinates.

        Required fields are ``model`` (``kb4`` or ``fisheye624``), ``width``,
        ``height``, a zero-skew 3x3 ``intrinsics`` matrix, and ``radial`` (four
        or six coefficients). FISHEYE624 additionally accepts
        ``tangential_prism`` ordered ``[p0, p1, s0, s1, s2, s3]``.
        """
        return cls(
            calibration["intrinsics"],
            calibration["radial"],
            calibration["width"],
            calibration["height"],
            calibration.get("tangential_prism"),
            model_name=calibration["model"].upper(),
        )

    def to_dict(self) -> dict:
        """Return a JSON-compatible calibration in the source image coordinates."""
        calibration = {
            "model": self.model_name.lower(),
            "width": self.width,
            "height": self.height,
            "intrinsics": self.K.tolist(),
            "radial": self.radial.tolist(),
        }
        if self.model_name == "FISHEYE624":
            calibration["tangential_prism"] = self.tangential_prism.tolist()
        return calibration

    def project(self, rays: np.ndarray) -> np.ndarray:
        """Project camera rays through radial and optional tangential/prism distortion.

        Args:
            rays: Camera-coordinate vectors shaped (..., 3), with positive z forward.

        Returns:
            Pixel coordinates shaped (..., 2); nonfinite and rear-facing rays produce NaN.
        """
        rays = np.asarray(rays, dtype=np.float64)
        flat_rays = rays.reshape(-1, 3)
        ray_xy_norm = np.hypot(flat_rays[:, 0], flat_rays[:, 1])
        theta = np.arctan2(ray_xy_norm, flat_rays[:, 2])
        distorted_radii = theta * np.polynomial.polynomial.polyval(
            theta * theta, self._radial_polynomial
        )
        xy_directions = np.divide(
            flat_rays[:, :2],
            ray_xy_norm[:, None],
            out=np.zeros_like(flat_rays[:, :2]),
            where=(ray_xy_norm[:, None] > 0) & np.isfinite(ray_xy_norm[:, None]),
        )
        distorted_normalized_coords = xy_directions * distorted_radii[:, None]
        if np.any(self.tangential_prism):
            # Add FISHEYE624 tangential/thin-prism offsets. KB4 has no such terms.
            distorted_normalized_coords = (
                distorted_normalized_coords
                + self._fisheye624_tangential_prism_offsets(distorted_normalized_coords)
            )
        pixel_coords = distorted_normalized_coords * [self.K[0, 0], self.K[1, 1]] + self.K[:2, 2]
        pixel_coords[
            (flat_rays[:, 2] <= 0) | ~np.isfinite(flat_rays).all(axis=1) | ~np.isfinite(ray_xy_norm)
        ] = np.nan
        return pixel_coords.reshape(*rays.shape[:-1], 2)

    def unproject(self, pixels: np.ndarray) -> np.ndarray:
        """Invert optional tangential/prism offsets and the shared radial polynomial.

        Args:
            pixels: Source image coordinates shaped (..., 2).

        Returns:
            Unit forward rays shaped (..., 3), with NaN outside the lens domain
            or when inversion fails. KB4 inversion covers radii beyond pi/2.
        """
        pixels = np.asarray(pixels, dtype=np.float64)
        flat_pixels = pixels.reshape(-1, 2)
        valid_pixels = np.isfinite(flat_pixels).all(axis=1)
        distorted_normalized_coords = (
            np.where(valid_pixels[:, None], flat_pixels, self.K[:2, 2]) - self.K[:2, 2]
        )
        distorted_normalized_coords /= [self.K[0, 0], self.K[1, 1]]
        radial_coords = distorted_normalized_coords.copy()
        if np.any(self.tangential_prism):
            # Undo FISHEYE624 tangential/thin-prism offsets before radial inversion.
            p0, p1, s0, s1, s2, s3 = self.tangential_prism
            for _ in range(_MAX_INVERSE_ITERATIONS):
                offset_residual = (
                    distorted_normalized_coords
                    - radial_coords
                    - self._fisheye624_tangential_prism_offsets(radial_coords)
                )
                x, y = radial_coords.T
                radius_squared = x * x + y * y
                jacobian_xx = 1 + 6 * p0 * x + 2 * p1 * y + 2 * s0 * x + 4 * s1 * radius_squared * x
                jacobian_xy = 2 * p0 * y + 2 * p1 * x + 2 * s0 * y + 4 * s1 * radius_squared * y
                jacobian_yx = 2 * p1 * x + 2 * p0 * y + 2 * s2 * x + 4 * s3 * radius_squared * x
                jacobian_yy = 1 + 6 * p1 * y + 2 * p0 * x + 2 * s2 * y + 4 * s3 * radius_squared * y
                jacobian_determinant = jacobian_xx * jacobian_yy - jacobian_xy * jacobian_yx
                offset_step_numerator = np.column_stack(
                    (
                        jacobian_yy * offset_residual[:, 0] - jacobian_xy * offset_residual[:, 1],
                        jacobian_xx * offset_residual[:, 1] - jacobian_yx * offset_residual[:, 0],
                    )
                )
                offset_step = np.divide(
                    offset_step_numerator,
                    jacobian_determinant[:, None],
                    out=np.zeros_like(radial_coords),
                    where=jacobian_determinant[:, None] != 0,
                )
                radial_coords += offset_step
                if np.all(np.abs(offset_step) < _FISHEYE624_TOLERANCE):
                    break
            # Recompute offsets after the last update to check the inverse result.
            offset_residual = (
                radial_coords
                + self._fisheye624_tangential_prism_offsets(radial_coords)
                - distorted_normalized_coords
            )
            valid_pixels &= np.max(np.abs(offset_residual), axis=1) < _FISHEYE624_TOLERANCE

        distorted_radii = np.hypot(radial_coords[:, 0], radial_coords[:, 1])
        max_distorted_radius = (np.pi / 2) * np.polynomial.polynomial.polyval(
            (np.pi / 2) ** 2, self._radial_polynomial
        )
        valid_pixels &= np.isfinite(distorted_radii) & (distorted_radii < max_distorted_radius)
        target_radii = distorted_radii[valid_pixels]
        # Solve the radial polynomial with its analytic derivative.
        theta = np.minimum(target_radii, np.pi / 2)
        if target_radii.size:
            try:
                theta = newton(
                    lambda theta, radius: (
                        theta
                        * np.polynomial.polynomial.polyval(theta * theta, self._radial_polynomial)
                        - radius
                    ),
                    theta,
                    fprime=lambda theta, radius: np.polynomial.polynomial.polyval(
                        theta * theta, self._radial_derivative
                    ),
                    args=(target_radii,),
                    maxiter=_MAX_INVERSE_ITERATIONS,
                    disp=False,
                )
            except RuntimeError:
                theta.fill(np.nan)
        radius_residual = (
            theta * np.polynomial.polynomial.polyval(theta * theta, self._radial_polynomial)
            - target_radii
        )
        if self.model_name == "KB4":
            radius_tolerance = _KB4_MAX_PIXEL_ERROR / max(self.K[0, 0], self.K[1, 1])
        else:
            radius_tolerance = _FISHEYE624_TOLERANCE
        pixel_ray_angles = np.zeros_like(distorted_radii)
        pixel_ray_angles[valid_pixels] = theta
        valid_pixels[valid_pixels] = (
            (theta >= 0) & (theta < np.pi / 2) & (np.abs(radius_residual) < radius_tolerance)
        )
        xy_directions = np.divide(
            radial_coords,
            distorted_radii[:, None],
            out=np.zeros_like(radial_coords),
            where=distorted_radii[:, None] > 0,
        )
        rays = np.column_stack(
            (
                xy_directions * np.sin(pixel_ray_angles)[:, None],
                np.cos(pixel_ray_angles),
            )
        )
        rays[~valid_pixels] = np.nan
        return rays.reshape(*pixels.shape[:-1], 3)

    def _fisheye624_tangential_prism_offsets(self, radial_coords: np.ndarray) -> np.ndarray:
        """Compute FISHEYE624 tangential and thin-prism displacements.

        Args:
            radial_coords: Normalized image coordinates after radial distortion,
                shaped (N, 2).

        Returns:
            Displacements (dx, dy) to add in normalized coordinates, shaped (N, 2).
        """
        x, y = radial_coords.T
        radius_squared = x * x + y * y
        p0, p1, s0, s1, s2, s3 = self.tangential_prism
        return np.column_stack(
            (
                p0 * (3 * x * x + y * y)
                + 2 * p1 * x * y
                + s0 * radius_squared
                + s1 * radius_squared * radius_squared,
                p1 * (x * x + 3 * y * y)
                + 2 * p0 * x * y
                + s2 * radius_squared
                + s3 * radius_squared * radius_squared,
            )
        )


def perspective_crop(
    image_rgb: np.ndarray,
    bbox: np.ndarray,
    camera,
    crop_side_pixels: int = 256,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create a square perspective crop of a detected hand region.

    The camera turns toward the box-center ray using the shortest rotation from
    the source optical axis, following UmeTrack with ``camera_angle=0``.
    Its focal length fits valid directions densely sampled along the box edges.
    Each crop pixel is sampled from the source image; unavailable samples are black.

    Args:
        image_rgb: ``(H, W, 3)`` source image in RGB channel order.
        bbox: ``(3,)`` square box ordered ``center_x``, ``center_y``, ``side_length``,
            in source image pixels.
        camera: Source camera with project/unproject methods for ``image_rgb``.
        crop_side_pixels: Side length of the square crop to produce, in
            pixels. The caller sets it from the resolution its model consumes.

    Returns:
        Tuple of the square RGB crop (uint8, same range as input),
        virtual pinhole intrinsics, full-crop bbox ``[center_x, center_y,
        side_length]`` in crop pixels, and the rotation from crop-camera to
        source-camera coordinates, in that order.
    """
    bbox_center_x, bbox_center_y, bbox_side = np.asarray(bbox, dtype=np.float64)

    # The curved perimeter, including each corner, determines the full angular
    # extent; considering only four corners can cut into a fisheye hand region.
    bbox_left = bbox_center_x - bbox_side / 2
    bbox_right = bbox_center_x + bbox_side / 2
    bbox_top = bbox_center_y - bbox_side / 2
    bbox_bottom = bbox_center_y + bbox_side / 2
    # Sample each edge at four times the crop resolution, including both endpoints.
    edge_sample_count = 4 * crop_side_pixels + 1
    # Row 0 is the box center; the rest walk its four edges.
    bbox_sample_pixels = np.concatenate(
        [
            [[bbox_center_x, bbox_center_y]],
            np.linspace([bbox_left, bbox_top], [bbox_right, bbox_top], edge_sample_count),
            np.linspace([bbox_left, bbox_bottom], [bbox_right, bbox_bottom], edge_sample_count),
            np.linspace([bbox_left, bbox_top], [bbox_left, bbox_bottom], edge_sample_count),
            np.linspace([bbox_right, bbox_top], [bbox_right, bbox_bottom], edge_sample_count),
        ]
    )
    bbox_sample_rays = camera.unproject(bbox_sample_pixels)
    center_ray = bbox_sample_rays[0]
    source_perimeter_rays = bbox_sample_rays[1:]
    source_perimeter_rays = source_perimeter_rays[np.isfinite(source_perimeter_rays).all(axis=1)]
    if not np.isfinite(center_ray).all() or len(source_perimeter_rays) == 0:
        raise ValueError("Cannot undistort this bounding box")

    # Match UmeTrack's make_look_at_matrix with camera_angle=0: rotate the source
    # optical axis onto the box-center ray by the shortest rotation.
    # https://github.com/facebookresearch/UmeTrack/blob/main/lib/common/affine.py
    crop_rotation, _ = Rotation.align_vectors(center_ray, [0.0, 0.0, 1.0])
    crop_to_source_rotation = crop_rotation.as_matrix()

    # The edge rays set the crop camera's focal length: each lands on its z=1 plane, and
    # the one furthest off axis is put at the crop's edge.
    virtual_perimeter_rays = source_perimeter_rays @ crop_to_source_rotation
    if np.any(virtual_perimeter_rays[:, 2] <= 0):
        raise ValueError("ROI perimeter reaches behind the virtual pinhole camera")
    normalized_perimeter_coords = virtual_perimeter_rays[:, :2] / virtual_perimeter_rays[:, 2, None]
    crop_center = (crop_side_pixels - 1) / 2.0
    max_normalized_extent = np.max(np.abs(normalized_perimeter_coords))
    focal_length = (crop_side_pixels / 2.0) / max_normalized_extent

    # Each crop pixel is a ray. Rotated into source camera coordinates and projected, it
    # gives the source pixel that crop pixel samples.
    crop_grid_y, crop_grid_x = np.indices((crop_side_pixels, crop_side_pixels), dtype=np.float64)
    virtual_crop_rays = np.stack(
        (
            (crop_grid_x - crop_center) / focal_length,
            (crop_grid_y - crop_center) / focal_length,
            np.ones_like(crop_grid_x),
        ),
        axis=-1,
    )
    source_pixels = camera.project(virtual_crop_rays @ crop_to_source_rotation.T)
    # Map unavailable projections outside the image so cv2.remap fills them
    # with black via BORDER_CONSTANT and borderValue=0.
    source_pixels = np.where(np.isfinite(source_pixels), source_pixels, -1.0)
    crop_rgb = cv2.remap(
        image_rgb,
        source_pixels[..., 0].astype(np.float32),
        source_pixels[..., 1].astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    K_virtual = np.array(
        [
            [focal_length, 0, crop_center],
            [0, focal_length, crop_center],
            [0, 0, 1],
        ],
        dtype=np.float64,
    )
    bbox_virtual = np.array(
        [crop_center, crop_center, crop_side_pixels],
        dtype=np.float64,
    )
    return crop_rgb, K_virtual, bbox_virtual, crop_to_source_rotation
