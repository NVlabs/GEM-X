# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Optional Open3D mesh rendering with explicit camera coordinates.

Vertices are already in the caller's world or camera frame. ``extrinsic`` is
world-to-camera in OpenCV coordinates (+X right, +Y down, +Z forward). This
module applies no body translation, axis conversion, grounding or framing.
"""

import os
from importlib.metadata import version
from pathlib import Path

import numpy as np


class MotionMeshRenderer:
    """Reuse one headless rendering context for camera overlays or world views.

    Open3D is an optional dependency, imported at construction. ``software``
    selects Mesa EGL in the current process; ``egl`` keeps the caller's EGL
    driver selection. Choose the backend before creating any Open3D context.
    """

    def __init__(self, width, height, *, backend="software"):
        if backend not in {"software", "egl"}:
            raise ValueError("Render backend must be 'software' or 'egl'")
        installed = version("open3d")
        if tuple(int(part) for part in installed.split(".")[:2]) != (0, 19):
            raise RuntimeError(
                f"GEM rendering requires Open3D 0.19; installed version is {installed}. "
                "Render saved results in an Open3D 0.19 environment, such as the supported "
                "x86 setup; ARM64 Open3D 0.20 video rendering has not passed validation."
            )
        if backend == "software":
            mesa = Path("/usr/share/glvnd/egl_vendor.d/50_mesa.json")
            if not mesa.is_file():
                raise RuntimeError("Software rendering requires the Mesa EGL vendor library")
            os.environ["EGL_PLATFORM"] = "surfaceless"
            os.environ["LIBGL_ALWAYS_SOFTWARE"] = "true"
            os.environ["__EGL_VENDOR_LIBRARY_FILENAMES"] = str(mesa)
        import open3d as o3d

        self._o3d = o3d
        self.width, self.height = int(width), int(height)
        self._renderer = o3d.visualization.rendering.OffscreenRenderer(self.width, self.height)
        self._ground_bounds = None
        self._body_material = self._material("defaultLit", [0.4, 0.8, 0.4, 1])
        self._body_material.base_metallic = 0.0
        self._body_material.base_roughness = 0.72
        self._ground_material = self._material("defaultUnlit", [1, 1, 1, 1])
        self._line_material = self._material("unlitLine", [0.12, 0.3, 0.7, 1])
        self._line_material.line_width = 3

    @staticmethod
    def _body_color(background):
        if background is not None:
            return [0.4, 0.8, 0.4, 1.0]
        # The deployed world viewer uses sRGB #98bdff. Material colors are
        # linear, as in Three.js's color-managed MeshStandardMaterial.
        rgb = np.array([0x98, 0xBD, 0xFF], dtype=float) / 255
        return [*(((rgb + 0.055) / 1.055) ** 2.4), 1.0]

    def _lighting(self, extrinsic, background):
        """A frontal key and indirect fill preserve readable surface shading.

        Filament uses only one dominant directional light, so the old
        three-light Raymond rig cannot be copied literally. Its frontal light
        and ambient fill are represented by one key plus image-based lighting.
        """
        if background is not None:
            direction = extrinsic[:3, :3].T @ np.array([-0.35, 0.35, np.sqrt(0.755)])
            key_intensity, fill_intensity = 90000, 37500
        else:
            # The deployed world viewer's key is at (+2.5,+4,+3), in Y-up axes.
            direction = np.array([-2.5, -4.0, -3.0])
            key_intensity, fill_intensity = 100000, 45000
        direction = direction / np.linalg.norm(direction)
        self._renderer.scene.set_lighting(
            self._renderer.scene.LightingProfile.NO_SHADOWS, direction
        )
        scene = self._renderer.scene.scene
        scene.set_sun_light(direction, [1, 1, 1], key_intensity)
        # NO_SHADOWS disables the key as well as shadows in Open3D0.19.
        # Restore direct surface lighting, then turn off cast shadows alone.
        scene.enable_sun_light(True)
        self._renderer.scene.view.set_shadowing(False)
        scene.set_indirect_light_intensity(fill_intensity)

    def _material(self, shader, color):
        material = self._o3d.visualization.rendering.MaterialRecord()
        material.shader = shader
        material.base_color = color
        return material

    def _mesh(self, vertices, faces):
        mesh = self._o3d.geometry.TriangleMesh(
            self._o3d.utility.Vector3dVector(np.asarray(vertices, dtype=np.float64)),
            self._o3d.utility.Vector3iVector(np.asarray(faces, dtype=np.int32)),
        )
        mesh.compute_vertex_normals()
        return mesh

    def _remove(self, name):
        if self._renderer.scene.has_geometry(name):
            self._renderer.scene.remove_geometry(name)

    def _ground(self, bounds):
        bounds = None if bounds is None else tuple(map(float, bounds))
        if bounds == self._ground_bounds:
            return
        self._remove("ground")
        self._ground_bounds = bounds
        if bounds is None:
            return
        xmin, xmax, zmin, zmax = bounds
        xs = np.r_[np.arange(xmin, xmax, 0.5), xmax]
        zs = np.r_[np.arange(zmin, zmax, 0.5), zmax]
        vertices, faces, colors = [], [], []
        for i in range(len(xs) - 1):
            for j in range(len(zs) - 1):
                offset = len(vertices)
                vertices.extend(
                    [
                        [xs[i], 0, zs[j]],
                        [xs[i], 0, zs[j + 1]],
                        [xs[i + 1], 0, zs[j + 1]],
                        [xs[i + 1], 0, zs[j]],
                    ]
                )
                faces.extend([[offset, offset + 1, offset + 2], [offset, offset + 2, offset + 3]])
                color = [0.88] * 3 if (i + j) % 2 else [0.68] * 3
                colors.extend([color] * 4)
        mesh = self._mesh(vertices, faces)
        mesh.vertex_colors = self._o3d.utility.Vector3dVector(np.asarray(colors))
        self._renderer.scene.add_geometry("ground", mesh, self._ground_material)

    def render(
        self,
        vertices,
        faces,
        intrinsics,
        extrinsic,
        *,
        background=None,
        opacity=1.0,
        ground_bounds=None,
        trajectory=None,
    ):
        """Return RGB uint8, preserving the supplied full-image intrinsic matrix.

        Camera overlays default to the deployed opaque green surface; world
        views use the deployed light blue. ``opacity`` controls depth-masked
        compositing only when an RGB uint8
        background is supplied. Ground bounds are ``(xmin,xmax,zmin,zmax)`` at
        Y=0; an optional trajectory is an ``(N,3)`` line in the vertex frame.
        """
        if self._renderer is None:
            raise RuntimeError("MotionMeshRenderer is closed")
        if not 0 <= opacity <= 1:
            raise ValueError("Render opacity must be in [0,1]")
        if background is not None:
            background = np.asarray(background)
            if background.shape != (self.height, self.width, 3) or background.dtype != np.uint8:
                raise ValueError("Render background must be RGB uint8 with the viewport dimensions")
        vertices = np.asarray(vertices)
        intrinsics = np.asarray(intrinsics, dtype=np.float64)
        extrinsic = np.asarray(extrinsic, dtype=np.float64)
        renderer = self._renderer
        self._body_material.base_color = self._body_color(background)
        self._remove("body")
        renderer.scene.add_geometry("body", self._mesh(vertices, faces), self._body_material)
        self._ground(ground_bounds)
        self._remove("trajectory")
        if trajectory is not None and len(trajectory) > 1:
            points = np.asarray(trajectory, dtype=np.float64)
            indices = np.arange(len(points) - 1)
            lines = self._o3d.geometry.LineSet(
                self._o3d.utility.Vector3dVector(points),
                self._o3d.utility.Vector2iVector(np.stack((indices, indices + 1), axis=-1)),
            )
            renderer.scene.add_geometry("trajectory", lines, self._line_material)
        renderer.scene.set_background([1, 1, 1, 1] if background is None else [0, 0, 0, 0])
        self._lighting(extrinsic, background)
        renderer.setup_camera(intrinsics, extrinsic, self.width, self.height)
        # Keep the historical 1cm near plane and preserve distant world motion.
        view_depth = vertices @ extrinsic[2, :3] + extrinsic[2, 3]
        far = max(100.0, float(view_depth.max()) * 1.1)
        renderer.scene.camera.set_projection(
            intrinsics, 0.01, far, float(self.width), float(self.height)
        )
        rgb = np.asarray(renderer.render_to_image()).copy()
        if background is None:
            return rgb
        depth = np.asarray(renderer.render_to_depth_image(z_in_view_space=True))
        foreground = np.isfinite(depth) & (depth > 0)
        blended = np.rint(rgb.astype(np.float32) * opacity + background * (1 - opacity))
        return np.where(foreground[..., None], blended, background).clip(0, 255).astype(np.uint8)

    def close(self):
        """Release the native offscreen context; safe to call more than once."""
        self._renderer = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
