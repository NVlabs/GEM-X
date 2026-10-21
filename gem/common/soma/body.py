# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SOMA-X v0.3 adapter for the native fixed-global-scale contract.

The public training/storage contract is intentionally smaller than the current
SOMA-X runtime contract::

    identity128 + external_scale57[exact_one, local56]

SOMA-X v0.3 evaluates 60 local bone-scale controls.  The four controls added
after the legacy 56-value layout are fixed to their neutral value of one at
this adapter boundary.  Global scale is likewise an architectural constant;
it is validated on input and is never represented by a trainable parameter in
this module.

Importing this module does not import :mod:`soma`.  The optional runtime is
created lazily on first use, while tests and callers may inject a small mock
runtime directly.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn as nn

SOMA_IDENTITY_DIM = 128
SOMA_LEARNED_LOCAL_SCALE_DIM = 56
SOMA_EXTERNAL_SCALE_DIM = 57
SOMA_RUNTIME_LOCAL_SCALE_DIM = 60
SOMA_PUBLIC_POSE_JOINTS = 77
SOMA_BODY_POSE_JOINTS = 76
SOMA_X_VERSION = "0.3.3"
SOMA_SUPPORTED_VERSIONS = ("0.3.0", SOMA_X_VERSION)
SOMA_APPENDED_SCALE_NAMES = (
    "LeftFoot",
    "RightFoot",
    "LeftToeBase",
    "RightToeBase",
)


@dataclass(frozen=True)
class SomaFixedGs1ShapeSpec:
    """Shape dimensions and conversions for native fixed GS1."""

    identity_dim: int = SOMA_IDENTITY_DIM
    learned_local_scale_dim: int = SOMA_LEARNED_LOCAL_SCALE_DIM
    runtime_local_scale_dim: int = SOMA_RUNTIME_LOCAL_SCALE_DIM
    fixed_global_scale: float = 1.0
    require_positive_scales: bool = True

    def __post_init__(self) -> None:
        expected = (
            SOMA_IDENTITY_DIM,
            SOMA_LEARNED_LOCAL_SCALE_DIM,
            SOMA_RUNTIME_LOCAL_SCALE_DIM,
            1.0,
        )
        actual = (
            self.identity_dim,
            self.learned_local_scale_dim,
            self.runtime_local_scale_dim,
            self.fixed_global_scale,
        )
        if actual != expected:
            raise ValueError(
                f"the fixed-GS1 shape contract is immutable: expected {expected}, got {actual}"
            )

    @property
    def external_scale_dim(self) -> int:
        return 1 + self.learned_local_scale_dim

    def compose_external_scale(self, local_scale56: torch.Tensor) -> torch.Tensor:
        """Return float32 ``[exact_one, local56]`` while preserving gradients."""

        local = _float32_tensor(local_scale56, name="local_scale56")
        _require_last_dim(local, self.learned_local_scale_dim, name="local_scale56")
        if self.require_positive_scales and bool((local <= 0).any()):
            raise ValueError("local_scale56 must be strictly positive")
        global_scale = torch.ones_like(local[..., :1], dtype=torch.float32)
        return torch.cat((global_scale, local), dim=-1)

    def split_external_scale(self, scale57: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Validate external GS1 and return ``(global_one, local56)``."""

        scale = _float32_tensor(scale57, name="scale57")
        _require_last_dim(scale, self.external_scale_dim, name="scale57")
        supplied_global_scale, local = scale[..., :1], scale[..., 1:]
        if not torch.equal(supplied_global_scale, torch.ones_like(supplied_global_scale)):
            raise ValueError("scale57 must contain an exact float32 global scale of one")
        if self.require_positive_scales and bool((local <= 0).any()):
            raise ValueError("scale57 local scales must be strictly positive")
        # Reconstruct rather than return the input view: GS1 is architecture,
        # not an optimizable channel even when a caller marks scale57 trainable.
        global_scale = torch.ones_like(supplied_global_scale, requires_grad=False)
        return global_scale, local

    def runtime_local_scale(self, local_scale56: torch.Tensor) -> torch.Tensor:
        """Append the four neutral v0.3 Foot/ToeBase controls."""

        local = _float32_tensor(local_scale56, name="local_scale56")
        _require_last_dim(local, self.learned_local_scale_dim, name="local_scale56")
        if self.require_positive_scales and bool((local <= 0).any()):
            raise ValueError("local_scale56 must be strictly positive")
        appended = torch.ones(
            *local.shape[:-1],
            self.runtime_local_scale_dim - self.learned_local_scale_dim,
            dtype=torch.float32,
            device=local.device,
        )
        return torch.cat((local, appended), dim=-1)


@dataclass(frozen=True)
class SomaRuntimeConfig:
    """Lazy construction options for an official SOMA-X v0.3 body layer."""

    data_root: str | Path | None = None
    lod: Literal["mid", "low", "xlo"] = "xlo"
    mode: str = "warp"
    device: str | torch.device = "cuda"
    apply_correctives: bool = False
    enable_procedural_transforms: bool = True
    procedural_reference: Literal["runtime", "legacy_pose021"] = "runtime"

    def __post_init__(self) -> None:
        if self.lod not in {"mid", "low", "xlo"}:
            raise ValueError(f"unsupported SOMA body LOD: {self.lod!r}")
        if not isinstance(self.mode, str) or not self.mode:
            raise ValueError("SOMA runtime mode must be a non-empty string")
        if type(self.apply_correctives) is not bool:
            raise TypeError("apply_correctives must be a bool")
        if type(self.enable_procedural_transforms) is not bool:
            raise TypeError("enable_procedural_transforms must be a bool")
        if self.procedural_reference not in {"runtime", "legacy_pose021"}:
            raise ValueError(f"Unknown SOMA procedural reference {self.procedural_reference!r}")


class _NonPersistentFrozenRuntime(nn.Module):
    """Own a frozen runtime without contributing entries to parent checkpoints."""

    def __init__(self, runtime: nn.Module):
        super().__init__()
        runtime.requires_grad_(False)
        runtime.eval()
        # Deliberately bypass nn.Module.__setattr__: the runtime is fixed external
        # geometry and must not participate in recursive state_dict/load_state_dict.
        object.__setattr__(self, "_runtime", runtime)

    @property
    def module(self) -> nn.Module:
        return object.__getattribute__(self, "_runtime")

    def _apply(self, fn, recurse=True):
        # SOMA-X overrides ``_apply(fn)`` without PyTorch's newer ``recurse``
        # keyword, so call the portable one-argument module hook.
        self.module._apply(fn)
        return super()._apply(fn, recurse=recurse)

    def train(self, mode: bool = True) -> _NonPersistentFrozenRuntime:
        # The body runtime is deterministic geometry, not a learned submodule.
        super().train(False)
        self.module.eval()
        return self

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        if destination is None:
            destination = {}
        return destination

    def _save_to_state_dict(self, destination, prefix, keep_vars) -> None:
        return


RuntimeFactory = Callable[[SomaRuntimeConfig], nn.Module]


class SomaFixedGs1BodyAdapter(nn.Module):
    """Evaluate native fixed-GS1 shapes with an official SOMA-X v0.3 runtime.

    ``global_orient`` is the public Hips/global rotation and ``body_pose`` holds
    the remaining 76 public joint rotations.  Axis-angle inputs may use a flat
    final dimension of ``76 * 3`` or ``(..., 76, 3)``.  Rotation-matrix inputs
    use ``(..., 76, 3, 3)`` and set ``pose2rot=False``.

    Identity, scale, and translation may either carry the full pose prefix or
    omit its final temporal dimension, e.g. ``(B, 128)`` may accompany a
    ``(B, L, 76, 3)`` pose.
    """

    def __init__(
        self,
        runtime_config: SomaRuntimeConfig | None = None,
        *,
        runtime: nn.Module | None = None,
        runtime_factory: RuntimeFactory | None = None,
        shape_spec: SomaFixedGs1ShapeSpec | None = None,
    ):
        super().__init__()
        if runtime is not None and runtime_factory is not None:
            raise ValueError("runtime and runtime_factory are mutually exclusive")
        self.runtime_config = runtime_config or SomaRuntimeConfig()
        self.shape_spec = shape_spec or SomaFixedGs1ShapeSpec()
        self._runtime_factory = runtime_factory
        self._runtime_holder: _NonPersistentFrozenRuntime | None = None
        if runtime is not None:
            self._install_runtime(runtime)

    @property
    def runtime_is_initialized(self) -> bool:
        return self._runtime_holder is not None

    @property
    def runtime(self) -> nn.Module:
        if self._runtime_holder is None:
            factory = self._runtime_factory or _build_official_soma_runtime
            self._install_runtime(factory(self.runtime_config))
        assert self._runtime_holder is not None
        return self._runtime_holder.module

    def _install_runtime(self, runtime: nn.Module) -> None:
        if not isinstance(runtime, nn.Module):
            raise TypeError("SOMA runtime must be an nn.Module")
        identity_dim = getattr(
            getattr(runtime, "identity_model", None), "num_identity_coeffs", None
        )
        if identity_dim != self.shape_spec.identity_dim:
            raise ValueError(f"SOMA runtime must expose identity128, got {identity_dim!r}")
        runtime_scale_dim = getattr(runtime, "num_scale_params", None)
        if runtime_scale_dim != self.shape_spec.runtime_local_scale_dim:
            raise ValueError(f"SOMA-X v0.3 runtime must expose local60, got {runtime_scale_dim!r}")
        scale_names = getattr(runtime, "scale_param_names", None)
        if scale_names is not None:
            names = tuple(str(name) for name in scale_names)
            if len(names) != self.shape_spec.runtime_local_scale_dim:
                raise ValueError("SOMA runtime scale_param_names must contain 60 entries")
            if names[-len(SOMA_APPENDED_SCALE_NAMES) :] != SOMA_APPENDED_SCALE_NAMES:
                raise ValueError(
                    "SOMA runtime appended scale controls do not match Foot/ToeBase semantics"
                )
        self._runtime_holder = _NonPersistentFrozenRuntime(runtime)

    def train(self, mode: bool = True) -> SomaFixedGs1BodyAdapter:
        super().train(mode)
        if self._runtime_holder is not None:
            self._runtime_holder.train(False)
        return self

    def forward(
        self,
        *,
        global_orient: torch.Tensor,
        body_pose: torch.Tensor,
        identity_coeffs: torch.Tensor,
        scale_params: torch.Tensor,
        transl: torch.Tensor | None = None,
        pose2rot: bool = True,
        apply_correctives: bool | None = None,
        return_joints_only: bool = False,
        repose_to_bind_pose: bool | None = None,
    ) -> dict[str, torch.Tensor | None]:
        if apply_correctives is None:
            apply_correctives = self.runtime_config.apply_correctives

        with torch.autocast(device_type=global_orient.device.type, enabled=False):
            full_pose, prefix = _join_public_pose(global_orient, body_pose, pose2rot=pose2rot)
            identity = _expand_feature_tensor(
                identity_coeffs,
                prefix=prefix,
                width=self.shape_spec.identity_dim,
                name="identity_coeffs",
            )
            external_scale = _expand_feature_tensor(
                scale_params,
                prefix=prefix,
                width=self.shape_spec.external_scale_dim,
                name="scale_params",
            )
            fixed_global, local56 = self.shape_spec.split_external_scale(external_scale)
            runtime_local60 = torch.cat((local56, local56.new_ones(*local56.shape[:-1], 4)), dim=-1)
            if transl is None:
                translation = torch.zeros(*prefix, 3, dtype=torch.float32, device=full_pose.device)
            else:
                translation = _expand_feature_tensor(transl, prefix=prefix, width=3, name="transl")

            tensors = (identity, fixed_global, runtime_local60, translation)
            if any(value.device != full_pose.device for value in tensors):
                raise ValueError("all SOMA adapter inputs must be on the same device")

            output = _run_soma_runtime(
                self.runtime,
                full_pose=full_pose,
                identity=identity,
                runtime_local_scale=runtime_local60,
                fixed_global=fixed_global,
                translation=translation,
                prefix=prefix,
                pose2rot=pose2rot,
                apply_correctives=apply_correctives,
                return_joints_only=return_joints_only,
                repose_to_bind_pose=repose_to_bind_pose,
            )

        return _normalize_runtime_output(output, prefix=prefix, device=full_pose.device)


def _build_official_soma_runtime(config: SomaRuntimeConfig) -> nn.Module:
    """Import and construct SOMA-X only when the adapter is first evaluated."""

    try:
        import soma
        from soma import SOMALayer, Unit
    except ImportError as exc:
        raise RuntimeError("SomaFixedGs1BodyAdapter requires py-soma-x==0.3.3") from exc
    version = getattr(soma, "__version__", None)
    if version not in SOMA_SUPPORTED_VERSIONS:
        raise RuntimeError(
            f"SomaFixedGs1BodyAdapter supports source versions {SOMA_SUPPORTED_VERSIONS}, "
            f"got {version!r}"
        )
    kwargs: dict[str, Any] = {
        "identity_model_type": "soma",
        "lod": config.lod,
        "mode": config.mode,
        "device": config.device,
        "output_unit": Unit.METERS,
        "enable_procedural_transforms": config.enable_procedural_transforms,
    }
    if not config.apply_correctives:
        # Motion training and static-FK skip the corrective network. Pose
        # compatibility can explicitly request its actual corrective weights.
        kwargs["correctives_model_path"] = None
    if config.data_root is not None:
        kwargs["data_root"] = str(config.data_root)
    if version == "0.3.3":
        # Preserve the reference used by deployed fixed-GS1 Motion models.
        # This pins orientations; it does not claim to restore an older rig.
        kwargs["reference_pose"] = {"version": "0.2.0"}
    runtime = SOMALayer(**kwargs)
    if config.procedural_reference != "runtime":
        from .procedural import configure_procedural_reference

        configure_procedural_reference(runtime, config.procedural_reference)
    return runtime


def _run_soma_runtime(
    runtime: nn.Module,
    *,
    full_pose: torch.Tensor,
    identity: torch.Tensor,
    runtime_local_scale: torch.Tensor,
    fixed_global: torch.Tensor,
    translation: torch.Tensor,
    prefix: tuple[int, ...],
    pose2rot: bool,
    apply_correctives: bool,
    return_joints_only: bool,
    repose_to_bind_pose: bool | None,
) -> Any:
    """Use SOMA-X's two-phase API for temporal data.

    Shape is constant over a motion clip. Preparing identity once per person and
    broadcasting the cached bind state avoids recomputing skeleton transfer for
    every frame. This is the same official SOMA-X mechanism used by the
    fixed-GS1 training runtime; LBS is also skipped for contact-only FK.
    """

    has_two_phase_api = hasattr(runtime, "prepare_identity") and hasattr(runtime, "pose")
    if len(prefix) == 2 and has_two_phase_api:
        batch_size, length = prefix
        identity_static = identity.mean(dim=1)
        local_scale_static = runtime_local_scale.mean(dim=1)
        global_static = fixed_global[:, 0, 0]
        repose = _resolve_repose_to_bind_pose(runtime, repose_to_bind_pose)
        runtime.prepare_identity(
            identity_static,
            local_scale_static,
            repose_to_bind_pose=repose,
            global_scale=global_static,
        )
        _broadcast_cached_identity_over_time(runtime, batch_size, length)
        flat_count = batch_size * length
        return runtime.pose(
            full_pose.reshape(flat_count, *full_pose.shape[2:]).contiguous(),
            transl=translation.reshape(flat_count, 3).contiguous(),
            pose2rot=pose2rot,
            apply_correctives=apply_correctives,
            fk_only=return_joints_only,
        )

    flat_count = math_product(prefix)
    flat_pose = full_pose.reshape(flat_count, *full_pose.shape[len(prefix) :]).contiguous()
    flat_identity = identity.reshape(flat_count, identity.shape[-1]).contiguous()
    flat_scale = runtime_local_scale.reshape(flat_count, runtime_local_scale.shape[-1]).contiguous()
    flat_global = fixed_global.reshape(flat_count, 1).contiguous()
    flat_translation = translation.reshape(flat_count, 3).contiguous()
    if has_two_phase_api and (return_joints_only or repose_to_bind_pose is not None):
        repose = _resolve_repose_to_bind_pose(runtime, repose_to_bind_pose)
        runtime.prepare_identity(
            flat_identity,
            flat_scale,
            repose_to_bind_pose=repose,
            global_scale=flat_global,
        )
        return runtime.pose(
            flat_pose,
            transl=flat_translation,
            pose2rot=pose2rot,
            apply_correctives=apply_correctives,
            fk_only=return_joints_only,
        )
    return runtime(
        poses=flat_pose,
        identity_coeffs=flat_identity,
        scale_params=flat_scale,
        transl=flat_translation,
        pose2rot=pose2rot,
        global_scale=flat_global,
        apply_correctives=apply_correctives,
    )


def _resolve_repose_to_bind_pose(runtime: nn.Module, override: bool | None) -> bool:
    if override is not None:
        return override
    return getattr(runtime, "procedural_transforms", None) is not None


def _broadcast_cached_identity_over_time(runtime: nn.Module, batch_size: int, length: int) -> None:
    for name in ("_cached_rest_shape", "_cached_bind_transforms_world", "_cached_scale_params"):
        value = getattr(runtime, name, None)
        if value is None:
            continue
        if value.shape[0] != batch_size:
            raise ValueError(
                f"SOMA runtime {name} cache has batch {value.shape[0]}, expected {batch_size}"
            )
        expanded = value[:, None].expand(batch_size, length, *value.shape[1:])
        setattr(runtime, name, expanded.reshape(batch_size * length, *value.shape[1:]).contiguous())


def _join_public_pose(
    global_orient: torch.Tensor,
    body_pose: torch.Tensor,
    *,
    pose2rot: bool,
) -> tuple[torch.Tensor, tuple[int, ...]]:
    root = _float32_tensor(global_orient, name="global_orient")
    body = _float32_tensor(body_pose, name="body_pose")

    if pose2rot:
        if body.shape[-1:] == (SOMA_BODY_POSE_JOINTS * 3,):
            prefix = tuple(body.shape[:-1])
            body = body.reshape(*prefix, SOMA_BODY_POSE_JOINTS, 3)
        elif body.shape[-2:] == (SOMA_BODY_POSE_JOINTS, 3):
            prefix = tuple(body.shape[:-2])
        else:
            raise ValueError("axis-angle body_pose must end in (228,) or (76,3)")
        if root.shape == (*prefix, 3):
            root = root.unsqueeze(-2)
        elif root.shape != (*prefix, 1, 3):
            raise ValueError(
                "axis-angle global_orient must match the body prefix and end in (3,) or (1,3)"
            )
    else:
        if body.shape[-3:] != (SOMA_BODY_POSE_JOINTS, 3, 3):
            raise ValueError("rotation-matrix body_pose must end in (76,3,3)")
        prefix = tuple(body.shape[:-3])
        if root.shape == (*prefix, 3, 3):
            root = root.unsqueeze(-3)
        elif root.shape != (*prefix, 1, 3, 3):
            raise ValueError(
                "rotation-matrix global_orient must match the body prefix and end in (3,3) or (1,3,3)"
            )

    if not prefix:
        raise ValueError("SOMA adapter inputs must include a batch dimension")
    full_pose = torch.cat((root, body), dim=-(3 if not pose2rot else 2)).contiguous()
    expected_tail = (
        (SOMA_PUBLIC_POSE_JOINTS, 3, 3) if not pose2rot else (SOMA_PUBLIC_POSE_JOINTS, 3)
    )
    if full_pose.shape[-len(expected_tail) :] != expected_tail:
        raise AssertionError("failed to construct the 77-joint public SOMA pose")
    return full_pose, prefix


def _expand_feature_tensor(
    value: torch.Tensor,
    *,
    prefix: tuple[int, ...],
    width: int,
    name: str,
) -> torch.Tensor:
    tensor = _float32_tensor(value, name=name)
    _require_last_dim(tensor, width, name=name)
    value_prefix = tuple(tensor.shape[:-1])
    if value_prefix == prefix:
        return tensor
    if len(prefix) >= 2 and value_prefix == prefix[:-1]:
        return tensor.unsqueeze(-2).expand(*prefix, width)
    raise ValueError(f"{name} prefix must be {prefix} or {prefix[:-1]}, got {value_prefix}")


def _normalize_runtime_output(
    output: Any,
    *,
    prefix: tuple[int, ...],
    device: torch.device,
) -> dict[str, torch.Tensor | None]:
    result: dict[str, torch.Tensor | None] = {}
    for name in ("vertices", "joints", "transforms"):
        value = output.get(name) if isinstance(output, Mapping) else getattr(output, name, None)
        if value is None:
            result[name] = None
            continue
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"SOMA runtime output {name} must be a torch.Tensor or None")
        if value.dtype != torch.float32 or value.device != device:
            raise ValueError(f"SOMA runtime output {name} must be float32 on {device}")
        if value.shape[0] != math_product(prefix):
            raise ValueError(f"SOMA runtime output {name} has an unexpected batch dimension")
        result[name] = value.reshape(*prefix, *value.shape[1:])

    joints = result["joints"]
    if not isinstance(joints, torch.Tensor) or joints.shape[-2:] != (SOMA_PUBLIC_POSE_JOINTS, 3):
        raise ValueError("SOMA runtime must return public joints with shape (N,77,3)")
    return result


def _float32_tensor(value: torch.Tensor, *, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    return value.to(dtype=torch.float32)


def _require_last_dim(value: torch.Tensor, width: int, *, name: str) -> None:
    if value.ndim == 0 or value.shape[-1] != width:
        raise ValueError(f"{name} must have final dimension {width}, got {tuple(value.shape)}")


def math_product(values: tuple[int, ...]) -> int:
    product = 1
    for value in values:
        product *= value
    return product


__all__ = [
    "SOMA_APPENDED_SCALE_NAMES",
    "SOMA_BODY_POSE_JOINTS",
    "SOMA_EXTERNAL_SCALE_DIM",
    "SOMA_IDENTITY_DIM",
    "SOMA_LEARNED_LOCAL_SCALE_DIM",
    "SOMA_PUBLIC_POSE_JOINTS",
    "SOMA_RUNTIME_LOCAL_SCALE_DIM",
    "SomaFixedGs1BodyAdapter",
    "SomaFixedGs1ShapeSpec",
    "SomaRuntimeConfig",
]
