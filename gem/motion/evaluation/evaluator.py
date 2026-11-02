# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One full-sequence evaluator for supplied raw or postprocessed predictions."""

from dataclasses import replace

import torch
from torch import nn

from .mesh_metrics import compute_soma_camcoord_metrics, compute_soma_global_metrics
from .metrics import compute_soma_camcoord_joint_metrics, compute_soma_global_joint_metrics

DATASET_JOINT_SETS = {"EMDB_1": "smpl24", "EMDB_2": "smpl24", "3DPW": "3dpw_j14"}


class MotionEvaluator(nn.Module):
    """Evaluate a batch of one complete sequence using caller-supplied GT.

    ``target_builder(batch, device)`` returns ``incam`` joints in metres and,
    for EMDB2, ``global`` joints. Each is (T,J,3) or (1,T,J,3). The default
    consumes the same mapping in ``batch['eval_targets']``. A caller-provided target builder can construct this mapping; the evaluator
    has no SMPL runtime dependency.

    An optional ``vertex_projector`` enables EMDB1 PVE/acceleration and EMDB2
    foot sliding. Targets then also supply ``verts_incam`` (EMDB1) or
    ``verts_global`` (EMDB2), each with corresponding SMPL6890 topology.
    ``evaluate_camera_root`` adds unaligned EMDB1 joint-zero translation error
    in mm; it does not replace pelvis-aligned pose metrics.

    Output parameter fields are unbatched temporal tensors, as returned by
    the original validation step. FK receives their restored B=1 axis so
    sequence-static identity/scale averaging stays unchanged. Geometry uses
    the explicitly injected eval LOD (the formal recipe uses MID, no correctives).
    """

    def __init__(
        self,
        body_adapter,
        joint_regressors,
        *,
        target_builder=None,
        vertex_projector=None,
        evaluate_camera_root=False,
    ):
        super().__init__()
        self.body_adapter = body_adapter
        self.joint_regressors = nn.ModuleDict(joint_regressors)
        self.target_builder = target_builder
        self.vertex_projector = vertex_projector
        self.evaluate_camera_root = evaluate_camera_root
        self._device = None

    def _vertices(self, parameters):
        return self.body_adapter(
            **{name: value[None] for name, value in parameters.items()},
            return_joints_only=False,
        )["vertices"][0]

    @torch.no_grad()
    def evaluate(self, outputs, batch):
        if batch["B"] != 1:
            raise ValueError("Motion evaluation requires one complete sequence per batch")
        meta = batch["meta"][0]
        dataset_id = meta["dataset_id"]
        joint_set = DATASET_JOINT_SETS[dataset_id]
        parameters = outputs["pred_soma_params_incam"]
        device = parameters["global_orient"].device
        # The layer is caller-owned and lazy: initialize it on the actual
        # validation rank's device, rather than a saved constructor device.
        if self._device != device:
            if not self.body_adapter.runtime_is_initialized:
                self.body_adapter.runtime_config = replace(
                    self.body_adapter.runtime_config, device=device
                )
            self.body_adapter.to(device)
            self.joint_regressors.to(device)
            if self.vertex_projector is not None:
                self.vertex_projector.to(device)
            self._device = device
        regressor = self.joint_regressors[joint_set]
        vertices = self._vertices(parameters)
        prediction = regressor(vertices)
        targets = (
            batch["eval_targets"]
            if self.target_builder is None
            else self.target_builder(batch, device)
        )
        target_incam = targets["incam"]
        if target_incam.ndim == 4:
            target_incam = target_incam[0]
        mask = batch["mask"][0].bool()
        if mask.shape != (len(prediction),) or not bool(mask.any()):
            raise ValueError("Evaluation mask must select frames from the complete sequence")
        camera_batch = {"pred_j3d": prediction, "target_j3d": target_incam}
        camera_metric = compute_soma_camcoord_joint_metrics
        if self.vertex_projector is not None and dataset_id == "EMDB_1":
            target_vertices = targets["verts_incam"]
            if target_vertices.ndim == 4:
                target_vertices = target_vertices[0]
            camera_batch.update(
                pred_verts=self.vertex_projector(vertices), target_verts=target_vertices
            )
            camera_metric = compute_soma_camcoord_metrics
        metrics = camera_metric(
            camera_batch,
            target_joint_set=joint_set,
            mask=mask,
        )
        if self.evaluate_camera_root and dataset_id == "EMDB_1":
            root_difference = prediction[:, 0].double().cpu() - target_incam[:, 0].double().cpu()
            metrics["camera_root_translation_error_mm"] = (
                root_difference[mask.cpu()].norm(2, dim=-1).numpy() * 1000
            )
        del vertices
        if dataset_id == "EMDB_2":
            global_vertices = self._vertices(outputs["pred_soma_params_global"])
            predicted_global = regressor(global_vertices)
            target_global = targets["global"]
            if target_global.ndim == 4:
                target_global = target_global[0]
            global_batch = {"pred_j3d_glob": predicted_global, "target_j3d_glob": target_global}
            global_metric = compute_soma_global_joint_metrics
            if self.vertex_projector is not None:
                target_vertices = targets["verts_global"]
                if target_vertices.ndim == 4:
                    target_vertices = target_vertices[0]
                global_batch.update(
                    pred_verts_glob=self.vertex_projector(global_vertices),
                    target_verts_glob=target_vertices,
                )
                global_metric = compute_soma_global_metrics
            metrics.update(
                global_metric(
                    global_batch,
                    target_joint_set=joint_set,
                    mask=mask,
                )
            )
        return {"dataset_id": dataset_id, "vid": str(meta["vid"]), "metrics": metrics}
