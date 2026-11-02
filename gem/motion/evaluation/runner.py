# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Standalone full-sequence Motion evaluation from local benchmark caches."""

import json
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from gem.common.soma import SomaFixedGs1BodyAdapter, SomaRuntimeConfig
from gem.inference.motion import load_predictor
from gem.inference.serialization import tensor_tree
from gem.inference.timing import prediction_fps
from gem.motion.postprocess.api import SomaContactPostprocessor

from .evaluator import MotionEvaluator
from .geometry import SomaJointRegressor, SomaMeshProjector

_PARAMETERS = ("pred_soma_params_incam", "pred_soma_params_global")


def load_config(path):
    """Resolve environment settings and model/asset paths relative to the YAML."""
    path = Path(path).resolve()
    config = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    for section, names in (
        ("pose", ("checkpoint",)),
        ("observations", ("checkpoint",)),
        ("motion", ("checkpoint", "stats")),
        ("evaluation", ("soma_assets", "mesh_projector")),
    ):
        for name in names:
            value = config.get(section, {}).get(name)
            if value is not None:
                config[section][name] = str((path.parent / value).resolve())
    evaluation = config.get("evaluation", {})
    evaluation["joint_projectors"] = {
        name: str((path.parent / value).resolve())
        for name, value in evaluation.get("joint_projectors", {}).items()
    }
    return config


@contextmanager
def _ieee_precision():
    matmul = torch.backends.cuda.matmul.allow_tf32
    cudnn = torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = matmul
        torch.backends.cudnn.allow_tf32 = cudnn


def summarize(records):
    """Pool metric samples, including individual foot contacts, across sequences."""
    metrics, counts = {}, {}
    for dataset, sequences in records.items():
        fields = defaultdict(list)
        for values in sequences.values():
            for name, array in values.items():
                fields[name].append(np.asarray(array).reshape(-1))
        metrics[dataset], counts[dataset] = {}, {}
        for name, parts in fields.items():
            values = np.concatenate(parts)
            if not np.isfinite(values).all():
                raise ValueError(f"Nonfinite Motion metric {dataset}/{name}")
            counts[dataset][name] = len(values)
            metrics[dataset][name] = float(values.mean(dtype=np.float64)) if len(values) else None
    return metrics, counts


def evaluate_index(
    predictor,
    evaluator,
    data_index,
    *,
    device="cpu",
    postprocessor=None,
    postprocess="both",
    sequences=None,
):
    """Evaluate each selected complete sequence once, keeping raw and PP separate."""
    if postprocess not in {"raw", "pp", "both"}:
        raise ValueError("postprocess must be raw, pp or both")
    if postprocess != "raw" and postprocessor is None:
        raise ValueError("Postprocessed evaluation requires a Contact/IK postprocessor")
    index_path = Path(data_index)
    index = json.loads(index_path.read_text())
    if index["format"] != "gem_motion_benchmark_index_v1":
        raise ValueError("Use the index produced by gem.motion.evaluation.prepare")
    requested = None if sequences is None else set(sequences)
    found, seen = set(), set()
    modes = ("raw", "pp") if postprocess == "both" else (postprocess,)
    pooled = {mode: defaultdict(dict) for mode in modes}
    rows = []
    for entry in index["sequences"]:
        dataset, name = entry["dataset_id"], entry["sequence_id"]
        qualified = f"{dataset}/{name}"
        if requested is not None and not ({name, qualified} & requested):
            continue
        found.update({name, qualified})
        if qualified in seen:
            continue
        seen.add(qualified)
        record = torch.load(
            index_path.parent / entry["path"], map_location="cpu", weights_only=True
        )
        if record["dataset_id"] != dataset or record["sequence_id"] != name:
            raise ValueError("Sequence cache identity differs from its index")
        conditions = record["conditions"]
        if "f_imgseq" not in conditions or not ({"obs", "kp2d_native77"} & conditions.keys()):
            raise ValueError("Extract benchmark features before evaluating: use --stage features")
        length = int(torch.as_tensor(conditions["length"]).item())
        mask = torch.as_tensor(record["mask"], dtype=torch.bool)
        if mask.shape != (length,) or not bool(mask.any()):
            raise ValueError("Benchmark mask must select frames on the complete sequence")
        # All retained temporal definitions and Contact/IK use this physical clock.
        if abs(prediction_fps({"conditions": conditions}) - 30.0) > 1e-4:
            raise ValueError("Motion benchmarks require a continuous, uniform 30 Hz timeline")
        batch = {
            "B": 1,
            "meta": [{"dataset_id": dataset, "vid": name}],
            "mask": mask[None],
            "eval_targets": tensor_tree(record["eval_targets"], device=device),
        }
        if evaluator.vertex_projector is not None:
            field = "verts_global" if dataset == "EMDB_2" else "verts_incam"
            if dataset.startswith("EMDB") and field not in batch["eval_targets"]:
                raise ValueError(
                    "Mesh metrics require preparation with --include-mesh, or mesh_projector: null"
                )
        with (
            torch.inference_mode(),
            _ieee_precision(),
            torch.autocast(torch.device(device).type, enabled=False),
        ):
            output = predictor.run(tensor_tree(conditions, device=device))
            raw = {key: output[key] for key in (*_PARAMETERS, "static_conf_logits")}
            del output
            summary = {}
            for mode in modes:
                prediction = (
                    raw
                    if mode == "raw"
                    else postprocessor(
                        raw[_PARAMETERS[0]], raw[_PARAMETERS[1]], raw["static_conf_logits"]
                    )
                )
                unbatched = {
                    key: {field: value[0].to(device) for field, value in prediction[key].items()}
                    for key in _PARAMETERS
                }
                values = evaluator.evaluate(unbatched, batch)["metrics"]
                pooled[mode][dataset][name] = values
                means, counts = summarize({dataset: {name: values}})
                summary[mode] = {"metrics": means[dataset], "metric_sample_counts": counts[dataset]}
        rows.append(
            {
                "dataset_id": dataset,
                "sequence_id": name,
                "input_frames": length,
                "valid_frames": int(mask.sum()),
                "input_protocol": record.get("protocol", {}),
                **summary,
            }
        )
        print(f"Evaluated {qualified}: {length} input / {int(mask.sum())} valid frames", flush=True)
    if requested is not None and requested - found:
        raise ValueError(f"Requested sequences not found: {sorted(requested - found)}")
    if not rows:
        raise ValueError("The benchmark index contains no selected sequences")
    results = {}
    for mode, values in pooled.items():
        metrics, counts = summarize(values)
        results[mode] = {"metrics": metrics, "metric_sample_counts": counts}
    return {
        "format": "gem_motion_evaluation_v1",
        "precision": "IEEE FP32",
        "postprocess": postprocess,
        "sequence_count": len(rows),
        "input_frames": sum(row["input_frames"] for row in rows),
        "valid_frames": sum(row["valid_frames"] for row in rows),
        "sequences": rows,
        "results": results,
    }


def run_evaluation(config, data_index, *, device="cpu", postprocess="both", sequences=None):
    model = config["motion"]
    predictor = load_predictor(
        model["checkpoint"], model, stats_path=model.get("stats"), device=device
    )
    options = config["evaluation"]
    body = SomaFixedGs1BodyAdapter(
        SomaRuntimeConfig(
            data_root=options["soma_assets"],
            lod="mid",
            device=device,
            apply_correctives=False,
        )
    ).to(device)
    mesh = options.get("mesh_projector")
    evaluator = MotionEvaluator(
        body,
        {
            key: SomaJointRegressor(value, key).to(device)
            for key, value in options["joint_projectors"].items()
        },
        vertex_projector=(
            None if mesh is None else SomaMeshProjector(mesh, source_num_vertices=18056).to(device)
        ),
        evaluate_camera_root=True,
    )
    processor = (
        None
        if postprocess == "raw"
        else SomaContactPostprocessor(options["soma_assets"], device=device)
    )
    result = evaluate_index(
        predictor,
        evaluator,
        data_index,
        device=device,
        postprocessor=processor,
        postprocess=postprocess,
        sequences=sequences,
    )
    result.update(checkpoint=model["checkpoint"], condition_spec=model["condition_spec"])
    return result
