# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Installation checks, including an isolated wheel smoke without model assets.

After installing the wheel into a clean environment, run that environment's
Python with ``-I tests/test_installation.py --output result.json``. Add
``--training``/``--evaluation`` after installing the corresponding extras.
Optional checkpoint arguments exercise real legacy metadata deserialization
with mmap; they do not construct a full ViT-H model.
"""

import argparse
import configparser
import importlib.metadata
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def test_package_discovery_is_public_only():
    from setuptools import find_packages

    root = Path(__file__).resolve().parents[1]
    config = configparser.ConfigParser()
    config.read(root / "setup.cfg")
    included = config["options.packages.find"]["include"].split()
    packages = find_packages(root, include=included)
    assert {"gem", "gem.common", "gem.pose", "gem.motion"} <= set(packages)
    assert all(
        name == "gem" or name.startswith(("gem.common", "gem.pose", "gem.motion", "gem.inference"))
        for name in packages
    )


def installed_smoke(*, training=False, evaluation=False, checkpoints=()):
    import numpy as np
    import soma
    import torch
    from transformers import DINOv3ViTConfig

    import gem
    from gem.common.backbones import DinoV3Backbone
    from gem.motion.api import MotionPredictor
    from gem.pose.api import body_config, hand_config
    from gem.pose.data.crop import crop_rgb

    assert sys.flags.isolated, "Run with python -I to exclude checkout/PYTHONPATH imports"
    assert Path(gem.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
    assert importlib.metadata.version("py-soma-x") == soma.__version__ == "0.3.3"
    assert importlib.metadata.version("transformers") == "5.3.0"
    assert MotionPredictor is not None and crop_rgb is not None
    assert body_config().task == "body" and hand_config().task == "hand"
    torch.set_num_threads(2)
    config = DINOv3ViTConfig(
        image_size=32,
        patch_size=16,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=64,
        num_register_tokens=2,
        use_gated_mlp=True,
        hidden_act="silu",
    )
    model = DinoV3Backbone(config=config).eval()
    with torch.no_grad():
        feature = model(torch.zeros(1, 3, 32, 32))
    assert feature.shape == (1, 32, 2, 2) and bool(feature.isfinite().all())
    np.testing.assert_array_equal(torch.from_numpy(np.zeros(2)).numpy(), np.zeros(2))

    distribution = importlib.metadata.distribution("gem")
    python_files = [str(path) for path in distribution.files if str(path).endswith(".py")]
    assert python_files
    assert all(
        path == "gem/__init__.py"
        or path.startswith(("gem/common/", "gem/pose/", "gem/motion/", "gem/inference/"))
        for path in python_files
    ), python_files
    assert {point.name for point in distribution.entry_points} == {
        "gem-infer",
        "gem-train",
        "gem-render",
        "gem-eval",
    }

    commands = ["gem-infer", "gem-render"]
    if training or evaluation:
        from gem.motion.data import MixedImageVideoDataModule
        from gem.motion.evaluation import MotionEvaluationCallback
        from gem.motion.model import MotionModel

        assert all((MixedImageVideoDataModule, MotionEvaluationCallback, MotionModel))
        commands.append("gem-train")
    else:
        assert importlib.util.find_spec("lightning") is None
    if evaluation:
        from gem.pose.evaluation import HandEvaluator

        assert callable(HandEvaluator)
        commands.append("gem-eval")
    environment = {name: value for name, value in os.environ.items() if name != "PYTHONPATH"}
    with tempfile.TemporaryDirectory(prefix="gem-wheel-smoke-") as directory:
        for name in commands:
            command = Path(sys.executable).parent / name
            result = subprocess.run(
                [str(command), "--help"],
                cwd=directory,
                env=environment,
                capture_output=True,
                text=True,
            )
            assert result.returncode == 0, result.stderr
            assert "usage:" in result.stdout.lower(), result.stdout

    checkpoint_results = []
    for path in checkpoints:
        checkpoint = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
        checkpoint_results.append(
            {"file": Path(path).name, "state_tensors": len(checkpoint["state_dict"])}
        )
        del checkpoint
    assert not any(name.split(".")[0] == "hmr4d" for name in sys.modules)
    return {
        "status": "passed",
        "python": sys.version.split()[0],
        "gem_import": gem.__file__,
        "soma_import": soma.__file__,
        "torch": torch.__version__,
        "soma": soma.__version__,
        "transformers": importlib.metadata.version("transformers"),
        "numpy": np.__version__,
        "training_extra": training,
        "evaluation_extra": evaluation,
        "installed_python_files": len(python_files),
        "console_commands": commands,
        "tiny_dinov3_feature_shape": list(feature.shape),
        "legacy_checkpoints": checkpoint_results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training", action="store_true")
    parser.add_argument("--evaluation", action="store_true")
    parser.add_argument("--checkpoint", action="append", type=Path, default=[])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = installed_smoke(
        training=args.training, evaluation=args.evaluation, checkpoints=args.checkpoint
    )
    payload = json.dumps(result, indent=2) + "\n"
    if args.output is not None:
        args.output.write_text(payload)
    print(payload, end="")


if __name__ == "__main__":
    main()
