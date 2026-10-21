# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The shared training command keeps legacy Motion routing and lazy help."""

import subprocess
import sys
from types import SimpleNamespace

import pytest

from gem.common.training import main


@pytest.mark.parametrize("prefix", [[], ["pose"], ["motion"]])
def test_training_help_does_not_load_model_or_training_libraries(prefix):
    code = f"""
import builtins
original = builtins.__import__
def lightweight(name, *args, **kwargs):
    if name.split('.')[0] in {{'torch','transformers','soma','cv2','lightning','hydra'}}:
        raise AssertionError('Eager runtime import: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = lightweight
from gem.common.training import main
main({prefix + ['--help']!r})
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout


@pytest.mark.parametrize(
    "arguments,target,expected",
    [
        (["--config", "motion.yaml"], "motion", ["--config", "motion.yaml"]),
        (["motion", "--config", "motion.yaml"], "motion", ["--config", "motion.yaml"]),
        (
            ["pose", "--task", "body", "--config", "pose.yaml"],
            "pose",
            ["--task", "body", "--config", "pose.yaml"],
        ),
        (
            ["pose", "--task", "hand", "--config", "hand.yaml"],
            "pose",
            ["--task", "hand", "--config", "hand.yaml"],
        ),
    ],
)
def test_dispatch_preserves_model_specific_arguments(monkeypatch, arguments, target, expected):
    seen = []
    monkeypatch.setitem(
        sys.modules,
        f"gem.{target}.training.cli",
        SimpleNamespace(main=lambda args: seen.append(args)),
    )
    main(arguments)
    assert seen == [expected]


@pytest.mark.parametrize("logger", [True, False, None, "absent"])
def test_trainer_logger_defaults_and_switches_remain_lightning_values(logger):
    from omegaconf import OmegaConf

    from gem.common.training import trainer_options

    config = {} if logger == "absent" else {"logger": logger}
    assert trainer_options(OmegaConf.create(config)) == config


def test_configured_csv_logger_writes_real_metrics(tmp_path):
    from omegaconf import OmegaConf

    from gem.common.training import trainer_options

    options = trainer_options(
        OmegaConf.create(
            {
                "logger": {
                    "_target_": "lightning.pytorch.loggers.CSVLogger",
                    "save_dir": str(tmp_path),
                    "name": "training",
                    "version": "run",
                }
            }
        )
    )
    logger = options["logger"]
    logger.log_metrics({"loss": 1.25}, step=7)
    logger.save()
    assert "1.25,7" in (tmp_path / "training/run/metrics.csv").read_text()
