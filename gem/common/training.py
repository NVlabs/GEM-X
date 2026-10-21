# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dispatch training commands while retaining the original Motion CLI form."""

import argparse
import sys


def trainer_options(config):
    """Resolve Trainer options, constructing an explicitly configured logger."""
    from hydra.utils import instantiate
    from omegaconf import DictConfig, ListConfig, OmegaConf

    options = OmegaConf.to_container(config, resolve=True)
    if isinstance(config.get("logger"), DictConfig | ListConfig):
        options["logger"] = instantiate(config.logger)
    return options


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "pose":
        from gem.pose.training.cli import main as train_pose

        return train_pose(argv[1:])
    if argv and argv[0] in {"-h", "--help"}:
        parser = argparse.ArgumentParser(
            description="GEM training: pose --task body|hand, or motion. Legacy --config selects Motion."
        )
        parser.add_argument("model", choices=("pose", "motion"), nargs="?")
        parser.epilog = "Use gem-train pose --help or gem-train motion --help for model options."
        parser.print_help()
        return
    from gem.motion.training.cli import main as train_motion

    return train_motion(argv[1:] if argv and argv[0] == "motion" else argv)


if __name__ == "__main__":
    main()
