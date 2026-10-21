# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Train or validate body/hand Pose with an explicit model and data configuration."""

import argparse
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("body", "hand"), required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", type=Path, help="Complete native Pose training checkpoint")
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args(argv)

    import lightning.pytorch as pl
    from hydra.utils import instantiate
    from lightning.pytorch.callbacks import ModelCheckpoint
    from omegaconf import DictConfig, OmegaConf

    from gem.common.training import trainer_options

    from .module import PoseTrainingModule
    from .recovery import PoseTrainingState

    config = OmegaConf.load(args.config)
    pl.seed_everything(config.get("seed", 42), workers=True)
    if args.resume is not None:
        # A full checkpoint already owns model state; an old initialization
        # file is not a dependency of resuming or validating that checkpoint.
        for name in ("initialization", "backbone_initialization"):
            if name in config.model:
                config.model[name] = None
    model = instantiate(config.model)
    if not isinstance(model, PoseTrainingModule):
        raise TypeError("Pose training config must construct a PoseTrainingModule")
    if model.release.task != args.task:
        raise ValueError(
            f"Requested {args.task} training but config constructs {model.release.task}"
        )
    data = instantiate(config.data)
    configured = config.get("callbacks", [])
    specs = configured.values() if isinstance(configured, DictConfig) else configured
    callbacks = [instantiate(spec) for spec in specs]
    if not args.validate:
        callbacks.insert(0, PoseTrainingState())
        callbacks.append(ModelCheckpoint(**OmegaConf.to_container(config.checkpoint, resolve=True)))
    options = trainer_options(config.trainer)
    options.update(
        callbacks=callbacks, enable_checkpointing=not args.validate, use_distributed_sampler=False
    )
    trainer = pl.Trainer(**options)
    if args.validate:
        trainer.validate(model, datamodule=data, ckpt_path=args.resume, weights_only=False)
    else:
        trainer.fit(model, datamodule=data, ckpt_path=args.resume, weights_only=False)


if __name__ == "__main__":
    main()
