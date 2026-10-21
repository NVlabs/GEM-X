# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Train native GEM-Motion with caller-configured geometry and datasets."""

import argparse
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="YAML with model, data, trainer, seed and checkpoint settings",
    )
    parser.add_argument("--resume", type=Path, help="Complete checkpoint from this native trainer")
    parser.add_argument(
        "--validate", action="store_true", help="Run configured validation without training"
    )
    args = parser.parse_args(argv)
    import lightning.pytorch as pl
    from hydra.utils import instantiate
    from omegaconf import DictConfig, OmegaConf

    from gem.common.training import trainer_options

    from .recovery import TrainingRNGState

    config = OmegaConf.load(args.config)
    pl.seed_everything(config.get("seed", 42), workers=True)
    model = instantiate(config.model)
    data = instantiate(config.data)
    configured = config.get("callbacks", [])
    specs = configured.values() if isinstance(configured, DictConfig) else configured
    callbacks = [instantiate(spec) for spec in specs]
    if not args.validate:
        checkpoint_options = OmegaConf.to_container(config.checkpoint, resolve=True)
        callbacks.insert(0, TrainingRNGState(**checkpoint_options))
    options = trainer_options(config.trainer)
    # The native data sampler and recovery callback own these two mechanisms.
    options.update(callbacks=callbacks, enable_checkpointing=False, use_distributed_sampler=False)
    trainer = pl.Trainer(**options)
    if args.validate:
        trainer.validate(model, datamodule=data, ckpt_path=args.resume, weights_only=False)
    else:
        trainer.fit(model, datamodule=data, ckpt_path=args.resume, weights_only=False)


if __name__ == "__main__":
    main()
