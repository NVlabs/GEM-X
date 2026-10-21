# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint training randomness and restore it after loader initialization.

For distributed training use this callback's scheduled saves or call
``save_checkpoint(trainer, path)`` on every rank. Collectives occur only there,
never in ``on_save_checkpoint``; a rank-zero-only save cannot deadlock here.
Disable Lightning's default ModelCheckpoint when using the scheduled saves.
"""

import random
from pathlib import Path

import lightning.pytorch as pl
import numpy as np
import torch
import torch.distributed as dist

RNG_CHECKPOINT_KEY = "gem_training_rng"


def capture_rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        # One device per distributed training process; avoid initializing
        # contexts on other ranks' GPUs just to serialize their unused RNGs.
        "cuda": torch.cuda.get_rng_state() if torch.cuda.is_initialized() else None,
    }


def restore_rng_state(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] is not None:
        torch.cuda.set_rng_state(state["cuda"].cpu())


class TrainingRNGState(pl.Callback):
    """Per-rank RNG recovery plus optional complete Lightning checkpoints.

    Model/optimizer/scheduler/precision and loop states are saved by Lightning.
    The DataModule owns the consumed-attempt cursor, not the prefetched index.
    On resume this callback restores model RNG at the first batch-start hook,
    after the loader iterator and worker seeds have been created.
    """

    def __init__(self, dirpath=None, *, every_n_train_steps=None, every_n_epochs=1):
        super().__init__()
        if every_n_train_steps is not None and every_n_train_steps < 1:
            raise ValueError("every_n_train_steps must be positive")
        if every_n_epochs is not None and every_n_epochs < 1:
            raise ValueError("every_n_epochs must be positive")
        self.dirpath = Path(dirpath) if dirpath is not None else None
        self.every_n_train_steps = every_n_train_steps
        self.every_n_epochs = every_n_epochs
        self._prepared_states = None
        self._pending_state = None
        self._last_saved_attempt = None

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        if self._prepared_states is not None:
            checkpoint[RNG_CHECKPOINT_KEY] = self._prepared_states
        elif trainer.world_size == 1:
            checkpoint[RNG_CHECKPOINT_KEY] = [capture_rng_state()]
        else:
            raise RuntimeError(
                "Distributed RNG checkpoints require TrainingRNGState.save_checkpoint on every rank"
            )

    def on_load_checkpoint(self, trainer, pl_module, checkpoint):
        states = checkpoint[RNG_CHECKPOINT_KEY]
        if len(states) != trainer.world_size:
            raise ValueError("Resuming per-rank RNG requires the same distributed world size")
        self._pending_state = states[trainer.global_rank]

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if self._pending_state is not None:
            restore_rng_state(self._pending_state)
            self._pending_state = None

    def save_checkpoint(self, trainer, path):
        """Save full training state; distributed callers enter on every rank."""
        local = capture_rng_state()
        if dist.is_available() and dist.is_initialized():
            states = [None] * dist.get_world_size()
            dist.all_gather_object(states, local)
        else:
            states = [local]
        self._prepared_states = states
        try:
            trainer.save_checkpoint(str(path), weights_only=False)
        finally:
            self._prepared_states = None

    def _save_scheduled(self, trainer, pl_module):
        if self.dirpath is None:
            return
        attempt = int(pl_module.attempts_completed)
        if attempt == self._last_saved_attempt:
            return
        self.save_checkpoint(trainer, self.dirpath / f"attempt-{attempt:09d}.ckpt")
        self._last_saved_attempt = attempt

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if (
            self.every_n_train_steps is not None
            and pl_module.attempts_completed % self.every_n_train_steps == 0
        ):
            self._save_scheduled(trainer, pl_module)

    def on_train_epoch_end(self, trainer, pl_module):
        if (
            self.every_n_epochs is not None
            and (trainer.current_epoch + 1) % self.every_n_epochs == 0
        ):
            self._save_scheduled(trainer, pl_module)
