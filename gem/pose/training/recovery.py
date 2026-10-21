# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint model randomness after consumed Pose batches, including DDP ranks."""

import random

import lightning.pytorch as pl
import numpy as np
import torch
import torch.distributed as dist


def _capture_rng():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state() if torch.cuda.is_initialized() else None,
    }


def _restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] is not None:
        torch.cuda.set_rng_state(state["cuda"].cpu())


class PoseTrainingState(pl.Callback):
    """Run before ModelCheckpoint so its state includes the completed batch.

    Stateful data providers expose ``mark_batch_consumed`` and own their loader
    state_dict/load_state_dict. Other Lightning data providers retain their own
    recovery policy. Runtime initialization and loader startup happen before the
    saved per-rank model RNG is restored at the next training-batch boundary.
    """

    def __init__(self):
        self._pending = None

    def on_train_epoch_start(self, trainer, pl_module):
        data = trainer.datamodule
        if hasattr(data, "set_epoch"):
            data.set_epoch(trainer.current_epoch)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        data = trainer.datamodule
        if hasattr(data, "mark_batch_consumed"):
            data.mark_batch_consumed(batch)

    def state_dict(self):
        state = _capture_rng()
        if dist.is_available() and dist.is_initialized():
            states = [None] * dist.get_world_size()
            dist.all_gather_object(states, state)
        else:
            states = [state]
        return {"rng_by_rank": states}

    def load_state_dict(self, state_dict):
        self._pending = state_dict["rng_by_rank"]

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if self._pending is not None:
            if len(self._pending) != trainer.world_size:
                raise ValueError("Pose training resume requires the saved rank count")
            _restore_rng(self._pending[trainer.global_rank])
            self._pending = None
