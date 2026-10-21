# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

ARG BASE_IMAGE=python:3.12-slim
FROM ${BASE_IMAGE}

# Let PyTorch's explicit matmul policy control precision, including the IEEE
# geometry/IK scopes. Some NVIDIA bases otherwise force TF32 regardless of it.
ENV TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0

# A PyTorch runtime image can supply the architecture-specific CUDA build.
# Use an empty GEM_EXTRAS value for base inference only.
ARG GEM_EXTRAS=train,video
ARG DEBIAN_FRONTEND=noninteractive

# GPU execution uses the host driver through NVIDIA Container Toolkit.
# Video decoding/normalization needs ffmpeg; optional Open3D rendering also
# needs Mesa EGL for the supported Open3D0.19 renderer.
RUN set -eu; \
    render_packages=""; \
    case ",${GEM_EXTRAS}," in \
        *,render,*) render_packages="libegl1 libegl-mesa0 libgl1 libgl1-mesa-dri" ;; \
    esac; \
    apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libgomp1 ffmpeg ${render_packages} \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/gem
COPY pyproject.toml setup.py setup.cfg README.md LICENSE ./
COPY gem ./gem

# SOMA is installed from PyPI. Checkpoints and SOMA assets are supplied at run
# time; building the image does not fetch model assets or require submodules.
# Normal pip installation preserves a compatible Torch already in BASE_IMAGE.
# ARM64 requires a matching usd-core wheel installed in that base or available
# through its pip index; an importable pxr tree alone is not a distribution.
RUN set -eu; \
    package="."; \
    if [ -n "${GEM_EXTRAS}" ]; then package=".[${GEM_EXTRAS}]"; fi; \
    python -m pip install --no-cache-dir "${package}"

CMD ["gem-infer", "--help"]
