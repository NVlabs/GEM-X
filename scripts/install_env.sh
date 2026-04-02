#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status.

echo "Installing gem in editable mode..."
uv pip install -e .

echo "Installing SAM-3D-Body runtime deps..."
uv pip install cloudpickle fvcore iopath pycocotools braceexpand roma 'setuptools<75'

if [[ "$(uname)" == "Darwin" ]]; then
    echo "macOS detected — skipping detectron2, installing ONNX Runtime..."
    uv pip install onnxruntime
else
    echo "Installing detectron2..."
    uv pip install 'git+https://github.com/facebookresearch/detectron2.git@a1ce2f9' --no-build-isolation --no-deps
fi

echo "Environment setup complete."
