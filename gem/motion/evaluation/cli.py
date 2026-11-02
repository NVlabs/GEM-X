# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Evaluate native Motion inference weights on prepared full benchmark sequences."""

import argparse
import json
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device")
    parser.add_argument("--postprocess", choices=("raw", "pp", "both"), default="both")
    parser.add_argument(
        "--sequences", nargs="+", help="Exact sequence names or DATASET/sequence names"
    )
    args = parser.parse_args(argv)
    from .runner import load_config, run_evaluation

    config = load_config(args.config)
    result = run_evaluation(
        config,
        args.data_index,
        device=args.device or config.get("device", "cuda"),
        postprocess=args.postprocess,
        sequences=args.sequences,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"Saved Motion evaluation to {args.output}")


if __name__ == "__main__":
    main()
