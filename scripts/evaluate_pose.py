# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Source-checkout wrapper for gem-eval pose."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gem.pose.evaluation.cli import main

if __name__ == "__main__":
    main(["pose", *sys.argv[1:]])
