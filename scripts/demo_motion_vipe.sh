#!/usr/bin/env bash
set -euo pipefail

# Optional shortcut; see docs/VIPE_CAMERA.md for setup and arguments.
exec gem-infer motion-video --task body --camera-mode vipe "$@"
