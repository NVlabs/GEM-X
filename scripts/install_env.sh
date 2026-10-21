#!/usr/bin/env bash
set -euo pipefail

# Run inside the intended environment. GPU users may install their preferred
# compatible PyTorch wheel first; this command preserves a satisfying version.
# Optional argument: comma-separated extras, for example "train,eval,dev".
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
extras="${1:-base}"
package="$project_root"
if [[ "$extras" != base ]]; then
    package="${project_root}[${extras}]"
fi

python -m pip install -e "$package"
