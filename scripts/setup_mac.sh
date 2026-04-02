#!/usr/bin/env bash
# One-step macOS setup for GEM on Apple Silicon.
#
# This script handles everything:
#   1. Checks prerequisites (Python 3.12+, uv, Git LFS, Apple Silicon)
#   2. Creates virtual environment
#   3. Installs PyTorch, SOMA, GEM, and all dependencies
#   4. Downloads ONNX models from HuggingFace
#   5. Sets up SOMA assets symlink
#   6. Creates RAM disk and installs packages for fast imports (optional)
#
# After running this script, launch the demo with:
#   python scripts/demo/demo_soma_onnx.py --video path/to/video.mp4
#
# Usage:
#   bash scripts/setup_mac.sh            # Full setup
#   bash scripts/setup_mac.sh --no-ramdisk   # Skip RAM disk creation
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_DIR"

RAMDISK_MOUNT="/Volumes/TorchRAM"
SKIP_RAMDISK=false
for arg in "$@"; do
    case "$arg" in
        --no-ramdisk) SKIP_RAMDISK=true ;;
    esac
done

# ---------------------------------------------------------------------------
#  Color helpers
# ---------------------------------------------------------------------------
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

ok()   { echo -e "${GREEN}[OK]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
fail() { echo -e "${RED}[FAIL]${NC} $1"; exit 1; }
step() { echo -e "\n${GREEN}==>${NC} $1"; }

# ---------------------------------------------------------------------------
#  Step 0: Prerequisites
# ---------------------------------------------------------------------------
step "Checking prerequisites..."

# Apple Silicon
if [[ "$(uname -m)" != "arm64" ]]; then
    fail "This script requires Apple Silicon (arm64). Detected: $(uname -m)"
fi
ok "Apple Silicon (arm64)"

# macOS version
MACOS_VERSION=$(sw_vers -productVersion | cut -d. -f1)
if [[ "$MACOS_VERSION" -lt 13 ]]; then
    fail "macOS 13 (Ventura) or later required. Detected: $(sw_vers -productVersion)"
fi
ok "macOS $(sw_vers -productVersion)"

# Python 3.12+
if command -v python3.12 &>/dev/null; then
    PYTHON_CMD="python3.12"
elif command -v python3 &>/dev/null; then
    PY_VER=$(python3 -c 'import sys; print(sys.version_info.minor)')
    if [[ "$PY_VER" -ge 12 ]]; then
        PYTHON_CMD="python3"
    else
        fail "Python 3.12+ required. Found: python3.$(python3 -c 'import sys; print(sys.version_info.minor)')"
    fi
else
    fail "Python 3.12+ not found. Install with: brew install python@3.12"
fi
ok "Python ($($PYTHON_CMD --version))"

# uv package manager
if ! command -v uv &>/dev/null; then
    warn "'uv' not found. Installing..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
    if ! command -v uv &>/dev/null; then
        fail "Failed to install uv. Install manually: https://github.com/astral-sh/uv"
    fi
fi
ok "uv $(uv --version 2>/dev/null | head -1)"

# Git LFS
if ! command -v git-lfs &>/dev/null; then
    warn "Git LFS not found. Installing via Homebrew..."
    if command -v brew &>/dev/null; then
        brew install git-lfs
        git lfs install
    else
        fail "Git LFS required. Install with: brew install git-lfs"
    fi
fi
ok "Git LFS"

# ---------------------------------------------------------------------------
#  Step 1: Submodules
# ---------------------------------------------------------------------------
step "Initializing submodules..."
git submodule update --init --recursive 2>/dev/null || true
ok "Submodules initialized"

# ---------------------------------------------------------------------------
#  Step 2: Virtual environment
# ---------------------------------------------------------------------------
step "Setting up virtual environment..."
if [ -d ".venv" ] && [ -f ".venv/bin/python" ]; then
    ok "Virtual environment already exists at .venv/"
else
    uv venv .venv --python 3.12
    ok "Created .venv/"
fi

VENV_PYTHON="$REPO_DIR/.venv/bin/python"
# shellcheck disable=SC1091
source .venv/bin/activate

# ---------------------------------------------------------------------------
#  Step 3: PyTorch (MPS backend)
# ---------------------------------------------------------------------------
step "Installing PyTorch..."
if uv pip show torch &>/dev/null; then
    TORCH_VER=$(uv pip show torch 2>/dev/null | grep Version | awk '{print $2}')
    ok "PyTorch already installed ($TORCH_VER)"
else
    uv pip install torch torchvision
    ok "PyTorch installed"
fi
ok "MPS (Metal) backend available on Apple Silicon"

# ---------------------------------------------------------------------------
#  Step 4: SOMA body model
# ---------------------------------------------------------------------------
step "Installing SOMA body model..."
if uv pip show py-soma-x &>/dev/null; then
    ok "SOMA already installed"
else
    uv pip install -e third_party/soma
    ok "SOMA installed"
fi

# Pull LFS files
if [ -f "third_party/soma/assets/SOMA_neutral.npz" ] && [ "$(wc -c < "third_party/soma/assets/SOMA_neutral.npz")" -gt 1000 ]; then
    ok "SOMA LFS assets already pulled"
else
    echo "    Pulling SOMA Git LFS files..."
    (cd third_party/soma && git lfs pull)
    ok "SOMA LFS assets pulled"
fi

# ---------------------------------------------------------------------------
#  Step 5: GEM + dependencies
# ---------------------------------------------------------------------------
step "Installing GEM and dependencies..."
# Check for the editable install marker and a key dependency
if [ -f ".venv/lib/python3.12/site-packages/__editable___gem_0_1_0_finder.py" ] \
   && [ -d ".venv/lib/python3.12/site-packages/roma" ]; then
    ok "GEM and dependencies already installed"
else
    uv pip install -e . 2>&1 | tail -1
    uv pip install cloudpickle fvcore iopath pycocotools braceexpand roma 'setuptools<75' 2>&1 | tail -1
    ok "GEM and dependencies installed"
fi

# ---------------------------------------------------------------------------
#  Step 6: ONNX Runtime
# ---------------------------------------------------------------------------
step "Installing ONNX Runtime..."
if uv pip show onnxruntime &>/dev/null; then
    ok "ONNX Runtime already installed"
else
    uv pip install onnxruntime
    ok "ONNX Runtime installed"
fi

# FP16 conversion dependencies (for auto-converting VitPose on first run)
if uv pip show onnx onnxconverter-common &>/dev/null; then
    ok "FP16 conversion tools already installed"
else
    uv pip install onnx onnxconverter-common 2>&1 | tail -1
    ok "FP16 conversion tools installed"
fi

# ---------------------------------------------------------------------------
#  Step 7: Download ONNX models
# ---------------------------------------------------------------------------
step "Checking ONNX models..."
if [ -f "inputs/onnx/vitpose.onnx" ] && [ -f "inputs/onnx/gem_denoiser_no_imgfeat.onnx" ]; then
    ok "ONNX models already present"
else
    echo "    ONNX models not found — they will auto-download on first demo run."
    echo "    To download now (requires internet, ~4.6 GB):"
    echo "      source .venv/bin/activate"
    echo "      python -c \"from gem.utils.hf_utils import download_all_onnx; download_all_onnx()\""
    warn "ONNX models will be downloaded on first run"
fi

# ---------------------------------------------------------------------------
#  Step 8: SOMA assets symlink
# ---------------------------------------------------------------------------
step "Setting up SOMA assets..."
if [ -L "inputs/soma_assets" ] || [ -d "inputs/soma_assets" ]; then
    ok "inputs/soma_assets already exists"
else
    mkdir -p inputs
    ln -sf "$REPO_DIR/third_party/soma/assets" inputs/soma_assets
    ok "Symlinked inputs/soma_assets -> third_party/soma/assets"
fi

# ---------------------------------------------------------------------------
#  Step 9: RAM disk
# ---------------------------------------------------------------------------
if [ "$SKIP_RAMDISK" = true ]; then
    warn "Skipping RAM disk setup (--no-ramdisk)"
else
    step "Setting up RAM disk for fast imports..."
    if [ -d "$RAMDISK_MOUNT" ]; then
        ok "RAM disk already mounted at $RAMDISK_MOUNT"
    else
        echo "    Creating 2 GB RAM disk at $RAMDISK_MOUNT..."
        DISK=$(hdiutil attach -nomount ram://4194304)
        DISK=$(echo "$DISK" | xargs)
        diskutil eraseDisk HFS+ TorchRAM "$DISK" >/dev/null 2>&1
        ok "RAM disk created"

        echo "    Installing packages to RAM disk (this takes ~1-2 minutes)..."
        uv pip install --target "$RAMDISK_MOUNT" \
            torch numpy onnxruntime opencv-python scipy einops viser \
            torchvision typing_extensions pyyaml pillow packaging trimesh \
            warp-lang \
            --no-deps --python "$VENV_PYTHON" --quiet 2>&1 | grep -v "^warning:" || true
        ok "Packages installed to RAM disk"
    fi
fi

# ---------------------------------------------------------------------------
#  Done
# ---------------------------------------------------------------------------
echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Setup complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Run the ONNX accelerated demo with:"
echo ""
echo "  source .venv/bin/activate"
echo "  python scripts/demo/demo_soma_onnx.py --video path/to/video.mp4"
echo ""
echo "ONNX models will auto-download on first run if not already present."
echo "VitPose FP16 conversion happens automatically on first run (~2-3 min, one-time)."
echo ""
echo "See docs/INSTALL_MACOS.md for detailed setup and troubleshooting."
