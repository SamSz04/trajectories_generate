#!/bin/bash
# build_xla.sh — Build patched XLA from source for a target GPU.
#
# Usage:
#   ./build_xla.sh --gpu a100    # A100: compute capability 8.0
#   ./build_xla.sh --gpu h100    # H100: compute capability 9.0
#
# Prerequisites:
#   - Bazelisk (or Bazel 6.x+)
#   - Clang 18+ (or GCC 12+)
#   - CUDA toolkit 12.x
#   - 32 GB+ RAM for compilation

set -euo pipefail

GPU_TARGET=""
XLA_DIR="${XLA_DIR:-./xla}"
PATCH_FILE="${PATCH_FILE:-$(dirname "$0")/priority_fusion_patch.diff}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu) GPU_TARGET="$2"; shift 2 ;;
        --xla-dir) XLA_DIR="$2"; shift 2 ;;
        --patch) PATCH_FILE="$2"; shift 2 ;;
        *) echo "Unknown flag: $1"; exit 1 ;;
    esac
done

if [[ -z "$GPU_TARGET" ]]; then
    echo "ERROR: --gpu flag is required. Specify 'a100' or 'h100'."
    exit 1
fi

case "$GPU_TARGET" in
    a100|A100|a100_40gb|a100-40gb)
        COMPUTE_CAP="8.0"
        GPU_NAME="A100"
        ;;
    h100|H100)
        COMPUTE_CAP="9.0"
        GPU_NAME="H100"
        ;;
    a100_h100|both)
        COMPUTE_CAP="8.0,9.0"
        GPU_NAME="A100+H100"
        ;;
    *)
        echo "ERROR: Unsupported GPU target '$GPU_TARGET'."
        echo "Supported: a100, h100, both"
        exit 1
        ;;
esac

echo "============================================"
echo "Building XLA for $GPU_NAME (sm_${COMPUTE_CAP//./})"
echo "============================================"
echo "XLA directory: $XLA_DIR"
echo "Patch file:    $PATCH_FILE"
echo "Compute cap:   $COMPUTE_CAP"
echo "============================================"
echo ""

# Step 1: Clone XLA if not present
if [[ ! -d "$XLA_DIR" ]]; then
    echo "=== Cloning XLA repository ==="
    git clone https://github.com/openxla/xla.git "$XLA_DIR"
fi

cd "$XLA_DIR"

# Step 2: Apply the patch
if [[ -f "$PATCH_FILE" ]]; then
    echo "=== Applying priority_fusion patch ==="
    # Check if already applied
    if git apply --check "$PATCH_FILE" 2>/dev/null; then
        git apply "$PATCH_FILE"
        echo "  Patch applied successfully."
    else
        echo "  Patch already applied or conflicts detected. Skipping."
    fi
else
    echo "WARNING: Patch file not found at $PATCH_FILE"
    echo "Continuing with unpatched XLA (xla_default strategy only)."
fi

# Step 3: Configure
echo ""
echo "=== Configuring XLA ==="
./configure.py --backend=CUDA --cuda_compute_capabilities="$COMPUTE_CAP"

# Step 4: Build
echo ""
echo "=== Building run_hlo_module ==="
echo "This will take 1-2 hours on first build..."
bazel build //xla/tools:run_hlo_module

# Step 5: Verify
BINARY="bazel-bin/xla/tools/run_hlo_module"
if [[ -f "$BINARY" ]]; then
    echo ""
    echo "============================================"
    echo "Build successful!"
    echo "Binary: $(pwd)/$BINARY"
    echo "============================================"
    echo ""
    echo "Usage:"
    echo "  export XLA_TOOL=$(pwd)/$BINARY"
    echo "  cd $(dirname "$0")"
    echo "  ./xla_runner.sh --gpu $GPU_TARGET --xla-tool \$XLA_TOOL"
else
    echo ""
    echo "ERROR: Build failed. Binary not found at $BINARY"
    exit 1
fi
