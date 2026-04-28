#!/bin/bash
# xla_runner.sh — Batch execution of all 50 fusion strategy configurations.
#
# Runs on a GPU machine (A100 or H100) with patched XLA.
# Each run produces a priority_fusion_dump.txt in the output directory.
#
# Usage:
#   ./xla_runner.sh --gpu a100 [--xla-tool PATH] [--hlo-input PATH] [--dump-base PATH]
#
# Prerequisites:
#   - Patched XLA built for the target GPU (see build_xla.sh)
#   - CUDA toolkit and target GPU available

set -euo pipefail

# =====================================================================
# Configuration (override via command-line flags or env vars)
# =====================================================================
XLA_TOOL="${XLA_TOOL:-/path/to/bazel-bin/xla/tools/run_hlo_module}"
HLO_INPUT="${HLO_INPUT:-hlo_dumps/llama3_8B_gqa_layer/gqa_layer_original.hlo}"
DUMP_BASE="${DUMP_BASE:-output/dumps}"
SA_NUM_TRAJECTORIES="${SA_NUM_TRAJECTORIES:-10}"
GPU_TARGET="${GPU_TARGET:-}"

# Parse command-line flags
while [[ $# -gt 0 ]]; do
    case "$1" in
        --xla-tool) XLA_TOOL="$2"; shift 2 ;;
        --hlo-input) HLO_INPUT="$2"; shift 2 ;;
        --dump-base) DUMP_BASE="$2"; shift 2 ;;
        --sa-trajectories) SA_NUM_TRAJECTORIES="$2"; shift 2 ;;
        --gpu) GPU_TARGET="$2"; shift 2 ;;
        *) echo "Unknown flag: $1"; exit 1 ;;
    esac
done

# Validate GPU target
if [[ -z "$GPU_TARGET" ]]; then
    echo "ERROR: --gpu flag is required. Specify 'a100' or 'h100'."
    exit 1
fi

case "$GPU_TARGET" in
    a100|A100|a100_40gb|a100-40gb)
        GPU_NAME="A100"
        COMPUTE_CAP="8.0"
        ;;
    h100|H100)
        GPU_NAME="H100"
        COMPUTE_CAP="9.0"
        ;;
    *)
        echo "ERROR: Unsupported GPU target '$GPU_TARGET'. Use 'a100' or 'h100'."
        exit 1
        ;;
esac

# Verify prerequisites
if [[ ! -f "$XLA_TOOL" ]]; then
    echo "ERROR: XLA tool not found at $XLA_TOOL"
    echo "Build it with: ./build_xla.sh --gpu $GPU_TARGET"
    exit 1
fi

if [[ ! -f "$HLO_INPUT" ]]; then
    echo "ERROR: HLO input not found at $HLO_INPUT"
    exit 1
fi

mkdir -p "$DUMP_BASE"

# Save GPU metadata for collect_trajectories.py
cat > "$DUMP_BASE/gpu_metadata.json" << EOF
{
    "gpu_target": "$GPU_TARGET",
    "gpu_name": "$GPU_NAME",
    "compute_capability": "$COMPUTE_CAP",
    "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
EOF

# =====================================================================
# Helper: run one XLA compilation with a given strategy
# =====================================================================
run_strategy() {
    local name="$1"
    shift
    local dump_dir="$DUMP_BASE/$name"
    mkdir -p "$dump_dir"

    echo "[$(date +%H:%M:%S)] Running strategy: $name"

    # Set XLA dump flags + any strategy-specific env vars
    env "$@" \
        XLA_FLAGS="--xla_dump_to=$dump_dir --xla_dump_hlo_as_text --xla_dump_hlo_pass_re=.*priority.fusion.*" \
        "$XLA_TOOL" --platform=CUDA --reference_platform='' "$HLO_INPUT" \
        2>&1 | tee "$dump_dir/xla_output.log" || true

    # Find the priority_fusion_dump.txt (may have module prefix)
    local dump_file
    dump_file=$(find "$dump_dir" -name "*priority_fusion_dump.txt" -print -quit 2>/dev/null)
    if [[ -n "$dump_file" && -f "$dump_file" ]]; then
        # Symlink to standard name for easy access
        ln -sf "$(basename "$dump_file")" "$dump_dir/priority_fusion_dump.txt" 2>/dev/null || true
        echo "  -> Dump saved: $(wc -l < "$dump_file") lines"
    else
        echo "  -> WARNING: No dump file produced for $name"
    fi
}

echo "============================================"
echo "XLA Fusion Trajectory Generation"
echo "============================================"
echo "GPU target:  $GPU_NAME (sm_${COMPUTE_CAP//.})"
echo "XLA tool:    $XLA_TOOL"
echo "HLO input:   $HLO_INPUT"
echo "Dump base:   $DUMP_BASE"
echo "============================================"
echo ""

# =====================================================================
# Strategy 1: XLA default (no strategy env var)
# =====================================================================
echo "=== Phase 1: XLA Default (1 run) ==="
run_strategy "xla_default"
echo ""

# =====================================================================
# Strategies 2-21: Random (seeds 0-19)
# =====================================================================
echo "=== Phase 2: Random Strategies (20 runs) ==="
for seed in $(seq 0 19); do
    run_strategy "random_$seed" \
        DT_FUSION_STRATEGY=random \
        DT_FUSION_SEED="$seed"
done
echo ""

# =====================================================================
# Strategies 22-30: Perturbed (sigma x seed combinations)
# =====================================================================
echo "=== Phase 3: Perturbed Strategies (9 runs) ==="
for sigma in 0.1 0.5 1.0; do
    for seed in 0 1 2; do
        run_strategy "perturbed_${sigma}_${seed}" \
            DT_FUSION_STRATEGY=perturbed \
            DT_FUSION_SIGMA="$sigma" \
            DT_FUSION_SEED="$seed"
    done
done
echo ""

# =====================================================================
# Strategies 31-40: Greedy heuristic variants
# =====================================================================
echo "=== Phase 4: Greedy Variants (10 runs) ==="
for strategy in fanout tensor_bytes flop_count reverse_topo; do
    run_strategy "greedy_$strategy" \
        DT_FUSION_STRATEGY="$strategy"
done

# Additional greedy variants via random with specific seeds that
# produce meaningfully different orderings
for seed in 100 101 102 103 104 105; do
    run_strategy "greedy_random_walk_$seed" \
        DT_FUSION_STRATEGY=random \
        DT_FUSION_SEED="$seed"
done
echo ""

# =====================================================================
# Strategies 41-50: Simulated Annealing (via sa_orchestrator.py)
# =====================================================================
echo "=== Phase 5: Simulated Annealing ($SA_NUM_TRAJECTORIES runs) ==="
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python3 "$SCRIPT_DIR/sa_orchestrator.py" \
    --xla-tool="$XLA_TOOL" \
    --hlo-input="$HLO_INPUT" \
    --dump-base="$DUMP_BASE" \
    --num-trajectories="$SA_NUM_TRAJECTORIES"
echo ""

# =====================================================================
# Summary
# =====================================================================
echo "============================================"
echo "All runs complete! (GPU: $GPU_NAME)"
echo "============================================"
num_dumps=$(find "$DUMP_BASE" -name "priority_fusion_dump.txt" | wc -l)
echo "Total dumps produced: $num_dumps"
echo "Dump directories:"
ls -1d "$DUMP_BASE"/*/
echo ""
echo "Next step: python collect_trajectories.py collect --dump-base=$DUMP_BASE --hlo-input=$HLO_INPUT"
