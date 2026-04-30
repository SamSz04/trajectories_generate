#!/bin/bash
# Collect trajectories for all modules with dumps on this node.
# Run on each A100 node after profiling (base, expanded, SA) completes.
#
# Usage: bash collect_all_on_node.sh [--output-dir DIR]
#
# Requires: go_env conda environment, PYTHONPATH to go_fusion

set -euo pipefail

DUMP_BASE="${DUMP_BASE:-$HOME/trajectories_generate/output/multi_dumps}"
OUTPUT_DIR="${1:-$HOME/trajectories_generate/output/multi_trajectories}"

mkdir -p "${OUTPUT_DIR}"

echo "============================================================"
echo "Trajectory Collection: $(hostname)"
echo "============================================================"
echo "Dump base: ${DUMP_BASE}"
echo "Output dir: ${OUTPUT_DIR}"
echo ""

# Activate conda if available
if [ -f "$HOME/miniconda3/bin/activate" ]; then
    source "$HOME/miniconda3/bin/activate" go_env 2>/dev/null || true
fi

# Set PYTHONPATH for GO_fusion
export PYTHONPATH="${PYTHONPATH:-$HOME/go_fusion}"

COLLECTED=0
SKIPPED=0
FAILED=0

for model_dir in "${DUMP_BASE}"/*/; do
    [ -d "$model_dir" ] || continue
    MODEL=$(basename "$model_dir")

    for module_dir in "${model_dir}"/*/; do
        [ -d "$module_dir" ] || continue
        MODULE=$(basename "$module_dir")

        # Skip if no xla_default dump
        if [ ! -d "${module_dir}/xla_default" ]; then
            continue
        fi

        # Count strategy dump directories (excluding sa_work_*)
        NUM_STRATS=$(find "$module_dir" -maxdepth 1 -type d \
            ! -name "sa_work_*" ! -name "$(basename "$module_dir")" \
            -exec sh -c 'ls "$1"/*priority_fusion_dump* 2>/dev/null | head -1' _ {} \; \
            | wc -l)

        if [ "$NUM_STRATS" -lt 2 ]; then
            echo "  SKIP ${MODEL}/${MODULE}: only ${NUM_STRATS} strategy dumps"
            SKIPPED=$((SKIPPED + 1))
            continue
        fi

        # Find HLO input
        HLO=$(find "${module_dir}/xla_default" -name "*before_priority-fusion*" 2>/dev/null | head -1)
        if [ -z "$HLO" ]; then
            HLO=$(find "${module_dir}/xla_default" -name "*before_optimizations*" 2>/dev/null | head -1)
        fi
        if [ -z "$HLO" ]; then
            echo "  SKIP ${MODEL}/${MODULE}: no HLO input"
            SKIPPED=$((SKIPPED + 1))
            continue
        fi

        OUTPUT_FILE="${OUTPUT_DIR}/${MODEL}__${MODULE}.pt"
        echo "  Collecting ${MODEL}/${MODULE} (${NUM_STRATS} strategies)..."

        if python3 collect_trajectories.py collect \
                --dump-base "$module_dir" \
                --hlo-input "$HLO" \
                --output "$OUTPUT_FILE" 2>&1 | tail -3; then
            COLLECTED=$((COLLECTED + 1))
        else
            echo "  FAILED ${MODEL}/${MODULE}"
            FAILED=$((FAILED + 1))
        fi
        echo ""
    done
done

echo "============================================================"
echo "Collection complete: ${COLLECTED} collected, ${SKIPPED} skipped, ${FAILED} failed"
echo "Output: ${OUTPUT_DIR}/"
echo "============================================================"
