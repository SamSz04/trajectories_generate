#!/bin/bash
# SA (Simulated Annealing) batch runner for fusion ordering search.
#
# For each assigned module:
#   1. Runs sa_orchestrator.py to find good orderings (10 trajectories, 100 iters × 4 restarts)
#   2. Profiles the resulting SA dumps with nsys to get real GPU kernel times
#
# Usage:
#   bash run_sa_batch.sh Llama-3-8B-sq4k-BF16-L1/gqa_layer DeepSeek-V3-BF16-L2/gate_logit ...
#
# SA creates: sa_0/, sa_1/, ..., sa_9/ dump dirs + sa_*_plan.json files per module

set -uo pipefail

# Configuration
XLA_TOOL="${XLA_TOOL:-$HOME/xla/bazel-bin/xla/tools/run_hlo_module}"
DUMP_BASE="${DUMP_BASE:-$HOME/trajectories_generate/output/multi_dumps}"
SA_OUTPUT_CSV="${SA_OUTPUT_CSV:-$HOME/trajectories_generate/output/gpu_profiles_sa.csv}"
SA_NUM_TRAJECTORIES="${SA_NUM_TRAJECTORIES:-10}"
SA_ITERATIONS="${SA_ITERATIONS:-100}"
PROFILE_ITERATIONS="${PROFILE_ITERATIONS:-10}"

MODULES=("$@")
if [ ${#MODULES[@]} -eq 0 ]; then
    echo "Usage: bash run_sa_batch.sh <model/module> [<model/module> ...]"
    echo "Example: bash run_sa_batch.sh Llama-3-8B-sq4k-BF16-L1/gqa_layer"
    exit 1
fi

echo "============================================================"
echo "SA Batch Runner"
echo "============================================================"
echo "Server: $(hostname)"
echo "XLA tool: ${XLA_TOOL}"
echo "Dump base: ${DUMP_BASE}"
echo "Modules: ${#MODULES[@]}"
echo "SA trajectories: ${SA_NUM_TRAJECTORIES}"
echo "SA iterations: ${SA_ITERATIONS}"
echo "Profile iterations: ${PROFILE_ITERATIONS}"
echo "Output CSV: ${SA_OUTPUT_CSV}"
echo ""

cd ~/trajectories_generate

for MODULE in "${MODULES[@]}"; do
    echo ""
    echo "============================================================"
    echo "=== SA Search: ${MODULE}"
    echo "============================================================"

    MODULE_DUMP="${DUMP_BASE}/${MODULE}"

    # Check dump directory exists
    if [ ! -d "${MODULE_DUMP}" ]; then
        echo "  SKIP: dump directory not found: ${MODULE_DUMP}"
        continue
    fi

    # Check if SA already completed (sa_0/ exists with dump file)
    if [ -d "${MODULE_DUMP}/sa_0" ] && ls "${MODULE_DUMP}"/sa_0/*priority_fusion_dump* >/dev/null 2>&1; then
        SA_COUNT=$(ls -d "${MODULE_DUMP}"/sa_* 2>/dev/null | grep -c '/sa_[0-9]')
        echo "  SA already done (${SA_COUNT} trajectories found). Skipping SA search."
    else
        # Find HLO input from xla_default dump directory
        HLO=$(find "${MODULE_DUMP}/xla_default" -name "*before_optimizations*" 2>/dev/null | head -1)
        if [ -z "$HLO" ]; then
            HLO=$(find "${MODULE_DUMP}/xla_default" -name "*before_priority-fusion*" 2>/dev/null | head -1)
        fi
        if [ -z "$HLO" ]; then
            echo "  SKIP: no HLO input found in ${MODULE_DUMP}/xla_default/"
            continue
        fi

        echo "  HLO input: ${HLO}"
        echo "  Starting SA search..."

        python3 sa_orchestrator.py \
            --xla-tool "${XLA_TOOL}" \
            --hlo-input "${HLO}" \
            --dump-base "${MODULE_DUMP}" \
            --num-trajectories "${SA_NUM_TRAJECTORIES}" \
            --iterations "${SA_ITERATIONS}"

        echo "  SA search complete for ${MODULE}"
    fi

    # Step 2: Profile SA dumps with nsys
    echo ""
    echo "  --- Profiling SA dumps for ${MODULE} ---"

    # Find all sa_*_plan.json files for this module
    SA_PLANS=$(ls "${MODULE_DUMP}"/sa_*_plan.json 2>/dev/null)
    if [ -z "$SA_PLANS" ]; then
        echo "  No SA plan files found, skipping profiling."
        continue
    fi

    # Profile using profile_strategies.py with plan-dir pointing to sa plans
    # We create a temporary plan dir structure that profile_strategies.py expects
    MODEL=$(dirname "${MODULE}")
    MOD=$(basename "${MODULE}")
    TEMP_PLAN_DIR=$(mktemp -d)
    SA_PLAN_SUBDIR="${TEMP_PLAN_DIR}/${MODEL}__${MOD}"
    mkdir -p "${SA_PLAN_SUBDIR}"

    # Copy SA plan files with expected naming (plan_*.json)
    IDX=0
    for plan in ${MODULE_DUMP}/sa_*_plan.json; do
        cp "$plan" "${SA_PLAN_SUBDIR}/plan_sa_${IDX}.json"
        IDX=$((IDX + 1))
    done

    echo "  Profiling ${IDX} SA plans..."

    python3 -u profile_strategies.py \
        --xla-tool "${XLA_TOOL}" \
        --dump-base "${DUMP_BASE}" \
        --output "${SA_OUTPUT_CSV}" \
        --iterations "${PROFILE_ITERATIONS}" \
        --strategy-set base \
        --strategies __none__ \
        --plan-dir "${TEMP_PLAN_DIR}" \
        --with-dumps \
        --timeout 600 \
        --modules "${MODULE}"

    rm -rf "${TEMP_PLAN_DIR}"
    echo "  Profiling complete for ${MODULE}"
done

echo ""
echo "============================================================"
echo "All SA batch work complete!"
echo "SA profiles in: ${SA_OUTPUT_CSV}"
echo "SA dumps in: ${DUMP_BASE}/<module>/sa_*/"
echo "============================================================"
