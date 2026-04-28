#!/bin/bash
# Expanded trajectory generation + profiling pipeline.
#
# Runs on a single A100 server. Handles:
# 1. Generate perturbed plans for assigned modules
# 2. Profile expanded strategies (random_20-99 + 20 perturbed sigmas) with dumps
# 3. Profile perturbed plan strategies with dumps
#
# Usage:
#   bash run_expanded_batch.sh <module1> <module2> ...
#
# Example:
#   # node-0: Llama-3 modules
#   bash run_expanded_batch.sh \
#       Llama-3-8B-sq4k-BF16-L1/gqa_layer \
#       Llama-3-8B-sq4k-BF16-L1/ffn_layer \
#       Llama-3-8B-sq4k-BF16-L1/whole_computation \
#       Llama-3-8B-sq4k-BF16-L1/attn_layer \
#       Llama-3-8B-sq4k-BF16-L1/embed_layer \
#       Llama-3-8B-sq4k-BF16-L1/final_layer
#
# Server distribution (copy-paste for each server):
#   node-0: Llama-3 × 6 modules
#   node-1: DeepSeek × 7 modules
#   node-2: SigLIP × 2 + PaliGemma × 1 + Mamba × 2 = 5 modules
#   node-3: Griffin × 4 modules

set -euo pipefail

# Configuration
XLA_TOOL="${XLA_TOOL:-$HOME/xla/bazel-bin/xla/tools/run_hlo_module}"
DUMP_BASE="${DUMP_BASE:-output/multi_dumps}"
PLAN_DIR="${PLAN_DIR:-output/perturbed_plans}"
OUTPUT_CSV="${OUTPUT_CSV:-output/gpu_profiles_expanded.csv}"
ITERATIONS="${ITERATIONS:-10}"
TIMEOUT="${TIMEOUT:-600}"
NUM_PLANS="${NUM_PLANS:-20}"
NUM_SWAPS="${NUM_SWAPS:-3}"

MODULES=("$@")
if [ ${#MODULES[@]} -eq 0 ]; then
    echo "Usage: bash run_expanded_batch.sh <module1> <module2> ..."
    echo ""
    echo "Example modules:"
    echo "  Llama-3-8B-sq4k-BF16-L1/gqa_layer"
    echo "  DeepSeek-V3-BF16-L1/moe_layer"
    exit 1
fi

echo "============================================================"
echo "Expanded Trajectory Generation + Profiling Pipeline"
echo "============================================================"
echo "Server: $(hostname)"
echo "XLA tool: ${XLA_TOOL}"
echo "Dump base: ${DUMP_BASE}"
echo "Modules: ${#MODULES[@]}"
echo "Output CSV: ${OUTPUT_CSV}"
echo "Iterations: ${ITERATIONS}"
echo "Timeout: ${TIMEOUT}s"
echo ""

# Detect WC modules and adjust timeout
get_timeout() {
    local module="$1"
    if [[ "$module" == *"whole_computation"* ]]; then
        echo 1800
    else
        echo "${TIMEOUT}"
    fi
}

# Step 1: Generate perturbed plans
echo "=== Step 1: Generating perturbed plans ==="
PYTHONPATH="${PYTHONPATH:-$HOME/go_fusion}" python3 generate_perturbed_plans.py \
    --dump-base "${DUMP_BASE}" \
    --output-dir "${PLAN_DIR}" \
    --num-plans "${NUM_PLANS}" \
    --num-swaps "${NUM_SWAPS}" \
    --modules "${MODULES[@]}"

echo ""

# Step 2: Profile expanded env-var strategies (random_20-99 + 20 perturbed sigmas) with dumps
echo "=== Step 2: Profiling expanded strategies (with dumps) ==="
PYTHONPATH="${PYTHONPATH:-$HOME/go_fusion}" python3 -u profile_strategies.py \
    --xla-tool "${XLA_TOOL}" \
    --dump-base "${DUMP_BASE}" \
    --output "${OUTPUT_CSV}" \
    --iterations "${ITERATIONS}" \
    --strategy-set expanded \
    --with-dumps \
    --timeout "${TIMEOUT}" \
    --modules "${MODULES[@]}"

echo ""

# Step 3: Profile plan perturbation strategies with dumps
echo "=== Step 3: Profiling plan perturbation strategies (with dumps) ==="
# Use a separate strategy-set 'base' with 0 env-var strategies,
# but with --plan-dir to pick up the plan files.
# We use --strategies with a dummy to skip all env-var strategies,
# relying only on plan-dir discovery.
PYTHONPATH="${PYTHONPATH:-$HOME/go_fusion}" python3 -u profile_strategies.py \
    --xla-tool "${XLA_TOOL}" \
    --dump-base "${DUMP_BASE}" \
    --output "${OUTPUT_CSV}" \
    --iterations "${ITERATIONS}" \
    --strategy-set base \
    --strategies __none__ \
    --plan-dir "${PLAN_DIR}" \
    --with-dumps \
    --timeout "${TIMEOUT}" \
    --modules "${MODULES[@]}"

echo ""
echo "============================================================"
echo "Done! Results in ${OUTPUT_CSV}"
echo "Dumps in ${DUMP_BASE}/<module>/<strategy>/"
echo "============================================================"
