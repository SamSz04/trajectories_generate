#!/bin/bash
# generate_all_dumps.sh — Re-generate priority_fusion_dump.txt for all 33 modules
# across all 40 non-SA strategies. Resumable: skips dumps that already exist.
#
# Usage:
#   bash generate_all_dumps.sh [--start-from N] [--only-model MODEL]
#
# Runs on the GPU server with patched XLA binary.

set -uo pipefail

XLA_TOOL="${XLA_TOOL:-/root/rivermind-data/xla/bazel-bin/xla/tools/run_hlo_module}"
HLO_BASE="${HLO_BASE:-/root/hlo_datasets}"
DUMP_BASE="${DUMP_BASE:-/root/rivermind-data/trajectories_generate/output/multi_dumps}"
LOG_FILE="${DUMP_BASE}/generation_log.txt"

START_FROM=0
ONLY_MODEL=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --start-from) START_FROM="$2"; shift 2 ;;
        --only-model) ONLY_MODEL="$2"; shift 2 ;;
        *) echo "Unknown flag: $1"; exit 1 ;;
    esac
done

if [[ ! -f "$XLA_TOOL" ]]; then
    echo "ERROR: XLA tool not found at $XLA_TOOL"
    exit 1
fi

mkdir -p "$DUMP_BASE"

# =====================================================================
# Module definitions: MODEL_NAME MODULE_NAME
# (33 modules matching multi_trajectories_merged/*.pt)
# =====================================================================
MODULES=(
    "DeepSeek-V3-BF16-L2 DeepSeekV3Model_computation"
    "DeepSeek-V3-BF16-L2 gate_logit"
    "DeepSeek-V3-BF16-L2 moe_permute"
    "DeepSeek-V3-BF16-L2 moe_unpermute"
    "DeepSeek-V3-BF16-L2 routed_block"
    "DeepSeek-V3-BF16-L2 routed_mlp"
    "DeepSeek-V3-BF16-L2 shared_block"
    "DeepSeek-V3-BF16-L2 shared_block_mlp_up_proj"
    "DeepSeek-V3-BF16-L2 whole_computation"
    "Griffin-2B-BF16-L6 GriffinModel_computation"
    "Griffin-2B-BF16-L6 attention_block"
    "Griffin-2B-BF16-L6 griffin_ffn"
    "Griffin-2B-BF16-L6 recurrent_block"
    "Griffin-2B-BF16-L6 rg_lru"
    "Llama-3-8B-sq4k-BF16-L1 Llama3Transformer_computation"
    "Llama-3-8B-sq4k-BF16-L1 ffn_layer"
    "Llama-3-8B-sq4k-BF16-L1 gqa_attention_dot"
    "Llama-3-8B-sq4k-BF16-L1 gqa_ffn_layer"
    "Llama-3-8B-sq4k-BF16-L1 gqa_layer"
    "Llama-3-8B-sq4k-BF16-L1 gqa_qkv_proj"
    "Llama-3-8B-sq4k-BF16-L1 mlp_up_proj"
    "Llama-3-8B-sq4k-BF16-L1 whole_computation"
    "Mamba2-370M-BF16-L1 Mamba2Model_computation"
    "Mamba2-370M-BF16-L1 mamba_block"
    "Mamba2-370M-BF16-L1 mamba_conv_ssm"
    "Mamba2-370M-BF16-L1 whole_computation"
    "PaliGemma-3B-BF16-L1 vision_attention"
    "PaliGemma-3B-BF16-L1 vision_mlp"
    "PaliGemma-3B-BF16-L1 whole_computation"
    "SigLIP-Base-BF16 patch_embed"
    "SigLIP-Base-BF16 vision_attention"
    "SigLIP-Base-BF16 vision_mlp"
    "SigLIP-Base-BF16 whole_computation"
)

# =====================================================================
# Strategy definitions (40 strategies, no SA)
# Format: STRATEGY_NAME ENV_VARS...
# =====================================================================
run_one() {
    local dump_dir="$1"
    local strat_name="$2"
    shift 2
    local strat_dir="$dump_dir/$strat_name"

    # Skip if dump already exists
    if [[ -f "$strat_dir/priority_fusion_dump.txt" ]] || \
       find "$strat_dir" -name "*priority_fusion_dump.txt" -print -quit 2>/dev/null | grep -q .; then
        return 0
    fi

    mkdir -p "$strat_dir"

    env "$@" \
        XLA_FLAGS="--xla_dump_to=$strat_dir --xla_dump_hlo_as_text --xla_dump_hlo_pass_re=.*priority.fusion.*" \
        "$XLA_TOOL" --platform=CUDA --reference_platform='' "$HLO_INPUT" \
        > "$strat_dir/xla_output.log" 2>&1 || true

    # Symlink dump file to standard name
    local dump_file
    dump_file=$(find "$strat_dir" -name "*priority_fusion_dump.txt" -print -quit 2>/dev/null)
    if [[ -n "$dump_file" && -f "$dump_file" ]]; then
        ln -sf "$(basename "$dump_file")" "$strat_dir/priority_fusion_dump.txt" 2>/dev/null || true
    fi
}

run_all_strategies() {
    local dump_dir="$1"

    # Phase 1: XLA Default (1)
    run_one "$dump_dir" "xla_default"

    # Phase 2: Random (20)
    for seed in $(seq 0 19); do
        run_one "$dump_dir" "random_$seed" \
            DT_FUSION_STRATEGY=random DT_FUSION_SEED="$seed"
    done

    # Phase 3: Perturbed (9)
    for sigma in 0.1 0.5 1.0; do
        for seed in 0 1 2; do
            run_one "$dump_dir" "perturbed_${sigma}_${seed}" \
                DT_FUSION_STRATEGY=perturbed DT_FUSION_SIGMA="$sigma" DT_FUSION_SEED="$seed"
        done
    done

    # Phase 4: Greedy variants (10)
    for strategy in fanout tensor_bytes flop_count reverse_topo; do
        run_one "$dump_dir" "greedy_$strategy" \
            DT_FUSION_STRATEGY="$strategy"
    done
    for seed in 100 101 102 103 104 105; do
        run_one "$dump_dir" "greedy_random_walk_$seed" \
            DT_FUSION_STRATEGY=random DT_FUSION_SEED="$seed"
    done
}

# =====================================================================
# Main loop
# =====================================================================
TOTAL=${#MODULES[@]}
echo "============================================"
echo "Batch Dump Generation"
echo "============================================"
echo "Modules: $TOTAL"
echo "Strategies per module: 40 (no SA)"
echo "Total runs: $((TOTAL * 40))"
echo "XLA tool: $XLA_TOOL"
echo "HLO base: $HLO_BASE"
echo "Dump base: $DUMP_BASE"
echo "Start from: module #$START_FROM"
[[ -n "$ONLY_MODEL" ]] && echo "Only model: $ONLY_MODEL"
echo "============================================"
echo ""

COMPLETED=0
FAILED=0

for i in "${!MODULES[@]}"; do
    if (( i < START_FROM )); then
        continue
    fi

    read -r model module <<< "${MODULES[$i]}"

    # Filter by model if specified
    if [[ -n "$ONLY_MODEL" && "$model" != "$ONLY_MODEL" ]]; then
        continue
    fi

    HLO_INPUT="$HLO_BASE/$model/$module/$module.hlo"
    MODULE_DUMP_DIR="$DUMP_BASE/$model/$module"

    if [[ ! -f "$HLO_INPUT" ]]; then
        echo "[$((i+1))/$TOTAL] SKIP $model/$module — HLO not found: $HLO_INPUT"
        ((FAILED++))
        continue
    fi

    # Count existing dumps
    existing=$(find "$MODULE_DUMP_DIR" -name "*priority_fusion_dump.txt" 2>/dev/null | wc -l)

    echo "[$((i+1))/$TOTAL] $model/$module ($existing/40 existing)"
    ts_start=$(date +%s)

    export HLO_INPUT
    run_all_strategies "$MODULE_DUMP_DIR"

    ts_end=$(date +%s)
    elapsed=$((ts_end - ts_start))
    new_count=$(find "$MODULE_DUMP_DIR" -name "*priority_fusion_dump.txt" 2>/dev/null | wc -l)
    echo "  -> Done: $new_count dumps in ${elapsed}s"
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $model/$module $new_count dumps ${elapsed}s" >> "$LOG_FILE"

    ((COMPLETED++))
done

echo ""
echo "============================================"
echo "Complete! $COMPLETED modules processed, $FAILED skipped"
echo "============================================"
total_dumps=$(find "$DUMP_BASE" -name "*priority_fusion_dump.txt" 2>/dev/null | wc -l)
echo "Total dump files: $total_dumps"
