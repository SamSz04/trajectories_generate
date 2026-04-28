#!/bin/bash
set -euo pipefail

# LOO Evaluation Pipeline
# For each holdout architecture:
# 1. Generate constrained DT plans using LOO checkpoint
# 2. Profile via nsys
#
# Usage: bash eval_loo.sh [--gen-only] [--profile-only]
#
# Requires: go_fusion in PYTHONPATH, XLA tool, nsys

cd ~/trajectories_generate

XLA_TOOL=~/xla/bazel-bin/xla/tools/run_hlo_module
TRAJ_DIR=output/multi_trajectories
GPU_CSV=output/gpu_profiles_all.csv
DUMP_BASE=output/multi_dumps

# Architecture -> holdout modules mapping
declare -A ARCH_MODULES
ARCH_MODULES["Griffin-2B"]="Griffin-2B-BF16-L6/attention_block Griffin-2B-BF16-L6/griffin_ffn Griffin-2B-BF16-L6/recurrent_block Griffin-2B-BF16-L6/rg_lru"
ARCH_MODULES["Mamba2-370M"]="Mamba2-370M-BF16-L1/mamba_block Mamba2-370M-BF16-L1/mamba_conv_ssm"
ARCH_MODULES["SigLIP-Base"]="SigLIP-Base-BF16/vision_attention SigLIP-Base-BF16/vision_mlp"
ARCH_MODULES["PaliGemma-3B"]="PaliGemma-3B-BF16-L1/vision_mlp"
ARCH_MODULES["DeepSeek-V3"]="DeepSeek-V3-BF16-L2/gate_logit DeepSeek-V3-BF16-L2/moe_permute DeepSeek-V3-BF16-L2/moe_unpermute DeepSeek-V3-BF16-L2/routed_block DeepSeek-V3-BF16-L2/routed_mlp DeepSeek-V3-BF16-L2/shared_block DeepSeek-V3-BF16-L2/shared_block_mlp_up_proj"
ARCH_MODULES["Llama-3-8B"]="Llama-3-8B-sq4k-BF16-L1/ffn_layer Llama-3-8B-sq4k-BF16-L1/gqa_attention_dot Llama-3-8B-sq4k-BF16-L1/gqa_ffn_layer Llama-3-8B-sq4k-BF16-L1/gqa_layer Llama-3-8B-sq4k-BF16-L1/gqa_qkv_proj Llama-3-8B-sq4k-BF16-L1/mlp_up_proj"

ARCHS=("Griffin-2B" "Mamba2-370M" "SigLIP-Base" "PaliGemma-3B" "DeepSeek-V3" "Llama-3-8B")

GEN_ONLY=false
PROFILE_ONLY=false
for arg in "$@"; do
    case $arg in
        --gen-only) GEN_ONLY=true ;;
        --profile-only) PROFILE_ONLY=true ;;
    esac
done

for arch in "${ARCHS[@]}"; do
    echo "============================================"
    echo "LOO Eval: $arch holdout"
    echo "Start: $(date)"
    echo "============================================"

    ckpt="output/dt_loo_${arch}/best.pt"
    plans_dir="output/dt_loo_plans_${arch}"
    profile_csv="output/gpu_profiles_loo_${arch}.csv"

    if [ ! -f "$ckpt" ]; then
        echo "[SKIP] No checkpoint: $ckpt"
        continue
    fi

    mkdir -p "$plans_dir"

    # Get holdout modules
    modules="${ARCH_MODULES[$arch]}"
    module_args=""
    for mod in $modules; do
        module_args="$module_args --modules $mod"
    done

    # Step 1: Generate constrained DT plans for holdout modules
    if [ "$PROFILE_ONLY" = false ]; then
        echo "  Generating plans for holdout modules..."
        PYTHONPATH=~/go_fusion python3 -u dt_eval_multi.py \
            --checkpoint "$ckpt" \
            --traj-dir "$TRAJ_DIR" \
            --gpu-csv "$GPU_CSV" \
            --reward-mode gpu_hybrid \
            --output-dir "$plans_dir" \
            --fusion-penalty 3.0 \
            --max-consec-fusion 2 \
            --max-steps 500 \
            --device cuda \
            $module_args \
            2>&1 | tail -20

        echo "  Plans generated in $plans_dir:"
        ls "$plans_dir"/*.json 2>/dev/null | wc -l
    fi

    # Step 2: Profile plans via nsys
    if [ "$GEN_ONLY" = false ]; then
        echo "  Profiling plans..."
        python3 profile_json_plan.py \
            --xla-tool "$XLA_TOOL" \
            --mode dt-plans \
            --plans-dir "$plans_dir" \
            --dump-base "$DUMP_BASE" \
            --output "$profile_csv" \
            --iterations 10 \
            2>&1 | tail -20
    fi

    echo ""
    echo "[DONE] $arch - $(date)"
    echo ""
done

echo "============================================"
echo "All LOO evaluations complete!"
echo "============================================"

# Summary: compare LOO DT vs full DT vs XLA
echo ""
echo "Profile CSVs:"
for arch in "${ARCHS[@]}"; do
    csv="output/gpu_profiles_loo_${arch}.csv"
    if [ -f "$csv" ]; then
        echo "  $csv: $(wc -l < "$csv") rows"
    fi
done
