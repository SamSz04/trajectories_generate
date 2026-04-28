#!/bin/bash
# multi_module_runner.sh — Run trajectory generation across all viable HLO modules.
#
# Phase 1: Validate each candidate (xla_default only, ~2 min each)
# Phase 2: Full 50-strategy generation for validated modules
# Phase 3: Collect per-module .pt trajectory files
#
# Usage:
#   bash multi_module_runner.sh [--validate-only] [--skip-validation] [--module MODEL/MODULE]
#
# Requires: patched XLA (run_hlo_module), GO_fusion (for collect_trajectories.py)

set -uo pipefail

# =====================================================================
# Configuration
# =====================================================================
HLO_DATASET_DIR="${HLO_DATASET_DIR:-$HOME/hlo_datasets}"
OUTPUT_DIR="${OUTPUT_DIR:-$HOME/trajectories_generate/output/multi_dumps}"
TRAJ_DIR="${TRAJ_DIR:-$HOME/trajectories_generate/output/multi_trajectories}"
XLA_TOOL="${XLA_TOOL:-$HOME/xla/bazel-bin/xla/tools/run_hlo_module}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
GPU_TARGET="${GPU_TARGET:-a100}"
MIN_FUSIONS=3
VALIDATE_ONLY=false
SKIP_VALIDATION=false
SINGLE_MODULE=""
VALIDATION_TIMEOUT=300  # 5 min timeout for validation runs

while [[ $# -gt 0 ]]; do
    case "$1" in
        --validate-only) VALIDATE_ONLY=true; shift ;;
        --skip-validation) SKIP_VALIDATION=true; shift ;;
        --module) SINGLE_MODULE="$2"; shift 2 ;;
        --gpu) GPU_TARGET="$2"; shift 2 ;;
        *) echo "Unknown flag: $1"; exit 1 ;;
    esac
done

# =====================================================================
# Define all candidate modules: MODEL_DIR/MODULE_DIR
# (≥10 instructions, no whole_computation duplicates)
# =====================================================================
ALL_CANDIDATES=(
    # Llama-3 8B (Tier 1 — clean, no weights)
    "Llama-3-8B-sq4k-BF16-L1/gqa_ffn_layer"
    "Llama-3-8B-sq4k-BF16-L1/gqa_layer"
    "Llama-3-8B-sq4k-BF16-L1/gqa_qkv_proj"
    "Llama-3-8B-sq4k-BF16-L1/gqa_attention_dot"
    "Llama-3-8B-sq4k-BF16-L1/ffn_layer"
    "Llama-3-8B-sq4k-BF16-L1/mlp_up_proj"

    # DeepSeek-V3 (Tier 1 — clean, no weights)
    "DeepSeek-V3-BF16-L2/moe_permute"
    "DeepSeek-V3-BF16-L2/routed_block"
    "DeepSeek-V3-BF16-L2/moe_unpermute"
    "DeepSeek-V3-BF16-L2/routed_mlp"
    "DeepSeek-V3-BF16-L2/shared_block"
    "DeepSeek-V3-BF16-L2/shared_block_mlp_up_proj"
    "DeepSeek-V3-BF16-L2/gate_logit"

    # Tier 2 — Embedded weights or unusual ops
    "Mamba2-370M-BF16-L1/mamba_conv_ssm"
    "Mamba2-370M-BF16-L1/mamba_block"
    "SigLIP-Base-BF16/vision_attention"
    "SigLIP-Base-BF16/vision_mlp"
    "PaliGemma-3B-BF16-L1/vision_mlp"
    "Griffin-2B-BF16-L6/attention_block"
    "Griffin-2B-BF16-L6/griffin_ffn"
    "Griffin-2B-BF16-L6/recurrent_block"
    "Griffin-2B-BF16-L6/rg_lru"
)

# Filter to single module if specified
if [[ -n "$SINGLE_MODULE" ]]; then
    ALL_CANDIDATES=("$SINGLE_MODULE")
fi

echo "============================================================"
echo "Multi-Module Trajectory Generation"
echo "============================================================"
echo "HLO dataset:  $HLO_DATASET_DIR"
echo "Output dumps: $OUTPUT_DIR"
echo "Trajectories: $TRAJ_DIR"
echo "XLA tool:     $XLA_TOOL"
echo "GPU target:   $GPU_TARGET"
echo "Candidates:   ${#ALL_CANDIDATES[@]}"
echo "============================================================"
echo ""

mkdir -p "$OUTPUT_DIR" "$TRAJ_DIR"

# =====================================================================
# Phase 1: Validation — run xla_default for each candidate
# =====================================================================
VIABLE_MODULES=()
VALIDATION_LOG="$OUTPUT_DIR/validation_results.txt"
> "$VALIDATION_LOG"

if [[ "$SKIP_VALIDATION" == "true" ]]; then
    echo "=== Phase 1: SKIPPED (--skip-validation) ==="
    # Assume all candidates are viable
    VIABLE_MODULES=("${ALL_CANDIDATES[@]}")
else
    echo "=== Phase 1: Validation (xla_default only) ==="
    echo ""

    for mod in "${ALL_CANDIDATES[@]}"; do
        model=$(echo "$mod" | cut -d/ -f1)
        module=$(echo "$mod" | cut -d/ -f2)
        hlo_file="$HLO_DATASET_DIR/$mod/$module.hlo"

        if [[ ! -f "$hlo_file" ]]; then
            echo "  SKIP: $mod — HLO file not found: $hlo_file"
            echo "SKIP $mod NO_HLO_FILE" >> "$VALIDATION_LOG"
            continue
        fi

        dump_dir="$OUTPUT_DIR/$mod/xla_default"
        mkdir -p "$dump_dir"

        echo -n "  Validating $mod ... "

        # Run xla_default strategy
        XLA_FLAGS="--xla_dump_to=$dump_dir --xla_dump_hlo_as_text --xla_dump_hlo_pass_re=.*priority.fusion.*" \
            timeout "$VALIDATION_TIMEOUT" \
            "$XLA_TOOL" --platform=CUDA --reference_platform='' "$hlo_file" \
            > "$dump_dir/xla_output.log" 2>&1
        exit_code=$?

        if [[ $exit_code -ne 0 ]]; then
            echo "FAIL (exit code $exit_code)"
            echo "FAIL $mod EXIT_$exit_code" >> "$VALIDATION_LOG"
            continue
        fi

        # Find the dump file (may have module prefix)
        dump_file=$(find "$dump_dir" -name "*priority_fusion_dump.txt" -not -name "*.symlink" -print -quit 2>/dev/null)

        if [[ -z "$dump_file" || ! -f "$dump_file" ]]; then
            echo "FAIL (no fusion dump produced)"
            echo "FAIL $mod NO_DUMP" >> "$VALIDATION_LOG"
            continue
        fi

        # Count fusion steps (update_priority = actual fusion events)
        fusions=$(grep -c "update_priority" "$dump_file" 2>/dev/null || echo 0)

        if [[ $fusions -lt $MIN_FUSIONS ]]; then
            echo "SKIP ($fusions fusions < $MIN_FUSIONS minimum)"
            echo "SKIP $mod FUSIONS_$fusions" >> "$VALIDATION_LOG"
            continue
        fi

        # Create symlink for easy access
        ln -sf "$(basename "$dump_file")" "$dump_dir/priority_fusion_dump.txt" 2>/dev/null || true

        echo "PASS ($fusions fusions)"
        echo "PASS $mod FUSIONS_$fusions" >> "$VALIDATION_LOG"
        VIABLE_MODULES+=("$mod")
    done

    echo ""
    echo "Validation complete: ${#VIABLE_MODULES[@]}/${#ALL_CANDIDATES[@]} modules viable"
    echo ""
    cat "$VALIDATION_LOG"
    echo ""
fi

if [[ "$VALIDATE_ONLY" == "true" ]]; then
    echo "=== Validation-only mode, stopping here ==="
    exit 0
fi

if [[ ${#VIABLE_MODULES[@]} -eq 0 ]]; then
    echo "ERROR: No viable modules found!"
    exit 1
fi

# =====================================================================
# Phase 2: Full trajectory generation for viable modules
# =====================================================================
echo "=== Phase 2: Full Trajectory Generation (${#VIABLE_MODULES[@]} modules) ==="
echo ""

PHASE2_START=$(date +%s)

for i in "${!VIABLE_MODULES[@]}"; do
    mod="${VIABLE_MODULES[$i]}"
    model=$(echo "$mod" | cut -d/ -f1)
    module=$(echo "$mod" | cut -d/ -f2)
    hlo_file="$HLO_DATASET_DIR/$mod/$module.hlo"
    dump_base="$OUTPUT_DIR/$mod"

    echo "[$((i+1))/${#VIABLE_MODULES[@]}] Processing $mod ..."
    mod_start=$(date +%s)

    # Run xla_runner.sh with module-specific paths
    bash "$SCRIPT_DIR/xla_runner.sh" \
        --gpu "$GPU_TARGET" \
        --xla-tool "$XLA_TOOL" \
        --hlo-input "$hlo_file" \
        --dump-base "$dump_base" \
        2>&1 | sed 's/^/    /'

    mod_elapsed=$(( $(date +%s) - mod_start ))
    echo "    Completed $mod in ${mod_elapsed}s"
    echo ""
done

PHASE2_ELAPSED=$(( $(date +%s) - PHASE2_START ))
echo "Phase 2 complete in ${PHASE2_ELAPSED}s"
echo ""

# =====================================================================
# Phase 3: Collect per-module trajectories
# =====================================================================
echo "=== Phase 3: Trajectory Collection ==="
echo ""

COLLECT_RESULTS="$TRAJ_DIR/collection_summary.txt"
> "$COLLECT_RESULTS"

for mod in "${VIABLE_MODULES[@]}"; do
    model=$(echo "$mod" | cut -d/ -f1)
    module=$(echo "$mod" | cut -d/ -f2)
    dump_base="$OUTPUT_DIR/$mod"

    # Find the before_priority-fusion.txt in xla_default dump
    hlo_input=$(find "$dump_base/xla_default" -name "*before_priority-fusion*" -print -quit 2>/dev/null)

    if [[ -z "$hlo_input" || ! -f "$hlo_input" ]]; then
        echo "  SKIP: $mod — no before_priority-fusion.txt found"
        echo "SKIP $mod NO_BEFORE_PF" >> "$COLLECT_RESULTS"
        continue
    fi

    output_file="$TRAJ_DIR/${model}__${module}.pt"

    echo -n "  Collecting $mod ... "

    PYTHONPATH="$HOME/go_fusion" \
        python "$SCRIPT_DIR/collect_trajectories.py" collect \
        --dump-base "$dump_base" \
        --hlo-input "$hlo_input" \
        --output "$output_file" \
        > "$TRAJ_DIR/${model}__${module}.log" 2>&1

    if [[ $? -eq 0 && -f "$output_file" ]]; then
        # Extract summary stats from the log
        trajs=$(grep "Total trajectories:" "$TRAJ_DIR/${model}__${module}.log" | grep -o '[0-9]*' | head -1)
        echo "OK ($trajs trajectories)"
        echo "OK $mod TRAJS_${trajs:-?}" >> "$COLLECT_RESULTS"
    else
        echo "FAIL"
        echo "FAIL $mod" >> "$COLLECT_RESULTS"
    fi
done

echo ""
echo "============================================================"
echo "Multi-Module Trajectory Generation Complete"
echo "============================================================"
echo ""
echo "Results:"
cat "$COLLECT_RESULTS"
echo ""
echo "Trajectory files:"
ls -lh "$TRAJ_DIR"/*.pt 2>/dev/null || echo "  (none)"
echo ""
echo "Total size:"
du -sh "$TRAJ_DIR" 2>/dev/null
