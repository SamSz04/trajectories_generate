#!/bin/bash
# Regenerate missing priority_fusion_dump.txt files.
# Runs run_hlo_module (without nsys) for each strategy dir that has
# HLO files but no priority_fusion_dump.txt.
#
# Usage: bash regen_dumps.sh [MODULE1 MODULE2 ...]
# Example: bash regen_dumps.sh Griffin-2B-BF16-L6/griffin_ffn
# Default: scan all modules

set -uo pipefail

XLA_TOOL="${XLA_TOOL:-$HOME/xla/bazel-bin/xla/tools/run_hlo_module}"
DUMP_BASE="${DUMP_BASE:-$HOME/trajectories_generate/output/multi_dumps}"
TIMEOUT=120  # seconds per compilation

MODULES=("$@")

REGENERATED=0
SKIPPED=0
FAILED=0
NEED_REGEN=0

for model_dir in "${DUMP_BASE}"/*/; do
    [ -d "$model_dir" ] || continue
    MODEL=$(basename "$model_dir")

    for module_dir in "${model_dir}"/*/; do
        [ -d "$module_dir" ] || continue
        MODULE=$(basename "$module_dir")
        MODULE_PATH="${MODEL}/${MODULE}"

        # Filter by args if provided
        if [ ${#MODULES[@]} -gt 0 ]; then
            MATCH=0
            for m in "${MODULES[@]}"; do
                if [ "$m" = "$MODULE_PATH" ]; then MATCH=1; fi
            done
            [ $MATCH -eq 0 ] && continue
        fi

        # Find HLO input - prefer original from hlo_datasets
        HLO=$(ls "${HOME}/hlo_datasets/${MODEL}/${MODULE}/"*.hlo 2>/dev/null | head -1)
        if [ -z "$HLO" ]; then
            HLO=$(find "${module_dir}/xla_default" -name "*before_optimizations*" 2>/dev/null | head -1)
        fi
        [ -z "$HLO" ] && continue
        echo "Module: ${MODULE_PATH} (HLO: $(basename "$HLO"))"

        for strat_dir in "${module_dir}"/*/; do
            [ -d "$strat_dir" ] || continue
            STRAT=$(basename "$strat_dir")
            [ "$STRAT" = "xla_default" ] && continue
            [[ "$STRAT" == sa_work_* ]] && continue

            # Skip if dump already exists
            if ls "${strat_dir}"/*priority_fusion_dump* >/dev/null 2>&1; then
                SKIPPED=$((SKIPPED + 1))
                continue
            fi

            # Skip if no HLO files at all (empty dir)
            if [ ! -f "${strat_dir}/module_0001"*.txt 2>/dev/null ] && \
               ! ls "${strat_dir}"/module_0001.* >/dev/null 2>&1; then
                continue
            fi

            # Determine strategy env vars from dir name
            ENV_VARS=""
            case "$STRAT" in
                random_*)
                    SEED=${STRAT#random_}
                    ENV_VARS="DT_FUSION_STRATEGY=random DT_FUSION_SEED=${SEED}"
                    ;;
                perturbed_*_*)
                    # perturbed_SIGMA_SEED
                    REST=${STRAT#perturbed_}
                    SEED=${REST##*_}
                    SIGMA=${REST%_*}
                    ENV_VARS="DT_FUSION_STRATEGY=perturbed DT_FUSION_SIGMA=${SIGMA} DT_FUSION_SEED=${SEED}"
                    ;;
                greedy_fanout) ENV_VARS="DT_FUSION_STRATEGY=fanout" ;;
                greedy_tensor_bytes) ENV_VARS="DT_FUSION_STRATEGY=tensor_bytes" ;;
                greedy_flop_count) ENV_VARS="DT_FUSION_STRATEGY=flop_count" ;;
                greedy_reverse_topo) ENV_VARS="DT_FUSION_STRATEGY=reverse_topo" ;;
                greedy_random_walk_*)
                    SEED=${STRAT#greedy_random_walk_}
                    ENV_VARS="DT_FUSION_STRATEGY=random DT_FUSION_SEED=${SEED}"
                    ;;
                sa_*|plan_*) continue ;;  # SA/plan dirs handled differently
                *) continue ;;
            esac

            echo -n "  ${MODULE_PATH}/${STRAT}..."
            NEED_REGEN=$((NEED_REGEN + 1))

            # Run XLA compilation to generate dump
            if timeout ${TIMEOUT} env ${ENV_VARS} \
                XLA_FLAGS="--xla_gpu_enable_command_buffer= --xla_gpu_autotune_level=0 --xla_dump_to=${strat_dir} --xla_dump_hlo_as_text --xla_dump_hlo_pass_re=.*priority.fusion.*" \
                "${XLA_TOOL}" \
                --platform=CUDA --reference_platform= --input_format=hlo --iterations=1 \
                "${HLO}" > /dev/null 2>&1; then

                # Clean up non-module_0001 files
                for f in "${strat_dir}"/module_0???.*; do
                    case "$(basename "$f")" in
                        module_0001.*) ;;
                        *) rm -f "$f" ;;
                    esac
                done

                if ls "${strat_dir}"/*priority_fusion_dump* >/dev/null 2>&1; then
                    echo " OK"
                    REGENERATED=$((REGENERATED + 1))
                else
                    echo " no dump produced"
                    FAILED=$((FAILED + 1))
                fi
            else
                echo " FAIL/TIMEOUT"
                FAILED=$((FAILED + 1))
            fi
        done
    done
done

echo "============================================================"
echo "Regen complete: ${REGENERATED} regenerated, ${SKIPPED} already had dump, ${NEED_REGEN} attempted, ${FAILED} failed"
echo "============================================================"
