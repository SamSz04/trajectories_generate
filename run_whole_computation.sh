#!/bin/bash
# Run xla_runner.sh for whole_computation modules.
# Usage: bash run_whole_computation.sh <model1> [model2] ...
# Example: bash run_whole_computation.sh Llama-3-8B-sq4k-BF16-L1 Mamba2-370M-BF16-L1

set -euo pipefail

XLA_TOOL="${XLA_TOOL:-$HOME/xla/bazel-bin/xla/tools/run_hlo_module}"
HLO_BASE="${HLO_BASE:-$HOME/hlo_datasets}"
DUMP_BASE="${DUMP_BASE:-$HOME/trajectories_generate/output/multi_dumps}"
SA_TRAJECTORIES="${SA_NUM_TRAJECTORIES:-0}"  # No SA for whole_computation (too slow)

for model in "$@"; do
    hlo="$HLO_BASE/$model/whole_computation/whole_computation_stripped.hlo"
    dump="$DUMP_BASE/$model/whole_computation"

    if [ ! -f "$hlo" ]; then
        echo "[SKIP] $model: stripped HLO not found at $hlo"
        continue
    fi

    echo "============================================"
    echo "Model: $model / whole_computation"
    echo "HLO: $hlo"
    echo "Dump: $dump"
    echo "============================================"

    SA_NUM_TRAJECTORIES=0 bash ~/trajectories_generate/xla_runner.sh \
        --gpu a100 \
        --xla-tool "$XLA_TOOL" \
        --hlo-input "$hlo" \
        --dump-base "$dump" \
        --sa-trajectories 0

    echo ""
    echo "[DONE] $model / whole_computation"
    echo ""
done

echo "All whole_computation modules completed."
