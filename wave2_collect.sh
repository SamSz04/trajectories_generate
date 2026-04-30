#!/bin/bash
# Wave 2 orchestration: collect all data from A100 nodes.
#
# Run from local machine after Wave 1 (SA + expanded profiling) completes.
#
# Steps:
# 1. Collect expanded and SA CSV files from all nodes
# 2. Merge CSVs into unified gpu_profiles_all_v2.csv
# 3. Trigger trajectory collection on each node (in parallel)
# 4. Download .pt trajectory files from each node
# 5. Merge trajectories that were collected on different nodes
#
# Usage: bash wave2_collect.sh

set -euo pipefail

LOCAL_OUTPUT="trajectories_generate/output"
CSV_DIR="${LOCAL_OUTPUT}/node_csvs"
TRAJ_DIR="${LOCAL_OUTPUT}/multi_trajectories_v2"
mkdir -p "${CSV_DIR}" "${TRAJ_DIR}"

# Node SSH names
ALL_NODES=(a100-node a100-node-1 a100-node-2 a100-node-3 \
           a100-node-4 a100-node-5 a100-node-6 a100-node-7 \
           a100-node-8 a100-node-9 a100-node-10)

echo "============================================================"
echo "Wave 2: Data Collection from ${#ALL_NODES[@]} nodes"
echo "============================================================"
echo ""

# ── Step 1: Collect CSVs ─────────────────────────────────────────

echo "=== Step 1: Collecting CSV files ==="

for node in "${ALL_NODES[@]}"; do
    echo -n "  ${node}: "

    # Expanded CSV
    scp -o ConnectTimeout=10 "${node}:~/trajectories_generate/output/gpu_profiles_expanded.csv" \
        "${CSV_DIR}/expanded_${node}.csv" 2>/dev/null && echo -n "expanded " || echo -n "- "

    # SA CSV
    scp -o ConnectTimeout=10 "${node}:~/trajectories_generate/output/gpu_profiles_sa.csv" \
        "${CSV_DIR}/sa_${node}.csv" 2>/dev/null && echo -n "sa " || echo -n "- "

    # WC CSV
    scp -o ConnectTimeout=10 "${node}:~/trajectories_generate/output/gpu_profiles_wc_*.csv" \
        "${CSV_DIR}/" 2>/dev/null && echo -n "wc " || echo -n "- "

    # Base CSV (original profiling)
    scp -o ConnectTimeout=10 "${node}:~/trajectories_generate/output/gpu_profiles_all.csv" \
        "${CSV_DIR}/base_${node}.csv" 2>/dev/null && echo -n "base " || echo -n "- "

    echo ""
done

echo ""

# ── Step 2: Merge CSVs ──────────────────────────────────────────

echo "=== Step 2: Merging CSVs ==="

MERGED="${LOCAL_OUTPUT}/gpu_profiles_all_v2.csv"

# Start with header from any CSV
HEADER=""
for f in "${CSV_DIR}"/*.csv; do
    if [ -f "$f" ]; then
        HEADER=$(head -1 "$f")
        break
    fi
done

if [ -z "$HEADER" ]; then
    echo "  ERROR: No CSV files found!"
    exit 1
fi

echo "$HEADER" > "${MERGED}"

# Append all CSVs (skip headers, deduplicate by module+strategy)
for f in "${CSV_DIR}"/*.csv; do
    [ -f "$f" ] || continue
    tail -n +2 "$f" >> "${MERGED}.tmp" 2>/dev/null || true
done

# Deduplicate: keep first occurrence of each (module, strategy) pair
if [ -f "${MERGED}.tmp" ]; then
    # Sort and deduplicate by first two fields (module, strategy)
    sort -t',' -k1,2 -u "${MERGED}.tmp" >> "${MERGED}"
    rm "${MERGED}.tmp"
fi

TOTAL_ROWS=$(wc -l < "${MERGED}")
echo "  Merged CSV: ${MERGED} (${TOTAL_ROWS} rows including header)"

echo ""

# ── Step 3: Deploy collection script and run on each node ────────

echo "=== Step 3: Trajectory collection on nodes ==="

# Deploy collect_all_on_node.sh
for node in "${ALL_NODES[@]}"; do
    scp -o ConnectTimeout=10 trajectories_generate/collect_all_on_node.sh \
        "${node}:~/trajectories_generate/" 2>/dev/null &
done
wait

echo "  Scripts deployed. Launching collection..."

# Launch collection on each node in parallel
for node in "${ALL_NODES[@]}"; do
    echo "  Starting collection on ${node}..."
    ssh -o ConnectTimeout=10 "${node}" "
        cd ~/trajectories_generate && \
        nohup bash collect_all_on_node.sh > /tmp/collect.log 2>&1 &
        echo PID=\$!
    " 2>/dev/null &
done
wait

echo ""
echo "  Collection launched on all nodes. Monitor with:"
echo "    ssh <node> 'tail -f /tmp/collect.log'"
echo ""

# ── Step 4: Wait and download .pt files ──────────────────────────

echo "=== Step 4: Waiting for collection to complete... ==="
echo "  (Check manually and run: bash wave2_collect.sh download)"
echo ""

if [ "${1:-}" = "download" ]; then
    echo "=== Downloading .pt files ==="

    for node in "${ALL_NODES[@]}"; do
        echo -n "  ${node}: "
        NODE_TRAJ_DIR="${TRAJ_DIR}/${node}"
        mkdir -p "${NODE_TRAJ_DIR}"

        scp -o ConnectTimeout=10 "${node}:~/trajectories_generate/output/multi_trajectories/*.pt" \
            "${NODE_TRAJ_DIR}/" 2>/dev/null && \
            echo "$(ls "${NODE_TRAJ_DIR}"/*.pt 2>/dev/null | wc -l) files" || \
            echo "none"
    done

    echo ""
    echo "  Downloaded to ${TRAJ_DIR}/<node>/"
    echo "  Next: run merge_trajectories.py to combine"
fi

echo "============================================================"
echo "Wave 2 complete!"
echo "============================================================"
