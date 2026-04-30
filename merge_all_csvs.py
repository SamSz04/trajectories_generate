#!/usr/bin/env python3
"""Merge all GPU profile CSVs from all nodes, deduplicate, and analyze gaps."""
import csv
import glob
import os
import sys
from collections import defaultdict

def get_expected_env_strategies():
    """Return all 140 expected env-var strategy names."""
    strategies = []
    # xla_default
    strategies.append("xla_default")
    # random 0-19 (base)
    for seed in range(20):
        strategies.append(f"random_{seed}")
    # perturbed base: 3 sigma x 3 seeds
    for sigma in ["0.1", "0.5", "1.0"]:
        for seed in range(3):
            strategies.append(f"perturbed_{sigma}_{seed}")
    # greedy variants
    for v in ["fanout", "tensor_bytes", "flop_count", "reverse_topo"]:
        strategies.append(f"greedy_{v}")
    # random walk 100-105
    for seed in range(100, 106):
        strategies.append(f"greedy_random_walk_{seed}")
    # expanded: random 20-99
    for seed in range(20, 100):
        strategies.append(f"random_{seed}")
    # expanded perturbed
    for sigma in [0.05, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.6, 0.7,
                  0.8, 1.2, 1.5, 1.8, 2.0, 2.5, 3.5, 5.0, 8.0, 15.0]:
        strategies.append(f"perturbed_{sigma}_0")
    return strategies

def main():
    csv_dir = sys.argv[1] if len(sys.argv) > 1 else "trajectories_generate/output/all_csvs"
    output = sys.argv[2] if len(sys.argv) > 2 else "trajectories_generate/output/gpu_profiles_merged.csv"

    # Collect all rows, dedup by (module, strategy) keeping first occurrence
    seen = {}  # (module, strategy) -> row dict
    file_count = 0

    for csv_path in sorted(glob.glob(os.path.join(csv_dir, "*.csv"))):
        fname = os.path.basename(csv_path)
        added = 0
        try:
            with open(csv_path, 'r') as f:
                reader = csv.DictReader(f)
                if not reader.fieldnames or 'module' not in reader.fieldnames:
                    continue
                for row in reader:
                    key = (row["module"], row["strategy"])
                    if key not in seen:
                        seen[key] = row
                        added += 1
            file_count += 1
            print(f"  {fname}: +{added} new (total rows in file: counted)")
        except Exception as e:
            print(f"  {fname}: ERROR {e}")

    print(f"\nMerged {file_count} CSV files → {len(seen)} unique (module, strategy) pairs")

    # Write merged CSV
    if seen:
        fieldnames = list(next(iter(seen.values())).keys())
        os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
        with open(output, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for key in sorted(seen):
                writer.writerow(seen[key])
        print(f"Written to: {output}")

    # Gap analysis
    expected = get_expected_env_strategies()
    modules = sorted(set(k[0] for k in seen))

    print(f"\n{'='*80}")
    print(f"GAP ANALYSIS: {len(modules)} modules × {len(expected)} env-var strategies = {len(modules)*len(expected)} expected")
    print(f"{'='*80}")

    # Per-module breakdown
    total_missing = 0
    module_gaps = {}
    fmt = "{:<55} {:>5} {:>5} {:>8}"
    print(fmt.format("Module", "Have", "Exp", "Missing"))
    print("-" * 80)

    for module in modules:
        have = set()
        for strat in expected:
            if (module, strat) in seen:
                have.add(strat)
        # Also count non-env strategies (SA, plan, etc.)
        all_strats = [k[1] for k in seen if k[0] == module]
        non_env = [s for s in all_strats if s not in set(expected)]

        missing = [s for s in expected if s not in have]
        total_missing += len(missing)
        module_gaps[module] = missing

        extra = f" (+{len(non_env)} other)" if non_env else ""
        print(fmt.format(module, len(have), len(expected), len(missing)) + extra)

    print("-" * 80)
    print(f"Total missing env-var profiles: {total_missing}")
    print(f"Total unique pairs: {len(seen)}")

    # Show which strategies are most commonly missing
    if total_missing > 0:
        print(f"\n{'='*80}")
        print("MISSING STRATEGIES BY MODULE (showing modules with gaps):")
        print(f"{'='*80}")
        for module in modules:
            missing = module_gaps[module]
            if missing:
                # Group by prefix
                random_missing = [s for s in missing if s.startswith("random_")]
                perturbed_missing = [s for s in missing if s.startswith("perturbed_")]
                greedy_missing = [s for s in missing if s.startswith("greedy_")]
                other_missing = [s for s in missing if not any(s.startswith(p) for p in ["random_", "perturbed_", "greedy_"])]

                print(f"\n  {module} ({len(missing)} missing):")
                if random_missing:
                    seeds = sorted([int(s.split("_")[1]) for s in random_missing])
                    print(f"    random: seeds {seeds}")
                if perturbed_missing:
                    print(f"    perturbed: {perturbed_missing}")
                if greedy_missing:
                    print(f"    greedy: {greedy_missing}")
                if other_missing:
                    print(f"    other: {other_missing}")

    # SA/plan strategy summary
    print(f"\n{'='*80}")
    print("NON-ENV STRATEGIES (SA, plan, etc.):")
    print(f"{'='*80}")
    for module in modules:
        all_strats = sorted([k[1] for k in seen if k[0] == module])
        non_env = [s for s in all_strats if s not in set(expected)]
        if non_env:
            print(f"  {module}: {non_env}")

if __name__ == "__main__":
    main()
