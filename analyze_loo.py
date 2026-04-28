#!/usr/bin/env python3
"""Analyze LOO (Leave-One-Architecture-Out) generalization results.

Compares LOO DT (trained without holdout arch) vs Full DT vs XLA.

Usage:
    python3 analyze_loo.py \
        --profiles output/gpu_profiles_all_with_sa.csv \
        --dt-profiles output/gpu_profiles_dt.csv \
        --loo-dir output/  # looks for gpu_profiles_loo_*.csv
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
from collections import defaultdict
from typing import Dict, Tuple


def load_profiles(csv_path: str) -> Dict[Tuple[str, str], float]:
    data = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            key = (row["module"], row["strategy"])
            data[key] = float(row["avg_kernel_per_iter_us"])
    return data


def format_us(val: float) -> str:
    if val >= 1e6:
        return f"{val/1e6:.1f}s"
    elif val >= 1e3:
        return f"{val/1e3:.1f}ms"
    else:
        return f"{val:.1f}us"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profiles", default="output/gpu_profiles_all_with_sa.csv")
    parser.add_argument("--dt-profiles", default="output/gpu_profiles_dt.csv")
    parser.add_argument("--loo-dir", default="output/")
    args = parser.parse_args()

    profiles = load_profiles(args.profiles)
    dt_profiles = load_profiles(args.dt_profiles)

    # XLA defaults
    xla = {mod: val for (mod, strat), val in profiles.items()
           if strat == "xla_default"}

    # Full DT
    dt_full = {mod: val for (mod, strat), val in dt_profiles.items()}

    # Load LOO profiles
    loo_files = sorted(glob.glob(os.path.join(args.loo_dir, "gpu_profiles_loo_*.csv")))
    loo_data = {}  # {arch: {module: time_us}}
    for f in loo_files:
        arch = os.path.basename(f).replace("gpu_profiles_loo_", "").replace(".csv", "")
        loo_data[arch] = {}
        with open(f) as fh:
            for row in csv.DictReader(fh):
                loo_data[arch][row["module"]] = float(row["avg_kernel_per_iter_us"])

    if not loo_data:
        print("No LOO profile files found!")
        return

    print("## LOO Generalization Results")
    print()
    print(f"Found {len(loo_data)} architectures: {', '.join(sorted(loo_data.keys()))}")
    print()

    # Table
    print("| Module | XLA | Full DT | Full DT% | LOO DT | LOO DT% | Gap |")
    print("|---|---|---|---|---|---|---|")

    gaps = []
    for arch in sorted(loo_data.keys()):
        for mod in sorted(loo_data[arch].keys()):
            x = xla.get(mod, None)
            d_full = dt_full.get(mod, None)
            d_loo = loo_data[arch][mod]

            if x is None or d_full is None:
                continue

            full_pct = (d_full - x) / x * 100
            loo_pct = (d_loo - x) / x * 100
            gap = loo_pct - full_pct

            short = mod.split("/")[-1]
            model = mod.split("/")[0].split("-")[0]

            print(f"| {model}/{short} | {format_us(x)} | {format_us(d_full)} | "
                  f"{full_pct:+.2f}% | {format_us(d_loo)} | {loo_pct:+.2f}% | "
                  f"{gap:+.2f}% |")
            gaps.append(gap)

    print()
    if gaps:
        avg_gap = sum(gaps) / len(gaps)
        max_gap = max(gaps)
        min_gap = min(gaps)
        print(f"Avg generalization gap: {avg_gap:+.3f}%")
        print(f"Max gap: {max_gap:+.3f}%, Min gap: {min_gap:+.3f}%")
        print(f"Modules evaluated: {len(gaps)}")


if __name__ == "__main__":
    main()
