#!/usr/bin/env python3
"""Analyze and compare DT vs XLA default vs SA vs Best Alternative.

Generates tables and statistics for the paper.

Usage:
    python3 analyze_results.py \
        --profiles output/gpu_profiles_all_with_sa.csv \
        --dt-profiles output/gpu_profiles_dt.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple


def load_profiles(csv_path: str) -> Dict[Tuple[str, str], float]:
    """Load (module, strategy) -> avg_kernel_per_iter_us."""
    data = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            key = (row["module"], row["strategy"])
            data[key] = float(row["avg_kernel_per_iter_us"])
    return data


def get_xla_defaults(profiles: dict) -> Dict[str, float]:
    return {mod: val for (mod, strat), val in profiles.items()
            if strat == "xla_default"}


def get_best_per_module(profiles: dict, exclude_xla: bool = True) -> Dict[str, Tuple[float, str]]:
    """Best (lowest) time per module. Returns {module: (time_us, strategy)}."""
    best = {}
    for (mod, strat), val in profiles.items():
        if exclude_xla and strat == "xla_default":
            continue
        if mod not in best or val < best[mod][0]:
            best[mod] = (val, strat)
    return best


def get_sa_best(profiles: dict) -> Dict[str, Tuple[float, str]]:
    """Best SA time per module."""
    best = {}
    for (mod, strat), val in profiles.items():
        if strat.startswith("sa_"):
            if mod not in best or val < best[mod][0]:
                best[mod] = (val, strat)
    return best


def format_us(val: float) -> str:
    """Format microseconds with appropriate units."""
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
    parser.add_argument("--format", choices=["text", "latex", "markdown"],
                        default="markdown")
    args = parser.parse_args()

    profiles = load_profiles(args.profiles)
    dt_profiles = load_profiles(args.dt_profiles)

    xla = get_xla_defaults(profiles)
    best_alt = get_best_per_module(profiles, exclude_xla=True)
    sa_best = get_sa_best(profiles)

    dt = {}
    for (mod, strat), val in dt_profiles.items():
        dt[mod] = val

    modules = sorted(xla.keys())

    # --- Table 1: Full comparison ---
    print("## Table 1: DT vs XLA Default vs SA vs Best Alternative")
    print()

    if args.format == "markdown":
        print("| Module | XLA | DT | DT% | SA | SA% | Best Alt | BA% |")
        print("|---|---|---|---|---|---|---|---|")
    elif args.format == "latex":
        print(r"\begin{tabular}{l r r r r r r r}")
        print(r"\toprule")
        print(r"Module & XLA & DT & DT\% & SA & SA\% & Best & Best\% \\")
        print(r"\midrule")

    dt_better = 0
    dt_within_1pct = 0
    dt_worse = 0
    dt_cat = 0

    for mod in modules:
        x = xla[mod]
        d = dt.get(mod, None)
        s = sa_best.get(mod, (None, None))[0]
        b = best_alt.get(mod, (None, None))[0]

        d_pct = ((d - x) / x * 100) if d else None
        s_pct = ((s - x) / x * 100) if s else None
        b_pct = ((b - x) / x * 100) if b else None

        if d_pct is not None:
            if d_pct < -0.1:
                dt_better += 1
            elif abs(d_pct) <= 1.0:
                dt_within_1pct += 1
            elif d_pct > 10:
                dt_cat += 1
            else:
                dt_worse += 1

        # Short module name — disambiguate duplicates
        short = mod.split("/")[-1]
        model_prefix = mod.split("/")[0].split("-")[0]
        # Check for duplicate short names across modules
        short_counts = sum(1 for m in modules if m.split("/")[-1] == short)
        if short_counts > 1:
            short = f"{model_prefix}/{short}"

        if args.format == "markdown":
            d_str = format_us(d) if d else "—"
            s_str = format_us(s) if s else "—"
            b_str = format_us(b) if b else "—"
            dp_str = f"{d_pct:+.1f}" if d_pct else "—"
            sp_str = f"{s_pct:+.1f}" if s_pct else "—"
            bp_str = f"{b_pct:+.1f}" if b_pct else "—"
            print(f"| {short} | {format_us(x)} | {d_str} | {dp_str} | {s_str} | {sp_str} | {b_str} | {bp_str} |")
        elif args.format == "latex":
            d_str = format_us(d) if d else "---"
            s_str = format_us(s) if s else "---"
            b_str = format_us(b) if b else "---"
            dp_str = f"{d_pct:+.1f}" if d_pct else "---"
            sp_str = f"{s_pct:+.1f}" if s_pct else "---"
            bp_str = f"{b_pct:+.1f}" if b_pct else "---"
            print(f"{short} & {format_us(x)} & {d_str} & {dp_str} & {s_str} & {sp_str} & {b_str} & {bp_str} \\\\")

    if args.format == "latex":
        print(r"\bottomrule")
        print(r"\end{tabular}")

    print()
    print(f"DT vs XLA: {dt_better} better, {dt_within_1pct} within 1%, "
          f"{dt_worse} moderately worse, {dt_cat} catastrophic (>10%)")

    # --- Table 2: Strategy ranking ---
    print()
    print("## Table 2: Strategy Rankings (avg rank across all modules)")
    print()

    # Compute per-module rankings
    all_strategies = set()
    mod_times = defaultdict(dict)
    for (mod, strat), val in profiles.items():
        all_strategies.add(strat)
        mod_times[mod][strat] = val
    # Add DT
    for mod, val in dt.items():
        mod_times[mod]["dt_greedy"] = val
    all_strategies.add("dt_greedy")

    # Rank per module
    strategy_ranks = defaultdict(list)
    for mod in modules:
        times = mod_times[mod]
        sorted_strats = sorted(times, key=lambda s: times[s])
        for rank, strat in enumerate(sorted_strats, 1):
            strategy_ranks[strat].append(rank)

    # Average rank
    avg_ranks = {}
    for strat, ranks in strategy_ranks.items():
        if len(ranks) >= len(modules) * 0.5:  # At least half of modules
            avg_ranks[strat] = sum(ranks) / len(ranks)

    print("| Rank | Strategy | Avg Rank | Modules |")
    print("|---|---|---|---|")
    sorted_ranks = sorted(avg_ranks.items(), key=lambda x: x[1])
    shown_dt = False
    shown_xla = False
    for i, (strat, avg) in enumerate(sorted_ranks[:15], 1):
        n = len(strategy_ranks[strat])
        marker = " **" if strat in ("xla_default", "dt_greedy") else ""
        print(f"| {i} | {strat}{marker} | {avg:.1f} | {n} |")
        if strat == "dt_greedy":
            shown_dt = True
        if strat == "xla_default":
            shown_xla = True

    # Always show DT and XLA if not in top 15
    for strat_key, label in [("dt_greedy", "dt_greedy **"), ("xla_default", "xla_default **")]:
        if strat_key in ("dt_greedy",) and shown_dt:
            continue
        if strat_key in ("xla_default",) and shown_xla:
            continue
        if strat_key in avg_ranks:
            rank = sum(1 for _, v in avg_ranks.items() if v < avg_ranks[strat_key]) + 1
            n = len(strategy_ranks[strat_key])
            print(f"| {rank} | {label} | {avg_ranks[strat_key]:.1f} | {n} |")
        elif strat_key in strategy_ranks:
            ranks = strategy_ranks[strat_key]
            avg = sum(ranks) / len(ranks)
            rank = sum(1 for _, v in avg_ranks.items() if v < avg) + 1
            print(f"| {rank} | {label} | {avg:.1f} | {len(ranks)} (<50%) |")

    # --- Statistics ---
    print()
    print("## Key Statistics")
    print()

    # Per-architecture summary
    arch_mods = defaultdict(list)
    for mod in modules:
        arch = mod.split("/")[0].split("-")[0]  # e.g., "Llama"
        arch_mods[arch].append(mod)

    print("### Per-Architecture DT Performance")
    print("| Architecture | Modules | Avg DT vs XLA |")
    print("|---|---|---|")
    for arch in sorted(arch_mods):
        mods = arch_mods[arch]
        pcts = []
        for m in mods:
            if m in dt and m in xla:
                pcts.append((dt[m] - xla[m]) / xla[m] * 100)
        if pcts:
            avg = sum(pcts) / len(pcts)
            print(f"| {arch} | {len(mods)} | {avg:+.2f}% |")

    # Oracle selection
    print()
    print("### Oracle Strategy Selection")
    oracle_total = 0
    xla_total = 0
    dt_total = 0
    sa_total = 0
    for mod in modules:
        x = xla[mod]
        d = dt.get(mod, x)
        b = best_alt.get(mod, (x, ""))[0]
        oracle = min(x, d, b)
        if mod in sa_best:
            oracle = min(oracle, sa_best[mod][0])
            sa_total += sa_best[mod][0]
        else:
            sa_total += x  # Use XLA for modules without SA
        oracle_total += oracle
        xla_total += x
        dt_total += d
    print(f"XLA total: {format_us(xla_total)}")
    print(f"DT total: {format_us(dt_total)} ({(dt_total - xla_total) / xla_total * 100:+.2f}%)")
    print(f"SA total (w/ XLA fill): {format_us(sa_total)} ({(sa_total - xla_total) / xla_total * 100:+.2f}%)")
    print(f"Oracle (best per module): {format_us(oracle_total)} ({(oracle_total - xla_total) / xla_total * 100:+.2f}%)")

    # DT wins summary
    print()
    print("### DT Wins (modules where DT < XLA)")
    for mod in modules:
        x = xla[mod]
        d = dt.get(mod, x)
        pct = (d - x) / x * 100
        if pct < -0.05:
            s = sa_best.get(mod, (None, None))[0]
            sa_str = f", SA {(s - x) / x * 100:+.1f}%" if s else ""
            short = mod.split("/")[-1]
            print(f"- {short}: DT {format_us(d)} vs XLA {format_us(x)} ({pct:+.1f}%{sa_str})")


if __name__ == "__main__":
    main()
