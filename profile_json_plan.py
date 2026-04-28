#!/usr/bin/env python3
"""Profile fusion orderings that use the json_plan strategy via nsys.

Handles SA trajectories, DT-generated orderings, and direct json_plan files.
Appends results to the same CSV format as profile_strategies.py.

Usage:
    # Profile all SA plans in dump directories
    python3 profile_json_plan.py --xla-tool ~/xla/bazel-bin/xla/tools/run_hlo_module \
        --dump-base output/multi_dumps --mode sa-dumps

    # Profile a single json_plan file
    python3 profile_json_plan.py --xla-tool ~/xla/bazel-bin/xla/tools/run_hlo_module \
        --json-plan path/to/plan.json --hlo-input path/to/before_optimizations.txt \
        --mode direct

    # Profile orderings extracted from a .pt trajectory file
    python3 profile_json_plan.py --xla-tool ~/xla/bazel-bin/xla/tools/run_hlo_module \
        --trajectory output/multi_trajectories/Llama-3-8B__gqa_layer.pt \
        --strategies sa_0 sa_1 --mode trajectory
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Reuse from profile_strategies.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from profile_strategies import parse_nsys_kernsum, find_hlo_input, load_existing_results


def discover_sa_plans(dump_base: str) -> List[dict]:
    """Discover all SA plan files across all modules in the dump directory.

    Returns list of dicts with: module, strategy, plan_path, hlo_input.
    """
    plans = []
    for model_dir in sorted(glob.glob(os.path.join(dump_base, "*"))):
        if not os.path.isdir(model_dir):
            continue
        model_name = os.path.basename(model_dir)

        for module_dir in sorted(glob.glob(os.path.join(model_dir, "*"))):
            if not os.path.isdir(module_dir):
                continue
            module_name = os.path.basename(module_dir)
            module_key = f"{model_name}/{module_name}"

            # Find SA plan files
            for plan_file in sorted(glob.glob(os.path.join(module_dir, "sa_*_plan.json"))):
                plan_name = os.path.basename(plan_file)
                # sa_0_plan.json -> sa_0
                strategy = plan_name.replace("_plan.json", "")

                # Find HLO input
                hlo_input = find_hlo_input(dump_base, module_key)
                if hlo_input is None:
                    print(f"  [SKIP] {module_key}/{strategy}: no HLO input")
                    continue

                plans.append({
                    "module": module_key,
                    "strategy": strategy,
                    "plan_path": plan_file,
                    "hlo_input": hlo_input,
                })

    return plans


def extract_ordering_from_trajectory(
    pt_path: str, strategy_detail: str
) -> Optional[List[str]]:
    """Extract producer_name ordering from a .pt trajectory file.

    Args:
        pt_path: Path to the .pt file.
        strategy_detail: Strategy name to extract (e.g., "sa_0", "random_5").

    Returns:
        List of producer names in fusion order, or None if not found.
    """
    import torch
    data = torch.load(pt_path, weights_only=False)

    for traj in data["trajectories"]:
        if traj.get("strategy_detail") == strategy_detail:
            return [step["producer_name"] for step in traj["steps"]]

    print(f"  [WARN] Strategy '{strategy_detail}' not found in {pt_path}")
    return None


def profile_json_plan(
    xla_tool: str,
    hlo_file: str,
    plan_path: str,
    strategy_name: str,
    iterations: int,
    nsys_dir: str,
) -> Optional[dict]:
    """Profile a single json_plan via nsys.

    Same pattern as profile_strategies.profile_one() but with
    DT_FUSION_STRATEGY=json_plan and DT_FUSION_PLAN set.
    """
    nsys_output = os.path.join(nsys_dir, strategy_name)

    env = os.environ.copy()
    env["DT_FUSION_STRATEGY"] = "json_plan"
    env["DT_FUSION_PLAN"] = os.path.abspath(plan_path)
    env["XLA_FLAGS"] = "--xla_gpu_enable_command_buffer= --xla_gpu_autotune_level=0"

    cmd = [
        "nsys", "profile",
        "--stats=false",
        "-o", nsys_output,
        "-f", "true",
        "--trace=cuda",
        xla_tool,
        "--platform=CUDA",
        "--reference_platform=",
        "--input_format=hlo",
        f"--iterations={iterations}",
        hlo_file,
    ]

    try:
        t0 = time.time()
        result = subprocess.run(
            cmd, env=env, capture_output=True, text=True, timeout=600
        )
        wall_time = time.time() - t0

        if result.returncode != 0:
            if "compiled and ran" not in result.stderr:
                print(f"  [FAIL] {strategy_name}: {result.stderr[-300:]}", file=sys.stderr)
                return None

        nsys_rep = nsys_output + ".nsys-rep"
        if not os.path.exists(nsys_rep):
            print(f"  [FAIL] No .nsys-rep for {strategy_name}", file=sys.stderr)
            return None

        total_ns, n_types, n_instances = parse_nsys_kernsum(nsys_rep)
        avg_per_iter_ns = total_ns / iterations if iterations > 0 else 0

        # Clean up nsys files
        for ext in [".nsys-rep", ".sqlite"]:
            f = nsys_output + ext
            if os.path.exists(f):
                os.remove(f)

        return {
            "total_kernel_ns": total_ns,
            "avg_kernel_ns": avg_per_iter_ns,
            "num_kernel_types": n_types,
            "total_kernel_instances": n_instances,
            "wall_time_s": wall_time,
        }

    except subprocess.TimeoutExpired:
        print(f"  [TIMEOUT] {strategy_name}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"  [ERROR] {strategy_name}: {e}", file=sys.stderr)
        return None


def write_csv_row(writer, module, strategy, iterations, result):
    """Write a single profiling result to CSV."""
    writer.writerow({
        "module": module,
        "strategy": strategy,
        "iterations": iterations,
        "total_kernel_ns": result["total_kernel_ns"],
        "avg_kernel_per_iter_ns": int(result["avg_kernel_ns"]),
        "avg_kernel_per_iter_us": f"{result['avg_kernel_ns'] / 1000:.1f}",
        "num_kernel_types": result["num_kernel_types"],
        "total_kernel_instances": result["total_kernel_instances"],
        "wall_time_s": f"{result['wall_time_s']:.1f}",
    })


def main():
    parser = argparse.ArgumentParser(
        description="Profile json_plan fusion orderings via nsys"
    )
    parser.add_argument("--xla-tool", required=True,
                        help="Path to run_hlo_module binary")
    parser.add_argument("--mode", required=True,
                        choices=["sa-dumps", "direct", "trajectory", "dt-plans"],
                        help="Profiling mode")
    parser.add_argument("--output", default="output/gpu_profiles_all.csv",
                        help="Output CSV path (appended)")
    parser.add_argument("--nsys-dir", default="/tmp/nsys_profiles",
                        help="Temp dir for nsys output")
    parser.add_argument("--iterations", type=int, default=10,
                        help="Number of execution iterations")

    # Mode: sa-dumps
    parser.add_argument("--dump-base", help="Base directory for HLO dumps")
    parser.add_argument("--modules", nargs="*",
                        help="Specific modules to profile (for sa-dumps mode)")

    # Mode: direct
    parser.add_argument("--json-plan", help="Path to json_plan file")
    parser.add_argument("--hlo-input", help="Path to HLO input file")
    parser.add_argument("--strategy-name", default="json_plan",
                        help="Strategy name for CSV (default: json_plan)")
    parser.add_argument("--module-name", default="unknown",
                        help="Module name for CSV")

    # Mode: trajectory
    parser.add_argument("--trajectory", help="Path to .pt trajectory file")
    parser.add_argument("--strategies", nargs="*",
                        help="Strategy names to extract from trajectory")

    # Mode: dt-plans
    parser.add_argument("--plans-dir",
                        help="Directory with DT json_plan files (dt-plans mode)")

    args = parser.parse_args()

    os.makedirs(args.nsys_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    # Resume support
    done = load_existing_results(args.output)
    if done:
        print(f"Resuming: {len(done)} already profiled")

    # Open CSV for appending
    fieldnames = [
        "module", "strategy", "iterations",
        "total_kernel_ns", "avg_kernel_per_iter_ns", "avg_kernel_per_iter_us",
        "num_kernel_types", "total_kernel_instances", "wall_time_s",
    ]
    write_header = not os.path.exists(args.output)
    csvfile = open(args.output, "a", newline="")
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    if write_header:
        writer.writeheader()

    completed = 0
    t_start = time.time()

    if args.mode == "sa-dumps":
        if not args.dump_base:
            parser.error("--dump-base required for sa-dumps mode")

        plans = discover_sa_plans(args.dump_base)
        if args.modules:
            plans = [p for p in plans if p["module"] in args.modules]

        print(f"Discovered {len(plans)} SA plan files")

        for plan in plans:
            key = (plan["module"], plan["strategy"])
            if key in done:
                print(f"  [SKIP] {key[0]}/{key[1]}: already profiled")
                continue

            module_nsys = os.path.join(
                args.nsys_dir,
                plan["module"].replace("/", "__") + "_" + plan["strategy"],
            )
            os.makedirs(os.path.dirname(module_nsys) or args.nsys_dir, exist_ok=True)

            print(f"  Profiling {plan['module']}/{plan['strategy']}...", end=" ", flush=True)
            result = profile_json_plan(
                args.xla_tool, plan["hlo_input"], plan["plan_path"],
                plan["strategy"], args.iterations,
                os.path.join(args.nsys_dir, plan["module"].replace("/", "__")),
            )

            if result:
                write_csv_row(writer, plan["module"], plan["strategy"],
                              args.iterations, result)
                csvfile.flush()
                completed += 1
                avg_us = result["avg_kernel_ns"] / 1000
                print(f"{avg_us:.1f} us ({result['wall_time_s']:.1f}s)")
            else:
                print("FAILED")

    elif args.mode == "direct":
        if not args.json_plan or not args.hlo_input:
            parser.error("--json-plan and --hlo-input required for direct mode")

        key = (args.module_name, args.strategy_name)
        if key in done:
            print(f"Already profiled: {key}")
        else:
            print(f"Profiling {args.module_name}/{args.strategy_name}...")
            result = profile_json_plan(
                args.xla_tool, args.hlo_input, args.json_plan,
                args.strategy_name, args.iterations, args.nsys_dir,
            )
            if result:
                write_csv_row(writer, args.module_name, args.strategy_name,
                              args.iterations, result)
                csvfile.flush()
                completed += 1
                avg_us = result["avg_kernel_ns"] / 1000
                print(f"  Result: {avg_us:.1f} us ({result['wall_time_s']:.1f}s)")
            else:
                print("  FAILED")

    elif args.mode == "trajectory":
        if not args.trajectory:
            parser.error("--trajectory required for trajectory mode")

        import torch
        data = torch.load(args.trajectory, weights_only=False)
        filename = os.path.basename(args.trajectory).replace(".pt", "")
        module_key = filename.replace("__", "/")

        # Need a json_plan file and HLO input for profiling
        # Try to find HLO from dump_base if provided
        if args.dump_base:
            hlo_input = find_hlo_input(args.dump_base, module_key)
        elif args.hlo_input:
            hlo_input = args.hlo_input
        else:
            parser.error("--dump-base or --hlo-input required for trajectory mode")
            return

        if not hlo_input:
            print(f"No HLO input found for {module_key}")
            return

        # Determine which strategies to extract
        available = [t.get("strategy_detail", "") for t in data["trajectories"]]
        if args.strategies:
            to_profile = args.strategies
        else:
            to_profile = [s for s in available if s]

        print(f"Module: {module_key}, HLO: {os.path.basename(hlo_input)}")
        print(f"Strategies to profile: {to_profile}")

        # Import write_plan_file for creating temp json_plan files
        from sa_orchestrator import write_plan_file

        for strategy in to_profile:
            key = (module_key, strategy)
            if key in done:
                print(f"  [SKIP] {strategy}: already profiled")
                continue

            ordering = extract_ordering_from_trajectory(args.trajectory, strategy)
            if ordering is None:
                continue

            # Write temporary plan file
            plan_path = os.path.join(args.nsys_dir, f"{strategy}_plan.json")
            write_plan_file(ordering, plan_path)

            module_nsys = os.path.join(
                args.nsys_dir, module_key.replace("/", "__")
            )
            os.makedirs(module_nsys, exist_ok=True)

            print(f"  Profiling {strategy} ({len(ordering)} steps)...", end=" ", flush=True)
            result = profile_json_plan(
                args.xla_tool, hlo_input, plan_path,
                strategy, args.iterations, module_nsys,
            )

            if result:
                write_csv_row(writer, module_key, strategy,
                              args.iterations, result)
                csvfile.flush()
                completed += 1
                avg_us = result["avg_kernel_ns"] / 1000
                print(f"{avg_us:.1f} us ({result['wall_time_s']:.1f}s)")
            else:
                print("FAILED")

            # Clean up temp plan
            if os.path.exists(plan_path):
                os.remove(plan_path)

    elif args.mode == "dt-plans":
        if not args.plans_dir or not args.dump_base:
            parser.error("--plans-dir and --dump-base required for dt-plans mode")

        # Discover DT plan files: <Model>__<module>_<strategy>.json
        plans = []
        for plan_file in sorted(glob.glob(os.path.join(args.plans_dir, "*.json"))):
            fname = os.path.basename(plan_file).replace(".json", "")
            # Parse: Model__module_strategyname → module_key = Model/module
            # Find last occurrence of _dt_ to split module from strategy
            dt_idx = fname.rfind("_dt_")
            if dt_idx < 0:
                print(f"  [SKIP] {fname}: no _dt_ in filename")
                continue
            module_part = fname[:dt_idx]
            strategy = fname[dt_idx + 1:]  # dt_greedy, dt_beam5_0, etc.
            module_key = module_part.replace("__", "/")

            hlo_input = find_hlo_input(args.dump_base, module_key)
            if hlo_input is None:
                print(f"  [SKIP] {module_key}/{strategy}: no HLO input")
                continue

            plans.append({
                "module": module_key,
                "strategy": strategy,
                "plan_path": plan_file,
                "hlo_input": hlo_input,
            })

        if args.modules:
            plans = [p for p in plans if p["module"] in args.modules]

        print(f"Discovered {len(plans)} DT plan files")

        for plan in plans:
            key = (plan["module"], plan["strategy"])
            if key in done:
                print(f"  [SKIP] {key[0]}/{key[1]}: already profiled")
                continue

            module_nsys = os.path.join(
                args.nsys_dir,
                plan["module"].replace("/", "__"),
            )
            os.makedirs(module_nsys, exist_ok=True)

            print(f"  Profiling {plan['module']}/{plan['strategy']}...", end=" ", flush=True)
            result = profile_json_plan(
                args.xla_tool, plan["hlo_input"], plan["plan_path"],
                plan["strategy"], args.iterations, module_nsys,
            )

            if result:
                write_csv_row(writer, plan["module"], plan["strategy"],
                              args.iterations, result)
                csvfile.flush()
                completed += 1
                avg_us = result["avg_kernel_ns"] / 1000
                print(f"{avg_us:.1f} us ({result['wall_time_s']:.1f}s)")
            else:
                print("FAILED")

    csvfile.close()
    elapsed = time.time() - t_start
    print(f"\nDone: {completed} profiled in {elapsed:.0f}s")


if __name__ == "__main__":
    main()
