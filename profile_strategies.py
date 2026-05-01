#!/usr/bin/env python3
"""
Batch GPU profiling of fusion strategies using nsys.

For each module × strategy, runs nsys + run_hlo_module and extracts
total GPU kernel execution time. Outputs CSV for analysis.

Usage:
    python3 profile_strategies.py --xla-tool ~/xla/bazel-bin/xla/tools/run_hlo_module \
        --dump-base output/multi_dumps --output output/gpu_profiles.csv \
        --iterations 10 [--modules Llama-3-8B-sq4k-BF16-L1/ffn_layer ...]
"""

import argparse
import csv
import glob
import os
import re
import signal
import subprocess
import sys
import time


# All env-var-controlled strategies (SA excluded — needs json plans)
def get_strategies(strategy_set="base"):
    """Returns list of (name, env_dict) tuples.

    Args:
        strategy_set: "base" (original 40), "expanded" (new 100),
                      "all" (140), or "plan" (json_plan from plan files only)
    """
    strategies = []

    if strategy_set in ("base", "all"):
        # XLA default (no env vars)
        strategies.append(("xla_default", {}))

        # Random (20 seeds)
        for seed in range(20):
            strategies.append((
                f"random_{seed}",
                {"DT_FUSION_STRATEGY": "random", "DT_FUSION_SEED": str(seed)}
            ))

        # Perturbed (3 sigmas × 3 seeds)
        for sigma in ["0.1", "0.5", "1.0"]:
            for seed in range(3):
                strategies.append((
                    f"perturbed_{sigma}_{seed}",
                    {"DT_FUSION_STRATEGY": "perturbed",
                     "DT_FUSION_SIGMA": sigma,
                     "DT_FUSION_SEED": str(seed)}
                ))

        # Greedy variants
        for variant in ["fanout", "tensor_bytes", "flop_count", "reverse_topo"]:
            strategies.append((
                f"greedy_{variant}",
                {"DT_FUSION_STRATEGY": variant}
            ))

        # Random walk (seeds 100-105)
        for seed in range(100, 106):
            strategies.append((
                f"greedy_random_walk_{seed}",
                {"DT_FUSION_STRATEGY": "random", "DT_FUSION_SEED": str(seed)}
            ))

    if strategy_set in ("expanded", "all"):
        # 80 more random seeds (random_20 through random_99)
        for seed in range(20, 100):
            strategies.append((
                f"random_{seed}",
                {"DT_FUSION_STRATEGY": "random", "DT_FUSION_SEED": str(seed)}
            ))

        # 20 more perturbed sigma values (seed=0)
        for sigma in [0.05, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.6, 0.7,
                      0.8, 1.2, 1.5, 1.8, 2.0, 2.5, 3.5, 5.0, 8.0, 15.0]:
            strategies.append((
                f"perturbed_{sigma}_0",
                {"DT_FUSION_STRATEGY": "perturbed",
                 "DT_FUSION_SIGMA": str(sigma),
                 "DT_FUSION_SEED": "0"}
            ))

    return strategies


def discover_modules(dump_base):
    """Discover all model/module pairs from dump directory."""
    modules = []
    for model_dir in sorted(glob.glob(os.path.join(dump_base, "*"))):
        if not os.path.isdir(model_dir):
            continue
        model_name = os.path.basename(model_dir)
        for module_dir in sorted(glob.glob(os.path.join(model_dir, "*"))):
            if not os.path.isdir(module_dir):
                continue
            module_name = os.path.basename(module_dir)
            # Check for xla_default dump to confirm validity
            xla_dir = os.path.join(module_dir, "xla_default")
            if os.path.isdir(xla_dir):
                modules.append(f"{model_name}/{module_name}")
    return modules


def find_hlo_input(dump_base, module_path):
    """Find the HLO input file for a module.

    Prefers original .hlo from ~/hlo_datasets/ (required for --with-dumps
    to produce priority_fusion_dump.txt). Falls back to before_optimizations
    from xla_default (works for nsys-only profiling).
    """
    # Prefer original HLO from hlo_datasets (produces correct fusion dumps)
    model, module = module_path.split("/", 1)
    hlo_datasets = os.path.join(os.path.expanduser("~"), "hlo_datasets", model, module)
    hlo_glob = glob.glob(os.path.join(hlo_datasets, "*.hlo"))
    if hlo_glob:
        return hlo_glob[0]
    # Fallback: before_optimizations from xla_default
    xla_dir = os.path.join(dump_base, module_path, "xla_default")
    pattern = os.path.join(xla_dir, "*before_optimizations*")
    matches = glob.glob(pattern)
    if matches:
        return matches[0]
    # Fallback: before_priority-fusion
    pattern = os.path.join(xla_dir, "*before_priority-fusion*")
    matches = glob.glob(pattern)
    return matches[0] if matches else None


def parse_nsys_kernsum(nsys_rep_path):
    """Extract total GPU kernel time from nsys report.

    Runs nsys stats and parses the gpukernsum CSV output.
    Returns (total_kernel_time_ns, num_kernel_types, num_instances).
    """
    try:
        # Try new nsys report name first (2024.x+), fall back to old name
        for report_name in ["cuda_gpu_kern_sum", "gpukernsum"]:
            result = subprocess.run(
                ["nsys", "stats", "--report", report_name,
                 "--format", "csv", nsys_rep_path],
                capture_output=True, text=True, timeout=60
            )
            if "could not be found" not in result.stderr:
                break
        output = result.stdout

        total_time_ns = 0
        num_kernel_types = 0
        num_instances = 0

        # Parse CSV output
        lines = output.strip().split('\n')
        header_idx = -1
        for i, line in enumerate(lines):
            if '"Time (%)"' in line or 'Time (%)' in line:
                header_idx = i
                break

        if header_idx < 0:
            # Try alternate parsing: look for numeric data lines
            for line in lines:
                line = line.strip()
                if not line or line.startswith('#') or line.startswith('='):
                    continue
                # CSV format: "Time (%)","Total Time (ns)","Instances",...,"Name"
                parts = line.split(',')
                if len(parts) >= 3:
                    try:
                        time_ns = int(parts[1].strip().strip('"'))
                        instances = int(parts[2].strip().strip('"'))
                        total_time_ns += time_ns
                        num_instances += instances
                        num_kernel_types += 1
                    except (ValueError, IndexError):
                        continue
        else:
            for line in lines[header_idx + 1:]:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(',')
                if len(parts) >= 3:
                    try:
                        time_ns = int(parts[1].strip().strip('"'))
                        instances = int(parts[2].strip().strip('"'))
                        total_time_ns += time_ns
                        num_instances += instances
                        num_kernel_types += 1
                    except (ValueError, IndexError):
                        continue

        return total_time_ns, num_kernel_types, num_instances

    except Exception as e:
        print(f"  [WARN] nsys stats parse failed: {e}", file=sys.stderr)
        return 0, 0, 0


def profile_one(xla_tool, hlo_file, strategy_name, strategy_env,
                iterations, nsys_dir, timeout=600, dump_dir=None):
    """Profile a single module × strategy combination.

    Returns dict with timing results, or None on failure.
    """
    nsys_output = os.path.join(nsys_dir, f"{strategy_name}")

    env = os.environ.copy()
    env.update(strategy_env)
    xla_flags = "--xla_gpu_enable_command_buffer= --xla_gpu_autotune_level=0"
    if dump_dir:
        os.makedirs(dump_dir, exist_ok=True)
        xla_flags += (f" --xla_dump_to={dump_dir} --xla_dump_hlo_as_text"
                      " --xla_dump_hlo_pass_re=.*priority.fusion.*")
    env["XLA_FLAGS"] = xla_flags

    cmd = [
        "nsys", "profile",
        "--stats=false",  # Don't print stats, we'll query separately
        "-o", nsys_output,
        "-f", "true",  # Overwrite existing
        "--trace=cuda",
        xla_tool,
        "--platform=CUDA",
        "--reference_platform=",
        "--input_format=hlo",
        f"--iterations={iterations}",
        hlo_file
    ]

    try:
        t0 = time.time()
        # Use start_new_session so we can kill the entire process group
        # (nsys + run_hlo_module + nsys-tee) on timeout
        proc = subprocess.Popen(
            cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, start_new_session=True,
        )
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            # Kill entire process group to avoid zombie children
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            proc.wait()
            print(f"  [TIMEOUT] {strategy_name}", file=sys.stderr)
            return None
        wall_time = time.time() - t0
        result = subprocess.CompletedProcess(
            cmd, proc.returncode, stdout, stderr
        )

        if result.returncode != 0:
            # Check if it ran but with non-zero exit (some XLA warnings)
            if "compiled and ran" not in result.stderr:
                print(f"  [FAIL] {strategy_name}: {result.stderr[-200:]}", file=sys.stderr)
                return None

        # Parse nsys report
        nsys_rep = nsys_output + ".nsys-rep"
        if not os.path.exists(nsys_rep):
            print(f"  [FAIL] No .nsys-rep for {strategy_name}", file=sys.stderr)
            return None

        total_ns, n_types, n_instances = parse_nsys_kernsum(nsys_rep)
        avg_per_iter_ns = total_ns / iterations if iterations > 0 else 0

        # Clean up nsys files to save disk space
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

    except Exception as e:
        print(f"  [ERROR] {strategy_name}: {e}", file=sys.stderr)
        return None


def load_existing_results(csv_path):
    """Load already-profiled (module, strategy) pairs to enable resume."""
    done = set()
    if os.path.exists(csv_path):
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                done.add((row["module"], row["strategy"]))
    return done


def main():
    parser = argparse.ArgumentParser(description="Batch GPU profiling of fusion strategies")
    parser.add_argument("--xla-tool", required=True, help="Path to run_hlo_module binary")
    parser.add_argument("--dump-base", required=True, help="Base directory for HLO dumps")
    parser.add_argument("--output", default="output/gpu_profiles.csv", help="Output CSV path")
    parser.add_argument("--nsys-dir", default="/tmp/nsys_profiles", help="Temp dir for nsys output")
    parser.add_argument("--iterations", type=int, default=10, help="Number of execution iterations")
    parser.add_argument("--modules", nargs="*", help="Specific modules to profile (default: all)")
    parser.add_argument("--strategies", nargs="*", help="Specific strategies (default: all)")
    parser.add_argument("--timeout", type=int, default=600,
                        help="Timeout per run in seconds (default: 600, use 1800+ for whole_computation)")
    parser.add_argument("--strategy-set", default="base",
                        choices=["base", "expanded", "all"],
                        help="Strategy set: base (40), expanded (100), all (140)")
    parser.add_argument("--with-dumps", action="store_true",
                        help="Enable XLA dump output during profiling")
    parser.add_argument("--plan-dir", default=None,
                        help="Dir with per-module plan files for json_plan strategies")
    args = parser.parse_args()

    os.makedirs(args.nsys_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    # Discover modules
    if args.modules:
        modules = args.modules
    else:
        modules = discover_modules(args.dump_base)
    print(f"Modules to profile: {len(modules)}")

    # Get strategies
    all_strategies = get_strategies(args.strategy_set)
    if args.strategies:
        all_strategies = [(n, e) for n, e in all_strategies if n in args.strategies]
    # Count plan strategies per module
    plan_count = 0
    if args.plan_dir:
        for module in modules:
            plan_key = module.replace("/", "__")
            plan_count += len(glob.glob(os.path.join(
                args.plan_dir, plan_key, "plan_*.json")))

    print(f"Strategies per module: {len(all_strategies)} env-var" +
          (f" + plans ({plan_count} total)" if plan_count else ""))
    print(f"Total runs: {len(modules) * len(all_strategies) + plan_count}")

    # Resume support
    done = load_existing_results(args.output)
    if done:
        print(f"Resuming: {len(done)} already profiled")

    # Open CSV for appending
    write_header = not os.path.exists(args.output)
    csvfile = open(args.output, 'a', newline='')
    fieldnames = [
        "module", "strategy", "iterations",
        "total_kernel_ns", "avg_kernel_per_iter_ns", "avg_kernel_per_iter_us",
        "num_kernel_types", "total_kernel_instances",
        "wall_time_s"
    ]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    if write_header:
        writer.writeheader()

    total_runs = len(modules) * len(all_strategies) + plan_count - len(done)
    completed = 0
    t_start = time.time()

    for module in modules:
        hlo_file = find_hlo_input(args.dump_base, module)
        if not hlo_file:
            print(f"\n[SKIP] {module}: no HLO input found")
            continue

        print(f"\n{'='*60}")
        print(f"Module: {module}")
        print(f"HLO: {os.path.basename(hlo_file)}")
        print(f"{'='*60}")

        # Create module-specific nsys dir
        module_nsys_dir = os.path.join(args.nsys_dir, module.replace("/", "__"))
        os.makedirs(module_nsys_dir, exist_ok=True)

        # Combine env-var strategies with module-specific plan strategies
        module_strategies = list(all_strategies)
        if args.plan_dir:
            plan_key = module.replace("/", "__")
            for pf in sorted(glob.glob(os.path.join(
                    args.plan_dir, plan_key, "plan_*.json"))):
                plan_name = os.path.splitext(os.path.basename(pf))[0]
                module_strategies.append((
                    plan_name,
                    {"DT_FUSION_STRATEGY": "json_plan",
                     "DT_FUSION_PLAN": os.path.abspath(pf)}
                ))

        for strat_name, strat_env in module_strategies:
            if (module, strat_name) in done:
                continue

            completed += 1
            elapsed = time.time() - t_start
            rate = elapsed / completed if completed > 0 else 0
            eta = rate * (total_runs - completed)
            print(f"  [{completed}/{total_runs}] {strat_name} "
                  f"(ETA: {eta/60:.0f}m)", end="", flush=True)

            dump_dir = None
            if args.with_dumps:
                dump_dir = os.path.join(args.dump_base, module, strat_name)

            result = profile_one(
                args.xla_tool, hlo_file, strat_name, strat_env,
                args.iterations, module_nsys_dir, timeout=args.timeout,
                dump_dir=dump_dir,
            )

            if result:
                row = {
                    "module": module,
                    "strategy": strat_name,
                    "iterations": args.iterations,
                    "total_kernel_ns": result["total_kernel_ns"],
                    "avg_kernel_per_iter_ns": result["avg_kernel_ns"],
                    "avg_kernel_per_iter_us": result["avg_kernel_ns"] / 1000,
                    "num_kernel_types": result["num_kernel_types"],
                    "total_kernel_instances": result["total_kernel_instances"],
                    "wall_time_s": f"{result['wall_time_s']:.1f}",
                }
                writer.writerow(row)
                csvfile.flush()
                avg_us = result["avg_kernel_ns"] / 1000
                print(f" → {avg_us:.0f} us, {result['num_kernel_types']} kernels")
            else:
                print(f" → FAILED")

            # Clean up non-module_0001 dump files to save disk space.
            # Keep only module_0001.* files; delete module_00NN.* for NN != 01.
            if dump_dir and os.path.isdir(dump_dir):
                for f in os.listdir(dump_dir):
                    if f.startswith("module_") and not f.startswith("module_0001"):
                        try:
                            os.remove(os.path.join(dump_dir, f))
                        except OSError:
                            pass

    csvfile.close()
    total_time = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"Done! {completed} runs in {total_time/60:.1f} minutes")
    print(f"Results: {args.output}")


if __name__ == "__main__":
    main()
