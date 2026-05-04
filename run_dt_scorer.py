"""Integration script: DT scorer server + XLA profiling pipeline.

Orchestrates the full DT-guided fusion profiling workflow:
1. Start Python DT scorer server on a Unix socket
2. Launch XLA with DT_FUSION_STRATEGY=dt_server
3. Profile the after-fusion HLO with nsys
4. Compare with XLA default performance
5. Report results

Usage:
    PYTHONPATH=~/go_fusion python3 run_dt_scorer.py \
        --checkpoint output/dt_dynamic_checkpoints/best.pt \
        --hlo-dir ~/hlo_datasets/Llama-3-8B-sq4k-BF16-L1/gqa_layer/ \
        --xla-tool ~/xla/bazel-bin/xla/tools/run_hlo_module \
        --enriched-dir output/dynamic_features \
        --output output/dt_scorer_results.csv

    # Single module:
    PYTHONPATH=~/go_fusion python3 run_dt_scorer.py \
        --checkpoint output/dt_dynamic_checkpoints/best.pt \
        --hlo-file ~/hlo_datasets/.../gqa_layer.hlo \
        --enriched-pt output/dynamic_features/Llama-3-8B__gqa_layer.pt \
        --xla-tool ~/xla/bazel-bin/xla/tools/run_hlo_module \
        --output output/dt_scorer_results.csv
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
from typing import Dict, List, Optional, Tuple

# ======================================================================
# XLA runner
# ======================================================================

def run_xla_with_strategy(
    xla_tool: str,
    hlo_file: str,
    strategy: str,
    extra_env: Optional[Dict[str, str]] = None,
    timeout: int = 300,
) -> Tuple[bool, str, float]:
    """Run XLA binary with a fusion strategy.

    Returns (success, dump_dir, elapsed_seconds).
    """
    dump_dir = tempfile.mkdtemp(prefix=f"xla_{strategy}_")

    env = os.environ.copy()
    env["DT_FUSION_STRATEGY"] = strategy
    env["XLA_FLAGS"] = (
        f"--xla_dump_to={dump_dir} "
        "--xla_dump_hlo_pass_re=.*priority.fusion.* "
        "--xla_gpu_enable_priority_fusion=true"
    )
    if extra_env:
        env.update(extra_env)

    cmd = [
        xla_tool,
        f"--input_format=hlo",
        f"--platform=CUDA",
        f"--input_module={hlo_file}",
    ]

    t0 = time.time()
    try:
        result = subprocess.run(
            cmd, env=env, capture_output=True, text=True,
            timeout=timeout,
        )
        elapsed = time.time() - t0
        success = result.returncode == 0

        if not success:
            print(f"  XLA failed ({strategy}): {result.stderr[:200]}")

        return success, dump_dir, elapsed

    except subprocess.TimeoutExpired:
        return False, dump_dir, timeout


def run_nsys_profile(
    xla_tool: str,
    hlo_file: str,
    strategy: str = "",
    extra_env: Optional[Dict[str, str]] = None,
    num_iters: int = 5,
    timeout: int = 600,
) -> Optional[float]:
    """Profile HLO execution with nsys, return median kernel time in us.

    This runs the ALREADY-COMPILED HLO (after fusion) to measure
    actual GPU kernel execution time.
    """
    env = os.environ.copy()
    if strategy:
        env["DT_FUSION_STRATEGY"] = strategy
    if extra_env:
        env.update(extra_env)

    # Use run_hlo_module with multiple iterations for timing
    cmd = [
        xla_tool,
        f"--input_format=hlo",
        f"--platform=CUDA",
        f"--input_module={hlo_file}",
        f"--num_runs={num_iters}",
        "--print_execution_time=true",
    ]

    try:
        result = subprocess.run(
            cmd, env=env, capture_output=True, text=True,
            timeout=timeout,
        )
        if result.returncode != 0:
            print(f"  Profiling failed: {result.stderr[:200]}")
            return None

        # Parse execution time from output
        # Format: "Execution time: XXX us" or similar
        times = []
        for line in result.stdout.split("\n"):
            match = re.search(r"execution.*?(\d+(?:\.\d+)?)\s*(?:us|microsecond)",
                              line, re.IGNORECASE)
            if match:
                times.append(float(match.group(1)))

        if times:
            times.sort()
            median = times[len(times) // 2]
            return median

        # Fallback: try to parse wall time
        match = re.search(r"(\d+(?:\.\d+)?)\s*ms", result.stdout)
        if match:
            return float(match.group(1)) * 1000  # Convert ms to us

        print(f"  Could not parse timing from output")
        return None

    except subprocess.TimeoutExpired:
        print(f"  Profiling timed out")
        return None


# ======================================================================
# Module discovery
# ======================================================================

def find_modules(
    hlo_dir: str,
    enriched_dir: str,
    modules_filter: Optional[str] = None,
) -> List[Dict[str, str]]:
    """Discover modules available for profiling.

    Returns list of dicts with keys: module_key, hlo_file, enriched_pt.
    """
    # Find enriched .pt files
    enriched_files = {}
    if enriched_dir and os.path.isdir(enriched_dir):
        for pt_file in sorted(glob.glob(os.path.join(enriched_dir, "*.pt"))):
            name = os.path.basename(pt_file).replace(".pt", "")
            enriched_files[name] = pt_file

    # Find HLO files
    hlo_files = {}
    if hlo_dir and os.path.isdir(hlo_dir):
        for hlo_file in sorted(glob.glob(os.path.join(hlo_dir, "**/*.hlo"),
                                          recursive=True)):
            name = os.path.basename(hlo_file).replace(".hlo", "")
            hlo_files[name] = hlo_file

    # Match modules
    modules = []
    filter_set = None
    if modules_filter:
        filter_set = set(modules_filter.split(","))

    for name, enriched_pt in enriched_files.items():
        if filter_set and not any(f in name for f in filter_set):
            continue

        # Try to find matching HLO
        # Enriched names are like "Llama-3-8B__gqa_layer"
        # HLO names are like "gqa_layer"
        module_key = name.split("__")[-1] if "__" in name else name
        hlo_file = hlo_files.get(module_key, "")

        if "computation" in name.lower():
            continue  # Skip whole-computation modules

        modules.append({
            "module_key": name,
            "hlo_file": hlo_file,
            "enriched_pt": enriched_pt,
        })

    return modules


# ======================================================================
# Main pipeline
# ======================================================================

def profile_module(
    module: Dict[str, str],
    checkpoint: str,
    xla_tool: str,
    socket_path: str,
    device: str = "cuda",
    target_rtg: float = 1.5,
    context_len: int = 20,
    num_profile_iters: int = 5,
    gpu_csv_path: Optional[str] = None,
) -> Optional[Dict]:
    """Profile one module with DT-guided fusion vs XLA default.

    Steps:
    1. Start DT server
    2. Run XLA with dt_server strategy (DT-guided fusion)
    3. Run XLA with default strategy (XLA default)
    4. Profile both
    5. Compare

    Returns dict with results or None on failure.
    """
    module_key = module["module_key"]
    hlo_file = module.get("hlo_file", "")
    enriched_pt = module["enriched_pt"]

    if not hlo_file:
        print(f"  Skipping {module_key}: no HLO file found")
        return None

    print(f"\n{'='*60}")
    print(f"Module: {module_key}")
    print(f"  HLO: {hlo_file}")
    print(f"  Enriched: {enriched_pt}")
    print(f"{'='*60}")

    # 1. Start DT scorer server
    print("  Starting DT scorer server...")
    server_cmd = [
        sys.executable, "dt_scorer_server.py",
        "--checkpoint", checkpoint,
        "--enriched-pt", enriched_pt,
        "--socket", socket_path,
        "--device", device,
        "--target-rtg", str(target_rtg),
        "--context-len", str(context_len),
    ]

    server_env = os.environ.copy()
    go_fusion = os.environ.get("PYTHONPATH", "")
    if go_fusion:
        server_env["PYTHONPATH"] = go_fusion

    server_proc = subprocess.Popen(
        server_cmd,
        env=server_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Wait for server to start listening
    time.sleep(3)
    if server_proc.poll() is not None:
        stderr = server_proc.stderr.read().decode()
        print(f"  Server failed to start: {stderr[:300]}")
        return None

    try:
        # 2. Run XLA with dt_server strategy
        print("  Running XLA with DT-guided fusion...")
        dt_success, dt_dump_dir, dt_compile_time = run_xla_with_strategy(
            xla_tool, hlo_file, "dt_server",
            extra_env={"DT_FUSION_SCORER": socket_path},
        )

    finally:
        # Kill server
        server_proc.terminate()
        try:
            server_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server_proc.kill()

    # 3. Run XLA with default strategy
    print("  Running XLA with default fusion...")
    default_env = os.environ.copy()
    # No DT_FUSION_STRATEGY = XLA default
    default_env.pop("DT_FUSION_STRATEGY", None)
    default_success, default_dump_dir, default_compile_time = (
        run_xla_with_strategy(xla_tool, hlo_file, "", extra_env={})
    )

    # 4. Profile both (if compilation succeeded)
    dt_us = None
    default_us = None

    if dt_success:
        print("  Profiling DT-guided HLO...")
        dt_us = run_nsys_profile(
            xla_tool, hlo_file, "dt_server",
            extra_env={"DT_FUSION_SCORER": socket_path},
            num_iters=num_profile_iters,
        )

    if default_success:
        print("  Profiling XLA default HLO...")
        default_us = run_nsys_profile(
            xla_tool, hlo_file, "",
            num_iters=num_profile_iters,
        )

    # 5. Look up known best from GPU CSV
    best_strategy_us = None
    best_strategy_name = None
    if gpu_csv_path and os.path.isfile(gpu_csv_path):
        best_strategy_us, best_strategy_name = lookup_best_from_csv(
            gpu_csv_path, module_key)

    # Build result
    result = {
        "module": module_key,
        "dt_us": dt_us,
        "default_us": default_us,
        "best_strategy_us": best_strategy_us,
        "best_strategy": best_strategy_name,
        "dt_compile_s": round(dt_compile_time, 1) if dt_success else None,
        "default_compile_s": round(default_compile_time, 1) if default_success else None,
    }

    if dt_us and default_us:
        result["dt_speedup"] = round(default_us / dt_us, 4)
    if dt_us and best_strategy_us:
        result["vs_best"] = round(best_strategy_us / dt_us, 4)

    # Print summary
    print(f"\n  Results for {module_key}:")
    if dt_us:
        print(f"    DT-guided:   {dt_us:.1f} us")
    if default_us:
        print(f"    XLA default: {default_us:.1f} us")
    if dt_us and default_us:
        speedup = default_us / dt_us
        print(f"    Speedup:     {speedup:.3f}x")
    if best_strategy_us:
        print(f"    Best known:  {best_strategy_us:.1f} us ({best_strategy_name})")

    return result


def lookup_best_from_csv(
    csv_path: str,
    module_key: str,
) -> Tuple[Optional[float], Optional[str]]:
    """Look up best known strategy from GPU profiles CSV."""
    try:
        import csv as csv_mod
        best_us = None
        best_name = None
        with open(csv_path) as f:
            reader = csv_mod.DictReader(f)
            for row in reader:
                row_module = row.get("module", "")
                if module_key not in row_module:
                    continue
                try:
                    us = float(row.get("gpu_kernel_us", 0) or
                               row.get("kernel_time_us", 0))
                except (ValueError, TypeError):
                    continue
                if us > 0 and (best_us is None or us < best_us):
                    best_us = us
                    best_name = row.get("strategy", "unknown")
        return best_us, best_name
    except Exception:
        return None, None


def main():
    parser = argparse.ArgumentParser(
        description="DT scorer profiling pipeline")

    # Required
    parser.add_argument("--checkpoint", required=True,
                        help="Path to DT model checkpoint")
    parser.add_argument("--xla-tool", required=True,
                        help="Path to run_hlo_module binary")

    # Input (choose one)
    parser.add_argument("--hlo-dir", default=None,
                        help="Directory with HLO files (multi-module)")
    parser.add_argument("--hlo-file", default=None,
                        help="Single HLO file (single module)")
    parser.add_argument("--enriched-dir", default="output/dynamic_features",
                        help="Directory with enriched .pt files")
    parser.add_argument("--enriched-pt", default=None,
                        help="Single enriched .pt file")

    # Filtering
    parser.add_argument("--modules", default=None,
                        help="Comma-separated module names to profile")

    # Options
    parser.add_argument("--socket", default="/tmp/dt_scorer.sock")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--target-rtg", type=float, default=1.5)
    parser.add_argument("--context-len", type=int, default=20)
    parser.add_argument("--num-iters", type=int, default=5,
                        help="Number of profiling iterations")
    parser.add_argument("--gpu-csv", default="output/gpu_profiles_merged.csv",
                        help="GPU profiles CSV for baseline comparison")

    # Output
    parser.add_argument("--output", default="output/dt_scorer_results.csv",
                        help="Output CSV path")

    args = parser.parse_args()

    # Discover modules
    if args.hlo_file and args.enriched_pt:
        # Single module mode
        name = os.path.basename(args.enriched_pt).replace(".pt", "")
        modules = [{
            "module_key": name,
            "hlo_file": args.hlo_file,
            "enriched_pt": args.enriched_pt,
        }]
    else:
        modules = find_modules(
            args.hlo_dir or "",
            args.enriched_dir,
            args.modules,
        )

    if not modules:
        print("No modules found to profile")
        sys.exit(1)

    print(f"Found {len(modules)} modules to profile")

    # Profile each module
    results = []
    for module in modules:
        result = profile_module(
            module,
            checkpoint=args.checkpoint,
            xla_tool=args.xla_tool,
            socket_path=args.socket,
            device=args.device,
            target_rtg=args.target_rtg,
            context_len=args.context_len,
            num_profile_iters=args.num_iters,
            gpu_csv_path=args.gpu_csv,
        )
        if result:
            results.append(result)

    # Write results CSV
    if results:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        fieldnames = [
            "module", "dt_us", "default_us", "dt_speedup",
            "best_strategy_us", "best_strategy", "vs_best",
            "dt_compile_s", "default_compile_s",
        ]
        with open(args.output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames,
                                    extrasaction="ignore")
            writer.writeheader()
            writer.writerows(results)
        print(f"\nResults written to {args.output}")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    speedups = [r["dt_speedup"] for r in results
                if r.get("dt_speedup") is not None]
    if speedups:
        import statistics
        geo_mean = statistics.geometric_mean(speedups)
        print(f"Modules profiled: {len(results)}")
        print(f"Modules with speedup data: {len(speedups)}")
        print(f"Geometric mean speedup: {geo_mean:.3f}x")
        wins = sum(1 for s in speedups if s > 1.0)
        print(f"Wins (DT > XLA default): {wins}/{len(speedups)}")
    else:
        print("No speedup data collected")


if __name__ == "__main__":
    main()
