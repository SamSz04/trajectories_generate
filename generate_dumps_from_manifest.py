"""Re-generate priority_fusion_dump.txt for all modules using exact manifest.

Reads dump_manifest.json (produced by strategy inventory extraction) and
runs XLA for each reproducible strategy-module pair. Skips plan_perturb,
plan_sa, and sa strategies (need special files). Resumable: skips existing dumps.

Usage:
    # Full run (all 33 modules, 3579 strategies)
    python3 generate_dumps_from_manifest.py

    # Single model
    python3 generate_dumps_from_manifest.py --only-model Griffin-2B-BF16-L6

    # Dry run (print commands without executing)
    python3 generate_dumps_from_manifest.py --dry-run
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path


# ======================================================================
# Strategy → env var mapping
# ======================================================================

def strategy_to_env(strategy_detail: str) -> dict:
    """Map a strategy directory name to XLA env vars.

    Returns dict of env vars to set, or None if not reproducible.
    """
    if strategy_detail == "xla_default":
        return {}

    # random_N → DT_FUSION_STRATEGY=random, DT_FUSION_SEED=N
    m = re.match(r"^random_(\d+)$", strategy_detail)
    if m:
        return {"DT_FUSION_STRATEGY": "random", "DT_FUSION_SEED": m.group(1)}

    # perturbed_SIGMA_SEED → DT_FUSION_STRATEGY=perturbed, DT_FUSION_SIGMA=SIGMA, DT_FUSION_SEED=SEED
    m = re.match(r"^perturbed_([\d.]+)_(\d+)$", strategy_detail)
    if m:
        return {
            "DT_FUSION_STRATEGY": "perturbed",
            "DT_FUSION_SIGMA": m.group(1),
            "DT_FUSION_SEED": m.group(2),
        }

    # greedy_STRATEGY → DT_FUSION_STRATEGY=STRATEGY
    m = re.match(r"^greedy_(fanout|tensor_bytes|flop_count|reverse_topo)$", strategy_detail)
    if m:
        return {"DT_FUSION_STRATEGY": m.group(1)}

    # greedy_random_walk_SEED → DT_FUSION_STRATEGY=random, DT_FUSION_SEED=SEED
    m = re.match(r"^greedy_random_walk_(\d+)$", strategy_detail)
    if m:
        return {"DT_FUSION_STRATEGY": "random", "DT_FUSION_SEED": m.group(1)}

    # plan_perturb_*, plan_sa_*, sa_* — not reproducible without extra files
    return None


# ======================================================================
# Module key → HLO path mapping
# ======================================================================

def module_key_to_paths(module_key: str, hlo_base: str, dump_base: str):
    """Map module key (e.g., 'Griffin-2B-BF16-L6__griffin_ffn') to HLO and dump paths."""
    parts = module_key.split("__", 1)
    if len(parts) != 2:
        return None, None
    model, module = parts
    hlo_path = os.path.join(hlo_base, model, module, f"{module}.hlo")
    dump_dir = os.path.join(dump_base, model, module)
    return hlo_path, dump_dir


# ======================================================================
# Run one XLA strategy
# ======================================================================

def run_strategy(
    xla_tool: str,
    hlo_input: str,
    dump_dir: str,
    strategy_detail: str,
    env_vars: dict,
    dry_run: bool = False,
) -> bool:
    """Run XLA with given strategy. Returns True if dump was produced."""
    strat_dir = os.path.join(dump_dir, strategy_detail)

    # Skip if dump already exists
    dump_file = os.path.join(strat_dir, "priority_fusion_dump.txt")
    if os.path.isfile(dump_file):
        return True

    # Also check for module-prefixed dump files
    if os.path.isdir(strat_dir):
        for f in os.listdir(strat_dir):
            if "priority_fusion_dump" in f and f.endswith(".txt"):
                return True

    if dry_run:
        env_str = " ".join(f"{k}={v}" for k, v in env_vars.items())
        print(f"  [DRY] {env_str} → {strat_dir}")
        return True

    os.makedirs(strat_dir, exist_ok=True)

    # Build environment
    env = os.environ.copy()
    env.update(env_vars)
    env["XLA_FLAGS"] = (
        f"--xla_dump_to={strat_dir} "
        f"--xla_dump_hlo_as_text "
        f"--xla_dump_hlo_pass_re=.*priority.fusion.*"
    )

    cmd = [xla_tool, "--platform=CUDA", "--reference_platform=", hlo_input]

    log_path = os.path.join(strat_dir, "xla_output.log")
    try:
        with open(log_path, "w") as log_f:
            subprocess.run(
                cmd, env=env, stdout=log_f, stderr=subprocess.STDOUT,
                timeout=600,  # 10 min max per run
            )
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT: {strategy_detail}")
        return False
    except Exception as e:
        print(f"  ERROR: {strategy_detail}: {e}")
        return False

    # Symlink dump file to standard name
    for f in os.listdir(strat_dir):
        if "priority_fusion_dump" in f and f.endswith(".txt") and f != "priority_fusion_dump.txt":
            try:
                os.symlink(f, dump_file)
            except OSError:
                pass
            return True

    return os.path.isfile(dump_file)


# ======================================================================
# Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser(description="Re-generate dumps from manifest")
    parser.add_argument("--manifest", default="output/dump_manifest.json",
                        help="Path to dump_manifest.json")
    parser.add_argument("--xla-tool",
                        default="/root/rivermind-data/xla/bazel-bin/xla/tools/run_hlo_module")
    parser.add_argument("--hlo-base", default="/root/hlo_datasets")
    parser.add_argument("--dump-base",
                        default="/root/rivermind-data/trajectories_generate/output/multi_dumps")
    parser.add_argument("--only-model", default=None,
                        help="Only process modules from this model")
    parser.add_argument("--only-module", default=None,
                        help="Only process this specific module key")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without executing")
    parser.add_argument("--log-file", default=None,
                        help="Log file path (default: <dump-base>/generation_log.txt)")
    args = parser.parse_args()

    # Load manifest
    with open(args.manifest) as f:
        manifest = json.load(f)

    if not os.path.isfile(args.xla_tool) and not args.dry_run:
        print(f"ERROR: XLA tool not found: {args.xla_tool}")
        sys.exit(1)

    os.makedirs(args.dump_base, exist_ok=True)
    log_file = args.log_file or os.path.join(args.dump_base, "generation_log.txt")

    # Count total work
    total_runs = 0
    modules_to_process = []
    for module_key, info in sorted(manifest.items()):
        if args.only_model and not module_key.startswith(args.only_model):
            continue
        if args.only_module and module_key != args.only_module:
            continue

        reproducible = info["reproducible"]
        if reproducible:
            modules_to_process.append((module_key, reproducible))
            total_runs += len(reproducible)

    print("=" * 60)
    print("Dump Generation from Manifest")
    print("=" * 60)
    print(f"Modules: {len(modules_to_process)}")
    print(f"Total reproducible runs: {total_runs}")
    print(f"XLA tool: {args.xla_tool}")
    print(f"Dump base: {args.dump_base}")
    if args.dry_run:
        print("MODE: DRY RUN")
    print("=" * 60)
    print()

    completed_runs = 0
    skipped_runs = 0
    failed_runs = 0
    t_global_start = time.time()

    for mod_idx, (module_key, strategies) in enumerate(modules_to_process):
        hlo_path, dump_dir = module_key_to_paths(
            module_key, args.hlo_base, args.dump_base)

        if hlo_path is None or (not os.path.isfile(hlo_path) and not args.dry_run):
            print(f"[{mod_idx+1}/{len(modules_to_process)}] SKIP {module_key} — HLO not found")
            failed_runs += len(strategies)
            continue

        # Count existing
        existing = 0
        if os.path.isdir(dump_dir):
            for sd in strategies:
                sd_path = os.path.join(dump_dir, sd)
                if os.path.isdir(sd_path):
                    for f in os.listdir(sd_path):
                        if "priority_fusion_dump" in f:
                            existing += 1
                            break

        print(f"[{mod_idx+1}/{len(modules_to_process)}] {module_key} "
              f"({existing}/{len(strategies)} existing, "
              f"{completed_runs}/{total_runs} total done)")
        t_mod_start = time.time()
        mod_new = 0

        for strat_detail in strategies:
            env_vars = strategy_to_env(strat_detail)
            if env_vars is None:
                skipped_runs += 1
                continue

            success = run_strategy(
                args.xla_tool, hlo_path, dump_dir,
                strat_detail, env_vars, args.dry_run)

            if success:
                completed_runs += 1
                mod_new += 1
            else:
                failed_runs += 1
                completed_runs += 1  # count for progress

        dt = time.time() - t_mod_start
        print(f"  -> {mod_new} new dumps in {dt:.0f}s")

        # Log
        if not args.dry_run:
            with open(log_file, "a") as lf:
                lf.write(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} "
                         f"{module_key} {len(strategies)} strategies {dt:.0f}s\n")

    dt_total = time.time() - t_global_start
    print()
    print("=" * 60)
    print(f"Complete in {dt_total:.0f}s ({dt_total/3600:.1f}h)")
    print(f"Completed: {completed_runs}, Skipped: {skipped_runs}, Failed: {failed_runs}")
    print("=" * 60)

    # Final count
    if not args.dry_run:
        total_dumps = 0
        for root, dirs, files in os.walk(args.dump_base):
            for f in files:
                if "priority_fusion_dump" in f and f.endswith(".txt"):
                    total_dumps += 1
        print(f"Total dump files on disk: {total_dumps}")


if __name__ == "__main__":
    main()
