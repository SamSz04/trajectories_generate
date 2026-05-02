"""Two-phase dump + nsys profile generation for all 44 modules × 145 strategies.

Discovers modules from HLO datasets directory, generates 140 env-var strategies
programmatically, classifies each as dump-only or dump+nsys based on existing
GPU profiles CSV, and runs in phases:
  Phase 1: Dump-only (parallel, strategies already profiled in CSV)
  Phase 2: Dump+nsys (serial, strategies needing GPU profiles)
  Phase 3: SA trajectories via sa_orchestrator.py (serial)

Supports multi-GPU execution via --gpu-id and --module-shard. Modules are
assigned to shards (not strategies), so all profiling for a module stays on
one GPU for consistency.

Usage:
    # Full run (all phases, single GPU)
    python3 generate_dumps_from_manifest.py

    # Multi-GPU: 3 GPUs across 2 servers
    # Server A (existing 80GB, 1 GPU):
    python3 generate_dumps_from_manifest.py --gpu-id 0 --module-shard 0/3

    # Server B (new 2×40GB):
    python3 generate_dumps_from_manifest.py --gpu-id 0 --module-shard 1/3
    python3 generate_dumps_from_manifest.py --gpu-id 1 --module-shard 2/3

    # Phase 1 only (parallel dump-only, 16 jobs)
    python3 generate_dumps_from_manifest.py --phase dump-only --jobs 16

    # Phase 2 only (serial dump+nsys)
    python3 generate_dumps_from_manifest.py --phase dump-nsys

    # Phase 3 only (SA, 5 trajectories per module)
    python3 generate_dumps_from_manifest.py --phase sa --sa-trajectories 5

    # Single model
    python3 generate_dumps_from_manifest.py --only-model Griffin-2B-BF16-L6

    # Dry run (show plan without executing)
    python3 generate_dumps_from_manifest.py --dry-run
"""

import argparse
import csv
import os
import re
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


# ======================================================================
# Strategy generation (140 env-var strategies)
# ======================================================================

PERTURBED_BASE = [(s, seed) for s in [0.1, 0.5, 1.0] for seed in [0, 1, 2]]
PERTURBED_EXPANDED = [
    (s, 0) for s in [0.05, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45,
                      0.6, 0.7, 0.8, 1.2, 1.5, 1.8, 2.0, 2.5,
                      3.5, 5.0, 8.0, 15.0]
]

# Known 44 modules (fallback when HLO base is unavailable)
KNOWN_MODULES = [
    ("DeepSeek-V3-BF16-L2", "DeepSeekV3Model_computation"),
    ("DeepSeek-V3-BF16-L2", "deepseek_ffn_layer"),
    ("DeepSeek-V3-BF16-L2", "deepseek_mla_layer"),
    ("DeepSeek-V3-BF16-L2", "gate_logit"),
    ("DeepSeek-V3-BF16-L2", "moe_permute"),
    ("DeepSeek-V3-BF16-L2", "moe_unpermute"),
    ("DeepSeek-V3-BF16-L2", "routed_block"),
    ("DeepSeek-V3-BF16-L2", "routed_mlp"),
    ("DeepSeek-V3-BF16-L2", "shared_block"),
    ("DeepSeek-V3-BF16-L2", "shared_block_mlp_down_proj"),
    ("DeepSeek-V3-BF16-L2", "shared_block_mlp_up_proj"),
    ("DeepSeek-V3-BF16-L2", "whole_computation"),
    ("Griffin-2B-BF16-L6", "GriffinModel_computation"),
    ("Griffin-2B-BF16-L6", "attention_block"),
    ("Griffin-2B-BF16-L6", "griffin_ffn"),
    ("Griffin-2B-BF16-L6", "recurrent_block"),
    ("Griffin-2B-BF16-L6", "rg_lru"),
    ("Griffin-2B-BF16-L6", "whole_computation"),
    ("Llama-3-8B-sq4k-BF16-L1", "Llama3Transformer_computation"),
    ("Llama-3-8B-sq4k-BF16-L1", "ffn_layer"),
    ("Llama-3-8B-sq4k-BF16-L1", "gqa_attention_dot"),
    ("Llama-3-8B-sq4k-BF16-L1", "gqa_ffn_layer"),
    ("Llama-3-8B-sq4k-BF16-L1", "gqa_layer"),
    ("Llama-3-8B-sq4k-BF16-L1", "gqa_out_proj"),
    ("Llama-3-8B-sq4k-BF16-L1", "gqa_qkv_proj"),
    ("Llama-3-8B-sq4k-BF16-L1", "mlp_down_proj"),
    ("Llama-3-8B-sq4k-BF16-L1", "mlp_up_proj"),
    ("Llama-3-8B-sq4k-BF16-L1", "whole_computation"),
    ("Mamba2-370M-BF16-L1", "Mamba2Model_computation"),
    ("Mamba2-370M-BF16-L1", "mamba_block"),
    ("Mamba2-370M-BF16-L1", "mamba_conv_ssm"),
    ("Mamba2-370M-BF16-L1", "whole_computation"),
    ("PaliGemma-3B-BF16-L1", "PaliGemmaModel_computation"),
    ("PaliGemma-3B-BF16-L1", "vision_attention"),
    ("PaliGemma-3B-BF16-L1", "vision_encoder"),
    ("PaliGemma-3B-BF16-L1", "vision_mlp"),
    ("PaliGemma-3B-BF16-L1", "whole_computation"),
    ("SigLIP-Base-BF16", "SigLIPModel_computation"),
    ("SigLIP-Base-BF16", "patch_embed"),
    ("SigLIP-Base-BF16", "siglip_image_encoder"),
    ("SigLIP-Base-BF16", "siglip_text_encoder"),
    ("SigLIP-Base-BF16", "vision_attention"),
    ("SigLIP-Base-BF16", "vision_mlp"),
    ("SigLIP-Base-BF16", "whole_computation"),
]


def generate_strategies():
    """Generate all 140 non-SA strategies as (name, env_dict) pairs."""
    strategies = []

    # xla_default (1)
    strategies.append(("xla_default", {}))

    # random_0..99 (100)
    for seed in range(100):
        strategies.append((
            "random_%d" % seed,
            {"DT_FUSION_STRATEGY": "random", "DT_FUSION_SEED": str(seed)},
        ))

    # perturbed: base (9) + expanded (20) = 29
    for sigma, seed in PERTURBED_BASE + PERTURBED_EXPANDED:
        strategies.append((
            "perturbed_%s_%d" % (sigma, seed),
            {"DT_FUSION_STRATEGY": "perturbed",
             "DT_FUSION_SIGMA": str(sigma),
             "DT_FUSION_SEED": str(seed)},
        ))

    # greedy heuristics (4)
    for h in ["fanout", "tensor_bytes", "flop_count", "reverse_topo"]:
        strategies.append((
            "greedy_%s" % h,
            {"DT_FUSION_STRATEGY": h},
        ))

    # random_walk: seeds 100-105 (6)
    for seed in range(100, 106):
        strategies.append((
            "greedy_random_walk_%d" % seed,
            {"DT_FUSION_STRATEGY": "random", "DT_FUSION_SEED": str(seed)},
        ))

    assert len(strategies) == 140, "Expected 140, got %d" % len(strategies)
    return strategies


# ======================================================================
# Module discovery
# ======================================================================

def discover_modules(hlo_base):
    """Discover all modules from HLO datasets directory.

    Returns list of (model, module, hlo_path) tuples.
    Falls back to KNOWN_MODULES list if hlo_base doesn't exist.
    """
    if not os.path.isdir(hlo_base):
        # Fallback for dry-run on local machine
        return [
            (m, mod, os.path.join(hlo_base, m, mod, "%s.hlo" % mod))
            for m, mod in KNOWN_MODULES
        ]

    modules = []
    for model in sorted(os.listdir(hlo_base)):
        model_dir = os.path.join(hlo_base, model)
        if not os.path.isdir(model_dir):
            continue
        for module in sorted(os.listdir(model_dir)):
            module_dir = os.path.join(model_dir, module)
            if not os.path.isdir(module_dir):
                continue
            hlo_path = os.path.join(module_dir, "%s.hlo" % module)
            if os.path.isfile(hlo_path):
                modules.append((model, module, hlo_path))

    return modules


# ======================================================================
# CSV operations
# ======================================================================

def load_profiled_set(csv_path):
    """Load set of (module_path, strategy) from GPU profiles CSV."""
    profiled = set()
    if not os.path.isfile(csv_path):
        return profiled

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            profiled.add((row["module"], row["strategy"]))

    return profiled


CSV_FIELDS = [
    "module", "strategy", "iterations", "total_kernel_ns",
    "avg_kernel_per_iter_ns", "avg_kernel_per_iter_us",
    "num_kernel_types", "total_kernel_instances", "wall_time_s",
]


def append_csv_row(csv_path, module_path, strategy, iterations,
                   total_kernel_ns, num_kernel_types, total_instances,
                   wall_time_s):
    """Append one row to the GPU profiles CSV."""
    avg_ns = total_kernel_ns / max(iterations, 1)
    row = {
        "module": module_path,
        "strategy": strategy,
        "iterations": iterations,
        "total_kernel_ns": total_kernel_ns,
        "avg_kernel_per_iter_ns": "%.1f" % avg_ns,
        "avg_kernel_per_iter_us": "%.1f" % (avg_ns / 1000.0),
        "num_kernel_types": num_kernel_types,
        "total_kernel_instances": total_instances,
        "wall_time_s": "%.1f" % wall_time_s,
    }

    write_header = not os.path.isfile(csv_path) or os.path.getsize(csv_path) == 0
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


# ======================================================================
# Dump helpers
# ======================================================================

def has_dump_in_dir(strat_dir):
    """Check if a priority_fusion_dump file exists in the directory."""
    if not os.path.isdir(strat_dir):
        return False
    for f in os.listdir(strat_dir):
        if "priority_fusion_dump" in f and f.endswith(".txt"):
            return True
    return False


def cleanup_strat_dir(strat_dir):
    """Remove non-essential files from a strategy directory.

    Keeps only: *priority_fusion_dump*, *before_priority-fusion*, xla_output.log,
                profile.nsys-rep
    Deletes: *.ll, *.ptx, *.pbtxt, *after_priority*, *after_spmd*, etc.
    """
    if not os.path.isdir(strat_dir):
        return 0
    keep_patterns = (
        "priority_fusion_dump",
        "before_priority-fusion",
        "xla_output.log",
        "profile.nsys-rep",
        "profile.sqlite",
    )
    removed = 0
    for f in os.listdir(strat_dir):
        fpath = os.path.join(strat_dir, f)
        if not os.path.isfile(fpath):
            continue
        if any(p in f for p in keep_patterns):
            continue
        try:
            os.remove(fpath)
            removed += 1
        except OSError:
            pass
    return removed


def symlink_dump(strat_dir):
    """Create standard symlink for module-prefixed dump file."""
    target = os.path.join(strat_dir, "priority_fusion_dump.txt")
    if os.path.exists(target):
        return
    for f in os.listdir(strat_dir):
        if ("priority_fusion_dump" in f and f.endswith(".txt")
                and f != "priority_fusion_dump.txt"):
            try:
                os.symlink(f, target)
            except OSError:
                pass
            return


# ======================================================================
# Phase 1: Dump-only runner
# ======================================================================

def run_dump_only(xla_tool, hlo_path, strat_dir, env_vars, timeout=600):
    """Run XLA to produce dump only (no nsys). Returns True on success."""
    os.makedirs(strat_dir, exist_ok=True)

    env = os.environ.copy()
    env.update(env_vars)
    env["XLA_FLAGS"] = (
        "--xla_dump_to=%s "
        "--xla_dump_hlo_as_text "
        "--xla_dump_hlo_pass_re=.*priority.fusion.*" % strat_dir
    )

    cmd = [xla_tool, "--platform=CUDA", "--reference_platform=", hlo_path]
    log_path = os.path.join(strat_dir, "xla_output.log")

    try:
        with open(log_path, "w") as log_f:
            subprocess.run(
                cmd, env=env, stdout=log_f, stderr=subprocess.STDOUT,
                timeout=timeout,
            )
    except subprocess.TimeoutExpired:
        return False
    except Exception:
        return False

    symlink_dump(strat_dir)
    cleanup_strat_dir(strat_dir)
    return has_dump_in_dir(strat_dir)


def _dump_only_worker(args):
    """Worker for parallel dump-only execution."""
    xla_tool, hlo_path, strat_dir, env_vars, strat_name, module_path = args

    if has_dump_in_dir(strat_dir):
        return (module_path, strat_name, True, "exists")

    ok = run_dump_only(xla_tool, hlo_path, strat_dir, env_vars)
    return (module_path, strat_name, ok, "ok" if ok else "failed")


# ======================================================================
# Phase 2: Dump + nsys runner
# ======================================================================

def run_dump_nsys(xla_tool, hlo_path, strat_dir, env_vars,
                  iterations=10, timeout=1200):
    """Run nsys-wrapped XLA to produce dump + profile.

    Returns (success, total_kernel_ns, n_types, n_instances, wall_s).
    """
    os.makedirs(strat_dir, exist_ok=True)
    nsys_output = os.path.join(strat_dir, "profile")

    env = os.environ.copy()
    env.update(env_vars)
    env["XLA_FLAGS"] = (
        "--xla_dump_to=%s "
        "--xla_dump_hlo_as_text "
        "--xla_dump_hlo_pass_re=.*priority.fusion.* "
        "--xla_gpu_enable_command_buffer= "
        "--xla_gpu_autotune_level=0" % strat_dir
    )

    cmd = [
        "nsys", "profile", "--stats=true", "--output=%s" % nsys_output,
        xla_tool, "--platform=CUDA", "--reference_platform=",
        "--iterations=%d" % iterations, hlo_path,
    ]

    log_path = os.path.join(strat_dir, "xla_output.log")
    t0 = time.time()

    try:
        with open(log_path, "w") as log_f:
            subprocess.run(
                cmd, env=env, stdout=log_f, stderr=subprocess.STDOUT,
                timeout=timeout,
            )
    except subprocess.TimeoutExpired:
        return (False, 0, 0, 0, time.time() - t0)
    except Exception:
        return (False, 0, 0, 0, time.time() - t0)

    wall_s = time.time() - t0
    symlink_dump(strat_dir)
    cleanup_strat_dir(strat_dir)

    nsys_rep = nsys_output + ".nsys-rep"
    if os.path.isfile(nsys_rep):
        total_ns, n_types, n_inst = parse_nsys_kernsum(nsys_rep)
        return (has_dump_in_dir(strat_dir), total_ns, n_types, n_inst, wall_s)

    return (has_dump_in_dir(strat_dir), 0, 0, 0, wall_s)


def parse_nsys_kernsum(nsys_rep_path):
    """Extract total GPU kernel time from nsys report.

    Returns (total_kernel_time_ns, num_kernel_types, num_instances).
    """
    try:
        for report_name in ["cuda_gpu_kern_sum", "gpukernsum"]:
            result = subprocess.run(
                ["nsys", "stats", "--report", report_name,
                 "--format", "csv", nsys_rep_path],
                capture_output=True, text=True, timeout=60,
            )
            if "could not be found" not in result.stderr:
                break

        total_ns = 0
        n_types = 0
        n_inst = 0

        lines = result.stdout.strip().split("\n")
        header_idx = -1
        for i, line in enumerate(lines):
            if '"Time (%)"' in line or "Time (%)" in line:
                header_idx = i
                break

        data_start = header_idx + 1 if header_idx >= 0 else 0
        for line in lines[data_start:]:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("="):
                continue
            parts = line.split(",")
            if len(parts) >= 3:
                try:
                    t = int(parts[1].strip().strip('"'))
                    inst = int(parts[2].strip().strip('"'))
                    total_ns += t
                    n_inst += inst
                    n_types += 1
                except (ValueError, IndexError):
                    continue

        return total_ns, n_types, n_inst
    except Exception:
        return 0, 0, 0


# ======================================================================
# Phase 3: SA runner
# ======================================================================

def run_sa_for_module(xla_tool, hlo_path, dump_dir, n_trajectories=5,
                      sa_iterations=100, script_dir=None):
    """Run sa_orchestrator.py for one module.

    Returns list of sa directory names produced (e.g., ['sa_0', ...]).
    """
    # Check existing
    existing = []
    for i in range(n_trajectories):
        if has_dump_in_dir(os.path.join(dump_dir, "sa_%d" % i)):
            existing.append("sa_%d" % i)
    if len(existing) == n_trajectories:
        return existing

    if script_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
    sa_script = os.path.join(script_dir, "sa_orchestrator.py")
    if not os.path.isfile(sa_script):
        print("  WARN: sa_orchestrator.py not found at %s" % sa_script)
        return existing

    cmd = [
        sys.executable, sa_script,
        "--xla-tool", xla_tool,
        "--hlo-input", hlo_path,
        "--dump-base", dump_dir,
        "--num-trajectories", str(n_trajectories),
        "--iterations", str(sa_iterations),
    ]

    log_path = os.path.join(dump_dir, "sa_orchestrator.log")
    try:
        with open(log_path, "w") as log_f:
            subprocess.run(
                cmd, stdout=log_f, stderr=subprocess.STDOUT,
                timeout=7200,  # 2h max
            )
    except subprocess.TimeoutExpired:
        print("  SA TIMEOUT")
    except Exception as e:
        print("  SA ERROR: %s" % e)

    produced = []
    for i in range(n_trajectories):
        if has_dump_in_dir(os.path.join(dump_dir, "sa_%d" % i)):
            produced.append("sa_%d" % i)
    return produced


def profile_sa_strategy(xla_tool, hlo_path, dump_dir, sa_name,
                        iterations=10, timeout=1200):
    """Profile an SA strategy with nsys using its saved plan file.

    Returns (success, total_kernel_ns, n_types, n_instances, wall_s).
    """
    plan_file = os.path.join(dump_dir, "%s_plan.json" % sa_name)
    sa_dir = os.path.join(dump_dir, sa_name)
    nsys_output = os.path.join(sa_dir, "profile")

    if not os.path.isfile(plan_file):
        return (False, 0, 0, 0, 0.0)

    env = os.environ.copy()
    env["DT_FUSION_STRATEGY"] = "json_plan"
    env["DT_FUSION_PLAN"] = plan_file
    env["XLA_FLAGS"] = (
        "--xla_gpu_enable_command_buffer= "
        "--xla_gpu_autotune_level=0"
    )

    cmd = [
        "nsys", "profile", "--stats=true", "--output=%s" % nsys_output,
        xla_tool, "--platform=CUDA", "--reference_platform=",
        "--iterations=%d" % iterations, hlo_path,
    ]

    log_path = os.path.join(sa_dir, "nsys_profile.log")
    t0 = time.time()
    try:
        with open(log_path, "w") as log_f:
            subprocess.run(
                cmd, env=env, stdout=log_f, stderr=subprocess.STDOUT,
                timeout=timeout,
            )
    except Exception:
        return (False, 0, 0, 0, time.time() - t0)

    wall_s = time.time() - t0
    nsys_rep = nsys_output + ".nsys-rep"
    if os.path.isfile(nsys_rep):
        total_ns, n_types, n_inst = parse_nsys_kernsum(nsys_rep)
        return (True, total_ns, n_types, n_inst, wall_s)

    return (False, 0, 0, 0, wall_s)


# ======================================================================
# Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate dumps + nsys profiles for all modules × strategies")
    parser.add_argument(
        "--xla-tool",
        default="/root/rivermind-data/xla/bazel-bin/xla/tools/run_hlo_module")
    parser.add_argument("--hlo-base", default="/root/hlo_datasets")
    parser.add_argument(
        "--dump-base",
        default="/root/rivermind-data/trajectories_generate/output/multi_dumps")
    parser.add_argument(
        "--gpu-csv",
        default="/root/rivermind-data/trajectories_generate/output/gpu_profiles_merged.csv")
    parser.add_argument(
        "--phase", default="all",
        choices=["all", "dump-only", "dump-nsys", "sa"],
        help="Which phase to run (default: all)")
    parser.add_argument(
        "--jobs", type=int, default=16,
        help="Parallel jobs for dump-only phase (default: 16)")
    parser.add_argument(
        "--sa-trajectories", type=int, default=5,
        help="SA trajectories per module (default: 5)")
    parser.add_argument(
        "--nsys-iterations", type=int, default=10,
        help="XLA iterations per nsys run (default: 10)")
    parser.add_argument("--only-model", default=None)
    parser.add_argument("--only-module", default=None)
    parser.add_argument(
        "--gpu-id", type=int, default=None,
        help="GPU index to use (sets CUDA_VISIBLE_DEVICES)")
    parser.add_argument(
        "--module-shard", default=None,
        help="Shard spec 'K/N': process shard K of N total (0-indexed). "
             "Splits modules (not strategies) so per-module profiling is "
             "consistent. E.g., --module-shard 0/3 for first of 3 GPUs")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--cleanup-existing", action="store_true",
        help="Clean up non-essential files (.ll, .ptx, .pbtxt, after-*) "
             "from existing dump dirs, then exit")
    parser.add_argument("--log-file", default=None)
    args = parser.parse_args()

    # Discover modules
    modules = discover_modules(args.hlo_base)
    if not modules:
        print("ERROR: No modules found in %s" % args.hlo_base)
        sys.exit(1)

    # Filter
    if args.only_model:
        modules = [(m, mod, p) for m, mod, p in modules
                   if m == args.only_model]
    if args.only_module:
        modules = [(m, mod, p) for m, mod, p in modules
                   if mod == args.only_module]

    # Apply module sharding (split by module, not strategy)
    total_modules_before_shard = len(modules)
    shard_k, shard_n = None, None
    if args.module_shard:
        parts = args.module_shard.split("/")
        if len(parts) != 2:
            print("ERROR: --module-shard must be K/N (e.g., 0/3)")
            sys.exit(1)
        shard_k, shard_n = int(parts[0]), int(parts[1])
        if shard_k < 0 or shard_k >= shard_n:
            print("ERROR: shard K must be in [0, N) (got %d/%d)" % (shard_k, shard_n))
            sys.exit(1)
        # Round-robin assignment: module i goes to shard (i % N)
        modules = [m for i, m in enumerate(modules) if i % shard_n == shard_k]

    # Set CUDA_VISIBLE_DEVICES for this process and all children
    if args.gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    # Cleanup mode: remove non-essential files from existing dumps
    if args.cleanup_existing:
        print("Cleaning up non-essential files from existing dumps...")
        total_removed = 0
        for model, module, hlo_path in modules:
            dump_dir = os.path.join(args.dump_base, model, module)
            if not os.path.isdir(dump_dir):
                continue
            for strat in os.listdir(dump_dir):
                strat_path = os.path.join(dump_dir, strat)
                if os.path.isdir(strat_path):
                    n = cleanup_strat_dir(strat_path)
                    total_removed += n
            print("  %s/%s: cleaned" % (model, module))
        print("Removed %d files total" % total_removed)
        return

    # Generate 140 env-var strategies
    strategies = generate_strategies()

    # Load existing CSV profiles
    profiled = load_profiled_set(args.gpu_csv)

    # Classify all (module, strategy) pairs
    dump_only_jobs = []   # (model, module, hlo_path, name, env, strat_dir)
    dump_nsys_jobs = []
    sa_modules = []       # (model, module, hlo_path)

    for model, module, hlo_path in modules:
        module_path = "%s/%s" % (model, module)
        dump_dir = os.path.join(args.dump_base, model, module)

        for strat_name, env_vars in strategies:
            strat_dir = os.path.join(dump_dir, strat_name)
            if (module_path, strat_name) in profiled:
                dump_only_jobs.append(
                    (model, module, hlo_path, strat_name, env_vars, strat_dir))
            else:
                dump_nsys_jobs.append(
                    (model, module, hlo_path, strat_name, env_vars, strat_dir))

        sa_modules.append((model, module, hlo_path))

    # Count SA needing nsys
    sa_nsys_needed = 0
    for model, module, _ in sa_modules:
        mp = "%s/%s" % (model, module)
        for i in range(args.sa_trajectories):
            if (mp, "sa_%d" % i) not in profiled:
                sa_nsys_needed += 1

    total_runs = (len(dump_only_jobs) + len(dump_nsys_jobs)
                  + len(sa_modules) * args.sa_trajectories)

    # ---- Summary ----
    print("=" * 60)
    print("Dump + Profile Generation")
    print("=" * 60)
    print("Modules:             %d" % len(modules))
    if shard_k is not None:
        print("  Shard:             %d/%d (%d of %d total modules)"
              % (shard_k, shard_n, len(modules), total_modules_before_shard))
    if args.gpu_id is not None:
        print("  GPU:               CUDA_VISIBLE_DEVICES=%d" % args.gpu_id)
    print("Strategies/module:   140 env-var + %d SA = %d"
          % (args.sa_trajectories, 140 + args.sa_trajectories))
    print("Phase 1 (dump-only): %d runs (%d parallel jobs)"
          % (len(dump_only_jobs), args.jobs))
    print("Phase 2 (dump+nsys): %d runs (serial)" % len(dump_nsys_jobs))
    print("Phase 3 (SA):        %d modules x %d traj"
          % (len(sa_modules), args.sa_trajectories))
    print("  SA nsys needed:    %d" % sa_nsys_needed)
    print("Total runs:          %d" % total_runs)
    print("XLA tool:            %s" % args.xla_tool)
    print("HLO base:            %s" % args.hlo_base)
    print("Dump base:           %s" % args.dump_base)
    print("GPU CSV:             %s" % args.gpu_csv)
    print("CSV entries loaded:  %d" % len(profiled))
    if args.dry_run:
        print("MODE:                DRY RUN")
    print("=" * 60)

    if args.dry_run:
        print("\nPer-model breakdown:")
        do_model = Counter(m for m, _, _, _, _, _ in dump_only_jobs)
        dn_model = Counter(m for m, _, _, _, _, _ in dump_nsys_jobs)
        all_models = sorted(set(list(do_model) + list(dn_model)))
        for m in all_models:
            n_mod = sum(1 for mm, _, _ in modules if mm == m)
            print("  %-30s %2d modules  %5d dump-only  %5d dump+nsys"
                  % (m, n_mod, do_model[m], dn_model[m]))

        if shard_k is not None:
            print("\nModules in this shard (%d/%d):" % (shard_k, shard_n))
            for model, module, _ in modules:
                print("  %s/%s" % (model, module))

        return

    # Validate XLA tool
    if not os.path.isfile(args.xla_tool):
        print("ERROR: XLA tool not found: %s" % args.xla_tool)
        sys.exit(1)

    os.makedirs(args.dump_base, exist_ok=True)
    log_file = args.log_file or os.path.join(args.dump_base, "generation_log.txt")
    csv_path = args.gpu_csv
    t_global = time.time()

    # ================================================================
    # Phase 1: Dump-only (parallel)
    # ================================================================
    if args.phase in ("all", "dump-only") and dump_only_jobs:
        print("\n" + "=" * 60)
        print("Phase 1: Dump-only (%d runs, %d parallel)"
              % (len(dump_only_jobs), args.jobs))
        print("=" * 60 + "\n")

        worker_args = [
            (args.xla_tool, hlo, sd, ev, sn, "%s/%s" % (m, mod))
            for m, mod, hlo, sn, ev, sd in dump_only_jobs
        ]

        done = 0
        skipped = 0
        failed = 0
        t1 = time.time()

        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            futures = {pool.submit(_dump_only_worker, wa): wa
                       for wa in worker_args}
            for fut in as_completed(futures):
                mp, sn, ok, status = fut.result()
                done += 1
                if status == "exists":
                    skipped += 1
                elif not ok:
                    failed += 1

                if done % 200 == 0 or done == len(worker_args):
                    el = time.time() - t1
                    rate = done / max(el, 0.1)
                    eta = (len(worker_args) - done) / max(rate, 0.01)
                    print("  Phase 1: %d/%d (exist=%d fail=%d) "
                          "[%.0fs elapsed, ~%.0fs ETA]"
                          % (done, len(worker_args), skipped, failed,
                             el, eta))

        dt1 = time.time() - t1
        new1 = done - skipped - failed
        print("\nPhase 1 done: %d new, %d existed, %d failed in %.0fs (%.1fh)"
              % (new1, skipped, failed, dt1, dt1 / 3600))
        with open(log_file, "a") as lf:
            lf.write("%s Phase1 %d runs %.0fs new=%d exist=%d fail=%d\n"
                     % (time.strftime("%Y-%m-%dT%H:%M:%SZ"), done, dt1,
                        new1, skipped, failed))

    # ================================================================
    # Phase 2: Dump+nsys (serial)
    # ================================================================
    if args.phase in ("all", "dump-nsys") and dump_nsys_jobs:
        print("\n" + "=" * 60)
        print("Phase 2: Dump+nsys (%d runs, serial)" % len(dump_nsys_jobs))
        print("=" * 60 + "\n")

        done = 0
        skipped = 0
        failed = 0
        profiled_count = 0
        t2 = time.time()

        for model, module, hlo_path, sn, ev, sd in dump_nsys_jobs:
            mp = "%s/%s" % (model, module)
            done += 1

            # Skip if both dump and profile exist
            if has_dump_in_dir(sd):
                nsys_rep = os.path.join(sd, "profile.nsys-rep")
                if os.path.isfile(nsys_rep):
                    skipped += 1
                    if done % 50 == 0:
                        el = time.time() - t2
                        print("  Phase 2: %d/%d (profiled=%d skip=%d fail=%d)"
                              % (done, len(dump_nsys_jobs), profiled_count,
                                 skipped, failed))
                    continue

            ok, total_ns, n_types, n_inst, wall_s = run_dump_nsys(
                args.xla_tool, hlo_path, sd, ev,
                iterations=args.nsys_iterations)

            if ok and total_ns > 0:
                append_csv_row(csv_path, mp, sn, args.nsys_iterations,
                               total_ns, n_types, n_inst, wall_s)
                profiled_count += 1
            elif not ok:
                failed += 1

            if done % 20 == 0 or done == len(dump_nsys_jobs):
                el = time.time() - t2
                rate = done / max(el, 0.1)
                eta = (len(dump_nsys_jobs) - done) / max(rate, 0.01)
                print("  Phase 2: %d/%d (profiled=%d fail=%d) "
                      "[%.0fs, ~%.1fh ETA]"
                      % (done, len(dump_nsys_jobs), profiled_count, failed,
                         el, eta / 3600))

        dt2 = time.time() - t2
        print("\nPhase 2 done: %d profiled, %d skipped, %d failed "
              "in %.0fs (%.1fh)"
              % (profiled_count, skipped, failed, dt2, dt2 / 3600))
        with open(log_file, "a") as lf:
            lf.write("%s Phase2 %d runs %.0fs profiled=%d skip=%d fail=%d\n"
                     % (time.strftime("%Y-%m-%dT%H:%M:%SZ"), done, dt2,
                        profiled_count, skipped, failed))

    # ================================================================
    # Phase 3: SA
    # ================================================================
    if args.phase in ("all", "sa") and sa_modules:
        print("\n" + "=" * 60)
        print("Phase 3: SA (%d modules x %d trajectories)"
              % (len(sa_modules), args.sa_trajectories))
        print("=" * 60 + "\n")

        t3 = time.time()
        total_produced = 0
        total_profiled_sa = 0

        for idx, (model, module, hlo_path) in enumerate(sa_modules):
            mp = "%s/%s" % (model, module)
            dump_dir = os.path.join(args.dump_base, model, module)

            print("  [%d/%d] %s" % (idx + 1, len(sa_modules), mp))

            # Run SA orchestrator
            produced = run_sa_for_module(
                args.xla_tool, hlo_path, dump_dir,
                n_trajectories=args.sa_trajectories)
            total_produced += len(produced)
            print("    SA dumps: %d/%d" % (len(produced), args.sa_trajectories))

            # Profile SA strategies not in CSV
            for sa_name in produced:
                if (mp, sa_name) in profiled:
                    continue

                ok, total_ns, n_types, n_inst, wall_s = profile_sa_strategy(
                    args.xla_tool, hlo_path, dump_dir, sa_name,
                    iterations=args.nsys_iterations)

                if ok and total_ns > 0:
                    append_csv_row(csv_path, mp, sa_name,
                                   args.nsys_iterations, total_ns,
                                   n_types, n_inst, wall_s)
                    total_profiled_sa += 1
                    avg_us = total_ns / args.nsys_iterations / 1000.0
                    print("    %s: %.1f us/iter" % (sa_name, avg_us))

        dt3 = time.time() - t3
        print("\nPhase 3 done: %d SA dumps, %d profiled in %.0fs (%.1fh)"
              % (total_produced, total_profiled_sa, dt3, dt3 / 3600))
        with open(log_file, "a") as lf:
            lf.write("%s Phase3 SA %d dumps %d profiled %.0fs\n"
                     % (time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                        total_produced, total_profiled_sa, dt3))

    # ================================================================
    # Final summary
    # ================================================================
    dt_total = time.time() - t_global
    print("\n" + "=" * 60)
    print("All phases complete in %.0fs (%.1fh)" % (dt_total, dt_total / 3600))
    print("=" * 60)

    # Count dumps on disk
    total_dumps = 0
    for root, dirs, files in os.walk(args.dump_base):
        for f in files:
            if "priority_fusion_dump" in f and f.endswith(".txt"):
                total_dumps += 1
    print("Total dump files on disk: %d" % total_dumps)

    # Count CSV entries
    if os.path.isfile(csv_path):
        with open(csv_path) as f:
            csv_count = sum(1 for _ in f) - 1
        print("Total CSV entries: %d" % csv_count)


if __name__ == "__main__":
    main()
