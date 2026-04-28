"""Simulated Annealing orchestrator for fusion ordering search.

Uses the json_plan strategy in the patched XLA to evaluate candidate
orderings. Each SA iteration:
  1. Perturbs the current ordering (swap 2 random producers)
  2. Writes a JSON plan file
  3. Runs XLA compilation with DT_FUSION_STRATEGY=json_plan
  4. Parses the resulting dump for cost model scores
  5. Accepts/rejects via Metropolis criterion

Saves the top-N orderings as trajectories.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

# Add parent directory for dump_parser import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dump_parser import parse_dump, get_producer_ordering


@dataclass
class SAState:
    """State of the SA search."""
    ordering: List[str]  # producer names in priority order (descending)
    objective: float = 0.0  # sum(us_unfused - us_fused) from cost model
    dump_dir: Optional[str] = None


@dataclass
class SAResult:
    """Result of SA search: top-N orderings."""
    orderings: List[Tuple[List[str], float]] = field(default_factory=list)
    best_objective: float = 0.0
    iterations: int = 0
    acceptance_rate: float = 0.0


def run_xla_compilation(
    xla_tool: str,
    hlo_input: str,
    plan_file: str,
    dump_dir: str,
) -> Optional[float]:
    """Run XLA compilation with a json_plan strategy and return the objective.

    Returns sum(us_unfused - us_fused) from the dump, or None on failure.
    """
    env = os.environ.copy()
    env["DT_FUSION_STRATEGY"] = "json_plan"
    env["DT_FUSION_PLAN"] = plan_file
    env["XLA_FLAGS"] = f"--xla_dump_to={dump_dir} --xla_dump_hlo_as_text --xla_dump_hlo_pass_re=.*priority.fusion.*"

    try:
        result = subprocess.run(
            [xla_tool, "--platform=CUDA", "--reference_platform=", hlo_input],
            env=env,
            capture_output=True,
            text=True,
            timeout=120,  # 2 minute timeout per compilation
        )
    except subprocess.TimeoutExpired:
        print(f"  WARNING: Compilation timed out for {plan_file}")
        return None
    except FileNotFoundError:
        print(f"  ERROR: XLA tool not found at {xla_tool}")
        return None

    # Parse the dump (may have module prefix)
    dump_file = os.path.join(dump_dir, "priority_fusion_dump.txt")
    if not os.path.exists(dump_file):
        # Check for module-prefixed variant
        import glob
        candidates = glob.glob(os.path.join(dump_dir, "*priority_fusion_dump.txt"))
        if candidates:
            dump_file = candidates[0]
        else:
            print(f"  WARNING: No dump produced at {dump_dir}")
            if result.stderr:
                print(f"  stderr: {result.stderr[:500]}")
            return None

    parsed = parse_dump(dump_file)

    # Objective: sum(us_unfused - us_fused) over all fused steps
    # Higher = more speedup from fusion = better
    objective = sum(
        s.us_unfused - s.us_fused
        for s in parsed.steps
        if s.was_fused and s.us_unfused > 0
    )
    return objective


def write_plan_file(ordering: List[str], plan_path: str, metadata: dict = None):
    """Write a JSON plan file for the json_plan strategy."""
    plan = {
        "producer_ordering": ordering,
    }
    if metadata:
        plan["metadata"] = metadata
    Path(plan_path).write_text(json.dumps(plan, indent=2))


def perturb_ordering(ordering: List[str], num_swaps: int = 1) -> List[str]:
    """Create a new ordering by swapping random pairs."""
    new_ordering = ordering.copy()
    n = len(new_ordering)
    if n < 2:
        return new_ordering

    for _ in range(num_swaps):
        i, j = random.sample(range(n), 2)
        new_ordering[i], new_ordering[j] = new_ordering[j], new_ordering[i]

    return new_ordering


def get_initial_ordering(
    dump_path: str,
) -> List[str]:
    """Extract the default XLA ordering from an existing dump."""
    parsed = parse_dump(dump_path)
    ordering = get_producer_ordering(parsed)
    return [name for name, _ in ordering]


def run_sa_search(
    xla_tool: str,
    hlo_input: str,
    dump_base: str,
    initial_ordering: List[str],
    num_iterations: int = 100,
    initial_temp: float = 100.0,
    cooling_rate: float = 0.95,
    num_swaps: int = 1,
    seed: int = 42,
) -> SAResult:
    """Run simulated annealing to find good fusion orderings.

    Args:
        xla_tool: Path to the run_hlo_module binary.
        hlo_input: Path to the HLO input file.
        dump_base: Base directory for XLA dumps.
        initial_ordering: Starting producer ordering.
        num_iterations: Number of SA iterations.
        initial_temp: Starting temperature.
        cooling_rate: Temperature multiplier per iteration.
        num_swaps: Number of swaps per perturbation.
        seed: Random seed.

    Returns:
        SAResult with top orderings found.
    """
    random.seed(seed)

    # Working directory for SA intermediate files
    sa_dir = os.path.join(dump_base, f"sa_work_seed{seed}")
    os.makedirs(sa_dir, exist_ok=True)

    # Evaluate initial ordering
    plan_path = os.path.join(sa_dir, "plan_init.json")
    dump_dir = os.path.join(sa_dir, "dump_init")
    os.makedirs(dump_dir, exist_ok=True)

    write_plan_file(initial_ordering, plan_path, {"type": "sa_initial"})
    initial_obj = run_xla_compilation(xla_tool, hlo_input, plan_path, dump_dir)

    if initial_obj is None:
        print("ERROR: Failed to evaluate initial ordering")
        return SAResult()

    current = SAState(
        ordering=initial_ordering,
        objective=initial_obj,
        dump_dir=dump_dir,
    )
    best = SAState(
        ordering=initial_ordering.copy(),
        objective=initial_obj,
        dump_dir=dump_dir,
    )

    print(f"  Initial objective: {initial_obj:.2f}")

    # Track top orderings
    top_orderings: List[Tuple[List[str], float]] = [
        (initial_ordering.copy(), initial_obj)
    ]

    temp = initial_temp
    accepted = 0

    for iteration in range(num_iterations):
        # Perturb
        candidate_ordering = perturb_ordering(current.ordering, num_swaps)

        # Evaluate
        plan_path = os.path.join(sa_dir, f"plan_iter{iteration}.json")
        dump_dir = os.path.join(sa_dir, f"dump_iter{iteration}")
        os.makedirs(dump_dir, exist_ok=True)

        write_plan_file(candidate_ordering, plan_path, {
            "type": "sa_iteration",
            "iteration": iteration,
            "seed": seed,
        })
        candidate_obj = run_xla_compilation(
            xla_tool, hlo_input, plan_path, dump_dir
        )

        if candidate_obj is None:
            # Compilation failed — skip this iteration
            temp *= cooling_rate
            continue

        # Metropolis acceptance
        delta = candidate_obj - current.objective
        if delta > 0 or (temp > 0 and random.random() < math.exp(delta / temp)):
            current = SAState(
                ordering=candidate_ordering,
                objective=candidate_obj,
                dump_dir=dump_dir,
            )
            accepted += 1

            if candidate_obj > best.objective:
                best = SAState(
                    ordering=candidate_ordering.copy(),
                    objective=candidate_obj,
                    dump_dir=dump_dir,
                )

            # Track in top orderings
            top_orderings.append((candidate_ordering.copy(), candidate_obj))
        else:
            # Clean up rejected dump
            shutil.rmtree(dump_dir, ignore_errors=True)
            if os.path.exists(plan_path):
                os.remove(plan_path)

        temp *= cooling_rate

        if (iteration + 1) % 10 == 0:
            print(
                f"  Iter {iteration + 1}/{num_iterations}: "
                f"current={current.objective:.2f} "
                f"best={best.objective:.2f} "
                f"T={temp:.2f} "
                f"accepted={accepted}/{iteration + 1}"
            )

    # Sort by objective and keep unique orderings
    top_orderings.sort(key=lambda x: x[1], reverse=True)

    return SAResult(
        orderings=top_orderings,
        best_objective=best.objective,
        iterations=num_iterations,
        acceptance_rate=accepted / max(num_iterations, 1),
    )


def main():
    parser = argparse.ArgumentParser(
        description="SA search for fusion orderings via patched XLA"
    )
    parser.add_argument("--xla-tool", required=True, help="Path to run_hlo_module")
    parser.add_argument("--hlo-input", required=True, help="Path to HLO file")
    parser.add_argument("--dump-base", required=True, help="Base dump directory")
    parser.add_argument("--num-trajectories", type=int, default=10,
                        help="Number of SA trajectories to produce")
    parser.add_argument("--iterations", type=int, default=100,
                        help="SA iterations per restart")
    parser.add_argument("--initial-temp", type=float, default=100.0)
    parser.add_argument("--cooling-rate", type=float, default=0.95)
    parser.add_argument("--initial-dump", type=str, default=None,
                        help="Path to existing XLA default dump for initial ordering")
    args = parser.parse_args()

    # Get initial ordering from the XLA default dump
    if args.initial_dump:
        initial_dump_path = args.initial_dump
    else:
        # Try to find the XLA default dump in dump_base
        default_dump = os.path.join(
            args.dump_base, "xla_default", "priority_fusion_dump.txt"
        )
        if not os.path.exists(default_dump):
            # Check for module-prefixed variant
            import glob
            candidates = glob.glob(os.path.join(
                args.dump_base, "xla_default", "*priority_fusion_dump.txt"
            ))
            if candidates:
                default_dump = candidates[0]
        if os.path.exists(default_dump):
            initial_dump_path = default_dump
        else:
            print("ERROR: No initial dump found. Run xla_default strategy first,")
            print("or specify --initial-dump=<path>")
            sys.exit(1)

    print(f"Loading initial ordering from {initial_dump_path}")
    initial_ordering = get_initial_ordering(initial_dump_path)
    print(f"  {len(initial_ordering)} producers in ordering")

    # Run SA with multiple restarts, each with a different seed
    all_orderings: List[Tuple[List[str], float]] = []
    num_restarts = max(1, args.num_trajectories // 3 + 1)

    for restart in range(num_restarts):
        print(f"\n=== SA Restart {restart + 1}/{num_restarts} (seed={restart}) ===")
        result = run_sa_search(
            xla_tool=args.xla_tool,
            hlo_input=args.hlo_input,
            dump_base=args.dump_base,
            initial_ordering=initial_ordering,
            num_iterations=args.iterations,
            initial_temp=args.initial_temp,
            cooling_rate=args.cooling_rate,
            seed=restart,
        )
        all_orderings.extend(result.orderings)
        print(
            f"  Restart {restart + 1} done: best={result.best_objective:.2f}, "
            f"acceptance={result.acceptance_rate:.1%}"
        )

    # Select top-N unique orderings
    # Deduplicate by converting ordering to tuple
    seen = set()
    unique_orderings = []
    for ordering, obj in sorted(all_orderings, key=lambda x: x[1], reverse=True):
        key = tuple(ordering)
        if key not in seen:
            seen.add(key)
            unique_orderings.append((ordering, obj))
        if len(unique_orderings) >= args.num_trajectories:
            break

    print(f"\n=== Selected {len(unique_orderings)} SA trajectories ===")

    # Write final plans and run them through XLA to get proper dumps
    for idx, (ordering, obj) in enumerate(unique_orderings):
        plan_name = f"sa_{idx}"
        plan_path = os.path.join(args.dump_base, f"{plan_name}_plan.json")
        dump_dir = os.path.join(args.dump_base, plan_name)
        os.makedirs(dump_dir, exist_ok=True)

        write_plan_file(ordering, plan_path, {
            "type": "sa_final",
            "rank": idx,
            "objective": obj,
        })

        print(f"  [{idx + 1}/{len(unique_orderings)}] "
              f"Running SA trajectory {plan_name} (obj={obj:.2f})")

        final_obj = run_xla_compilation(
            args.xla_tool, args.hlo_input, plan_path, dump_dir
        )
        if final_obj is not None:
            print(f"    -> Final objective: {final_obj:.2f}")

    # Clean up SA working directories
    for restart in range(num_restarts):
        sa_work = os.path.join(args.dump_base, f"sa_work_seed{restart}")
        if os.path.isdir(sa_work):
            shutil.rmtree(sa_work, ignore_errors=True)

    print(f"\nSA complete. {len(unique_orderings)} trajectories saved.")


if __name__ == "__main__":
    main()
