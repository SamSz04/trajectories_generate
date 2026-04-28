#!/usr/bin/env python3
"""Generate perturbed fusion plans from xla_default orderings.

For each module, reads the xla_default priority_fusion_dump.txt to get
the default producer ordering, then generates N perturbed plans by
randomly swapping K pairs. Output: JSON plan files for json_plan strategy.

Usage:
    python3 generate_perturbed_plans.py \
        --dump-base output/multi_dumps \
        --output-dir output/perturbed_plans \
        --num-plans 20 --num-swaps 3 \
        [--modules Llama-3-8B-sq4k-BF16-L1/ffn_layer ...]
"""

import argparse
import glob
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dump_parser import parse_dump, get_producer_ordering


def perturb_ordering(ordering, num_swaps=3):
    """Create a new ordering by swapping random pairs."""
    new_ordering = ordering.copy()
    n = len(new_ordering)
    if n < 2:
        return new_ordering
    for _ in range(num_swaps):
        i, j = random.sample(range(n), 2)
        new_ordering[i], new_ordering[j] = new_ordering[j], new_ordering[i]
    return new_ordering


def write_plan_file(ordering, plan_path, metadata=None):
    """Write a JSON plan file for the json_plan strategy."""
    plan = {"producer_ordering": ordering}
    if metadata:
        plan["metadata"] = metadata
    with open(plan_path, 'w') as f:
        json.dump(plan, f, indent=2)


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
            xla_dir = os.path.join(module_dir, "xla_default")
            if os.path.isdir(xla_dir):
                modules.append(f"{model_name}/{module_name}")
    return modules


def find_dump_file(dump_base, module):
    """Find priority_fusion_dump.txt for xla_default strategy."""
    xla_dir = os.path.join(dump_base, module, "xla_default")
    dump_path = os.path.join(xla_dir, "priority_fusion_dump.txt")
    if os.path.exists(dump_path):
        return dump_path
    # Fallback: search for any dump file
    for f in glob.glob(os.path.join(xla_dir, "*priority*dump*")):
        return f
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Generate perturbed fusion plans from xla_default orderings")
    parser.add_argument("--dump-base", required=True,
                        help="Base directory containing xla_default dumps")
    parser.add_argument("--output-dir", default="output/perturbed_plans",
                        help="Output directory for plan files")
    parser.add_argument("--num-plans", type=int, default=20,
                        help="Number of perturbed plans per module")
    parser.add_argument("--num-swaps", type=int, default=3,
                        help="Number of pair swaps per perturbation")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--modules", nargs="*",
                        help="Specific modules (default: all)")
    args = parser.parse_args()

    random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # Discover modules
    if args.modules:
        modules = args.modules
    else:
        modules = discover_modules(args.dump_base)

    print(f"Modules: {len(modules)}")
    print(f"Plans per module: {args.num_plans}")
    print(f"Swaps per plan: {args.num_swaps}")
    print(f"Total plans: {len(modules) * args.num_plans}")
    print()

    total_plans = 0
    for module in modules:
        dump_file = find_dump_file(args.dump_base, module)
        if not dump_file:
            print(f"[SKIP] {module}: no xla_default dump found")
            continue

        # Parse dump and get ordering
        result = parse_dump(dump_file)
        ordering_with_scores = get_producer_ordering(result)
        base_ordering = [name for name, _score in ordering_with_scores]

        if len(base_ordering) < 2:
            print(f"[SKIP] {module}: ordering too short ({len(base_ordering)})")
            continue

        # Create output directory for this module
        plan_key = module.replace("/", "__")
        module_plan_dir = os.path.join(args.output_dir, plan_key)
        os.makedirs(module_plan_dir, exist_ok=True)

        # Generate perturbed plans
        for i in range(args.num_plans):
            perturbed = perturb_ordering(base_ordering, num_swaps=args.num_swaps)
            plan_path = os.path.join(module_plan_dir, f"plan_perturb_{i}.json")
            write_plan_file(perturbed, plan_path, metadata={
                "type": "perturbed_plan",
                "base": "xla_default",
                "num_swaps": args.num_swaps,
                "index": i,
                "seed": args.seed,
                "num_producers": len(base_ordering),
            })
            total_plans += 1

        print(f"  {module}: {args.num_plans} plans "
              f"({len(base_ordering)} producers)")

    print(f"\nGenerated {total_plans} plan files in {args.output_dir}")


if __name__ == "__main__":
    main()
