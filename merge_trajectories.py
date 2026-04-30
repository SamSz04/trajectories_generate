#!/usr/bin/env python3
"""Merge trajectory .pt files collected from multiple nodes.

When trajectories for the same module are collected on different nodes
(each having a different subset of strategy dumps), this script merges
them into a single .pt file per module.

Usage:
    python3 merge_trajectories.py \
        --input-dirs output/multi_trajectories_v2/a100-node \
                     output/multi_trajectories_v2/a100-node-1 \
                     ... \
        --output-dir output/multi_trajectories_merged
"""

import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path

import torch


def merge_module_files(pt_files: list, output_path: str):
    """Merge multiple .pt files for the same module.

    Deduplicates trajectories by strategy_detail (dirname).
    Keeps the first occurrence of each strategy.
    """
    all_trajectories = {}  # strategy_detail -> trajectory dict
    graph = None
    hlo_file = None
    gpu_metadata = {}
    metadata = {}

    for pt_file in pt_files:
        data = torch.load(pt_file, weights_only=False)

        # Use the graph from the first file
        if graph is None:
            graph = data.get("initial_graph")
            hlo_file = data.get("hlo_file")
            gpu_metadata = data.get("gpu_metadata", {})
            metadata = data.get("metadata", {})

        for traj in data.get("trajectories", []):
            key = traj["strategy_detail"]
            if key not in all_trajectories:
                all_trajectories[key] = traj

    # Sort by strategy_detail for consistency
    trajectories = [all_trajectories[k] for k in sorted(all_trajectories)]

    save_data = {
        "trajectories": trajectories,
        "initial_graph": graph,
        "hlo_file": hlo_file,
        "num_trajectories": len(trajectories),
        "gpu_metadata": gpu_metadata,
        "metadata": metadata,
    }

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    torch.save(save_data, output_path)

    return len(trajectories), len(pt_files)


def main():
    parser = argparse.ArgumentParser(description="Merge trajectory files from multiple nodes")
    parser.add_argument("--input-dirs", nargs="+", required=True,
                        help="Directories containing .pt files from each node")
    parser.add_argument("--output-dir", required=True,
                        help="Output directory for merged .pt files")
    args = parser.parse_args()

    # Group .pt files by module name
    module_files = defaultdict(list)
    for input_dir in args.input_dirs:
        input_dir = Path(input_dir)
        if not input_dir.exists():
            print(f"  SKIP: {input_dir} does not exist")
            continue
        for pt_file in input_dir.glob("*.pt"):
            module_name = pt_file.stem  # e.g., Llama-3-8B-sq4k-BF16-L1__gqa_layer
            module_files[module_name].append(str(pt_file))

    print(f"Found {len(module_files)} modules across {len(args.input_dirs)} input dirs")
    print()

    os.makedirs(args.output_dir, exist_ok=True)

    total_trajs = 0
    for module_name in sorted(module_files):
        pt_files = module_files[module_name]
        output_path = os.path.join(args.output_dir, f"{module_name}.pt")

        num_trajs, num_files = merge_module_files(pt_files, output_path)
        total_trajs += num_trajs
        print(f"  {module_name}: {num_trajs} trajectories (from {num_files} files)")

    print(f"\nTotal: {len(module_files)} modules, {total_trajs} trajectories")
    print(f"Output: {args.output_dir}/")


if __name__ == "__main__":
    main()
