"""Collect all XLA dumps into final trajectory format.

After xla_runner.sh completes, this script:
1. Scans dump directories for priority_fusion_dump.txt files
2. Parses each dump via dump_parser.py
3. Builds the initial HLO graph (PyG Data) via GO_fusion's parser
4. Packages everything into the final trajectory format
5. Saves to output/trajectories.pt
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch

# Add current directory for dump_parser
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dump_parser import ParsedDump, parse_dump, get_fusion_sequence, summary

# Add GO_fusion for graph encoding
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "references", "GO_fusion"))

from src.hlo_parser.parser import parse_hlo_file
from src.hlo_parser.graph_builder import build_graph
from src.hlo_parser.feature_encoder import encode_features


def build_initial_graph(hlo_path: str):
    """Parse HLO and build PyG graph with encoded features.

    Returns:
        PyG Data object with x (num_nodes, 19), edge_index, node_names, etc.
    """
    module = parse_hlo_file(hlo_path)
    data = build_graph(module)
    data = encode_features(data)
    return data


def infer_strategy(dirname: str) -> str:
    """Infer the strategy name from the dump directory name."""
    if dirname == "xla_default":
        return "xla_default"
    if dirname.startswith("random_"):
        return "random"
    if dirname.startswith("perturbed_"):
        return "perturbed"
    if dirname.startswith("greedy_"):
        return dirname.replace("greedy_", "")
    if dirname.startswith("sa_"):
        return "simulated_annealing"
    return dirname


def build_name_to_id(data) -> Dict[str, int]:
    """Build mapping from node name to PyG node index."""
    return {name: idx for idx, name in enumerate(data.node_names)}


def build_trajectory(
    parsed: ParsedDump,
    data,
    name_to_id: Dict[str, int],
    strategy: str,
    dirname: str,
    hlo_path: str,
) -> dict:
    """Build a single trajectory dict from a parsed dump.

    Returns:
        Trajectory dict with steps, rewards, and metadata.
    """
    steps = []
    for s in parsed.steps:
        if not s.was_fused:
            continue  # only include fusion actions in trajectory

        producer_id = name_to_id.get(s.producer_name, -1)

        step_data = {
            "step_idx": len(steps),
            "producer_name": s.producer_name,
            "producer_id": producer_id,
            "consumer_names": s.consumer_names,
            "us_fused": s.us_fused,
            "us_unfused": s.us_unfused,
        }
        steps.append(step_data)

    # Cost model reward: ratio of total unfused to fused runtime
    total_fused = parsed.total_us_fused
    total_unfused = parsed.total_us_unfused
    reward_cost_model = total_unfused / total_fused if total_fused > 0 else 1.0

    trajectory = {
        "hlo_file": hlo_path,
        "hlo_module_name": parsed.hlo_module_name,
        "strategy": strategy,
        "strategy_detail": dirname,
        "steps": steps,
        "num_fusion_steps": len(steps),
        "total_us_fused": total_fused,
        "total_us_unfused": total_unfused,
        "reward_cost_model": reward_cost_model,
        "reward_real_gpu": None,  # filled in after GPU profiling
    }

    return trajectory


def collect_trajectories(
    dump_base: str,
    hlo_path: str,
    output_path: str,
):
    """Scan all dump directories and build trajectories.

    Args:
        dump_base: Directory containing per-strategy dump subdirectories.
        hlo_path: Path to the original HLO file.
        output_path: Path for the output .pt file.
    """
    dump_base = Path(dump_base)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Build initial graph once
    print(f"Building initial HLO graph from {hlo_path}...")
    data = build_initial_graph(hlo_path)
    name_to_id = build_name_to_id(data)
    print(f"  {data.num_nodes} nodes, {data.edge_index.shape[1]} edges, "
          f"features: {data.x.shape}")

    # Load GPU metadata if available
    gpu_metadata = {}
    gpu_meta_file = dump_base / "gpu_metadata.json"
    if gpu_meta_file.exists():
        with open(gpu_meta_file) as f:
            gpu_metadata = json.load(f)
        print(f"  GPU target: {gpu_metadata.get('gpu_name', 'unknown')}")

    # Scan dump directories
    trajectories = []
    errors = []

    dump_dirs = sorted([
        d for d in dump_base.iterdir()
        if d.is_dir() and not d.name.startswith("sa_work")
    ])

    print(f"\nProcessing {len(dump_dirs)} dump directories...")

    for dump_dir in dump_dirs:
        # Find priority_fusion_dump.txt (may have module prefix)
        dump_file = dump_dir / "priority_fusion_dump.txt"
        if not dump_file.exists():
            # Check for module-prefixed variant
            candidates = list(dump_dir.glob("*priority_fusion_dump.txt"))
            if candidates:
                dump_file = candidates[0]
            else:
                errors.append(f"  SKIP {dump_dir.name}: no dump file")
                continue

        try:
            parsed = parse_dump(str(dump_file))
            strategy = infer_strategy(dump_dir.name)

            traj = build_trajectory(
                parsed=parsed,
                data=data,
                name_to_id=name_to_id,
                strategy=strategy,
                dirname=dump_dir.name,
                hlo_path=hlo_path,
            )
            trajectories.append(traj)

            print(f"  {dump_dir.name}: {traj['num_fusion_steps']} fusions, "
                  f"reward={traj['reward_cost_model']:.4f}")

        except Exception as e:
            errors.append(f"  ERROR {dump_dir.name}: {e}")

    if errors:
        print(f"\nWarnings/errors:")
        for err in errors:
            print(err)

    # Save
    save_data = {
        "trajectories": trajectories,
        "initial_graph": data,
        "hlo_file": hlo_path,
        "num_trajectories": len(trajectories),
        "gpu_metadata": gpu_metadata,
        "metadata": {
            "hlo_module_name": data.hlo_module.name if hasattr(data, 'hlo_module') else "",
            "num_nodes": data.num_nodes,
            "feature_dim": data.x.shape[1] if data.x is not None else 0,
            "gpu_target": gpu_metadata.get("gpu_target", "unknown"),
        },
    }

    torch.save(save_data, str(output_path))
    print(f"\nSaved {len(trajectories)} trajectories to {output_path}")

    # Print summary
    _print_summary(trajectories)


def _print_summary(trajectories: List[dict]):
    """Print statistics about the collected trajectories."""
    if not trajectories:
        print("No trajectories collected.")
        return

    strategies = {}
    for t in trajectories:
        s = t["strategy"]
        strategies.setdefault(s, []).append(t)

    print(f"\n{'='*60}")
    print(f"Trajectory Summary")
    print(f"{'='*60}")
    print(f"Total trajectories: {len(trajectories)}")
    print(f"\nBy strategy:")

    rewards = [t["reward_cost_model"] for t in trajectories]

    for strategy, trajs in sorted(strategies.items()):
        r = [t["reward_cost_model"] for t in trajs]
        steps = [t["num_fusion_steps"] for t in trajs]
        print(f"  {strategy:25s}: {len(trajs):3d} trajs, "
              f"reward={min(r):.4f}-{max(r):.4f} (mean={sum(r)/len(r):.4f}), "
              f"steps={min(steps)}-{max(steps)}")

    print(f"\nOverall reward: min={min(rewards):.4f}, "
          f"max={max(rewards):.4f}, "
          f"mean={sum(rewards)/len(rewards):.4f}, "
          f"std={torch.tensor(rewards).std().item():.4f}")


def attach_rewards(
    trajectories_path: str,
    rewards_path: str,
    output_path: Optional[str] = None,
):
    """Attach real GPU profiling rewards to existing trajectories.

    Args:
        trajectories_path: Path to trajectories.pt
        rewards_path: Path to rewards.json (from GPU profiling)
        output_path: Output path (defaults to overwriting input)
    """
    if output_path is None:
        output_path = trajectories_path

    print(f"Loading trajectories from {trajectories_path}...")
    save_data = torch.load(trajectories_path, weights_only=False)
    trajectories = save_data["trajectories"]

    print(f"Loading rewards from {rewards_path}...")
    with open(rewards_path) as f:
        rewards = json.load(f)

    matched = 0
    for traj in trajectories:
        key = traj["strategy_detail"]
        if key in rewards:
            traj["reward_real_gpu"] = rewards[key]
            matched += 1

    print(f"Matched {matched}/{len(trajectories)} trajectories with GPU rewards")

    torch.save(save_data, output_path)
    print(f"Saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Collect XLA dumps into trajectory format"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command")

    # Collect command
    collect_parser = subparsers.add_parser("collect", help="Collect trajectories from dumps")
    collect_parser.add_argument("--dump-base", required=True, help="Dump directory base")
    collect_parser.add_argument("--hlo-input", required=True, help="Original HLO file")
    collect_parser.add_argument("--output", default="output/trajectories.pt",
                                help="Output file path")

    # Attach rewards command
    attach_parser = subparsers.add_parser("attach-rewards",
                                          help="Attach GPU profiling rewards")
    attach_parser.add_argument("--trajectories", required=True,
                               help="Path to trajectories.pt")
    attach_parser.add_argument("--rewards", required=True,
                               help="Path to rewards.json")
    attach_parser.add_argument("--output", default=None,
                               help="Output path (default: overwrite input)")

    args = parser.parse_args()

    if args.command == "collect":
        collect_trajectories(args.dump_base, args.hlo_input, args.output)
    elif args.command == "attach-rewards":
        attach_rewards(args.trajectories, args.rewards, args.output)
    else:
        # Default: collect mode with legacy flag interface
        parser.print_help()


if __name__ == "__main__":
    main()
