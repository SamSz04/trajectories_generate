"""Generate fusion orderings from trained priority scorer for XLA profiling.

Loads the trained scorer checkpoint, scores all fusable original nodes at
step 0 (initial state, no fusion.N clusters yet), and outputs sorted
orderings as json_plan files compatible with the XLA binary.

Usage:
    # Generate orderings for all modules
    python scorer_eval.py generate \
        --checkpoint output/scorer_checkpoints/best.pt \
        --data-dir output/dynamic_features \
        --output-dir output/scorer_plans

    # Profile generated orderings on GPU
    python profile_json_plan.py --xla-tool ~/xla/.../run_hlo_module \
        --mode dt-plans --plans-dir output/scorer_plans \
        --dump-base output/multi_dumps --output output/scorer_profiles.csv
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scorer_model import PriorityScorer, build_scorer, ORIGINAL_FEATURE_DIM
from cluster_features import CLUSTER_FEATURE_DIM


def generate_ordering(
    model: PriorityScorer,
    graph,
    device: torch.device,
) -> List[str]:
    """Score all fusable original nodes and return sorted ordering.

    Uses step-0 context (no fusions yet, all nodes alive) to score
    every fusable node. Returns node names sorted by score descending
    (highest priority first).
    """
    model.eval()

    # Identify fusable nodes
    fusable_indices = []
    fusable_names = []
    for idx in range(graph.num_nodes):
        if graph.x[idx, 18].item() > 0.5:  # _FEAT_IS_FUSABLE = 18
            fusable_indices.append(idx)
            fusable_names.append(graph.node_names[idx])

    if not fusable_indices:
        return []

    N = len(fusable_indices)

    # Build input tensors (batch size 1)
    orig_features = graph.x[fusable_indices].unsqueeze(0)       # [1, N, 19]
    orig_opcode_ids = graph.opcode_ids[fusable_indices].unsqueeze(0)  # [1, N]
    orig_mask = torch.ones(1, N, dtype=torch.bool)

    # No clusters at step 0
    cluster_features = torch.zeros(1, 1, CLUSTER_FEATURE_DIM)
    cluster_mask = torch.zeros(1, 1, dtype=torch.bool)

    # Step-0 context: beginning of fusion, all nodes alive
    context = torch.tensor([[
        0.0,                              # step_progress = 0
        0.0,                              # frac_fused = 0
        N / max(graph.num_nodes, 1),      # frac_alive
        0.0,                              # frac_fusion_cands = 0
        0.0,                              # log2(n_fusions + 1) = 0
    ]], dtype=torch.float32)

    # Move to device
    orig_features = orig_features.to(device)
    orig_opcode_ids = orig_opcode_ids.to(device)
    orig_mask = orig_mask.to(device)
    cluster_features = cluster_features.to(device)
    cluster_mask = cluster_mask.to(device)
    context = context.to(device)

    with torch.no_grad():
        scores = model(
            orig_features=orig_features,
            orig_opcode_ids=orig_opcode_ids,
            orig_mask=orig_mask,
            cluster_features=cluster_features,
            cluster_mask=cluster_mask,
            context=context,
        )  # [1, N + 1]

    # Extract original node scores (first N positions)
    node_scores = scores[0, :N].cpu().tolist()

    # Sort by score descending
    scored = sorted(zip(fusable_names, node_scores),
                    key=lambda x: x[1], reverse=True)

    return [name for name, _ in scored]


def generate_all(
    checkpoint_path: str,
    data_dir: str,
    output_dir: str,
    device_str: str = 'cpu',
):
    """Generate scorer orderings for all modules."""
    device = torch.device(device_str)
    output_dir_p = Path(output_dir)
    output_dir_p.mkdir(parents=True, exist_ok=True)

    # Load model
    model = build_scorer()
    ckpt = torch.load(checkpoint_path, weights_only=False, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(device)
    model.eval()

    epoch = ckpt.get('epoch', '?')
    val_metrics = ckpt.get('val_metrics', {})
    print(f"Loaded checkpoint: epoch {epoch}, "
          f"val_acc={val_metrics.get('top1_acc', 0):.4f}")

    # Process each enriched .pt file
    data_dir_p = Path(data_dir)
    pt_files = sorted(data_dir_p.glob("*.pt"))
    pt_files = [f for f in pt_files if f.stem != 'summary']

    print(f"Generating orderings for {len(pt_files)} modules...")

    results = []
    for pt_file in pt_files:
        module_key = pt_file.stem
        data = torch.load(str(pt_file), weights_only=False)
        graph = data['graph']

        ordering = generate_ordering(model, graph, device)

        if not ordering:
            print(f"  SKIP {module_key}: no fusable nodes")
            continue

        # Save as json_plan
        plan = {"producer_ordering": ordering}
        plan_path = output_dir_p / f"{module_key}_scorer.json"
        plan_path.write_text(json.dumps(plan, indent=2))

        results.append({
            'module': module_key,
            'n_nodes': graph.num_nodes,
            'n_fusable': len(ordering),
            'plan_path': str(plan_path),
        })
        print(f"  {module_key}: {len(ordering)} nodes -> {plan_path.name}")

    print(f"\nGenerated {len(results)} orderings in {output_dir_p}")
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Generate fusion orderings from trained scorer")
    sub = parser.add_subparsers(dest="command")

    p1 = sub.add_parser("generate", help="Generate orderings for all modules")
    p1.add_argument("--checkpoint", required=True,
                    help="Path to scorer checkpoint (best.pt)")
    p1.add_argument("--data-dir", required=True,
                    help="Directory with enriched .pt files")
    p1.add_argument("--output-dir", default="output/scorer_plans")
    p1.add_argument("--device", default="cpu")

    args = parser.parse_args()

    if args.command == "generate":
        generate_all(
            checkpoint_path=args.checkpoint,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            device_str=args.device,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
