"""Evaluation and autoregressive generation for the fusion Decision Transformer.

Usage:
    python dt_eval.py --checkpoint output/dt_checkpoints/best.pt [--target-return 0.9]
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Tuple

import torch
import torch.nn.functional as F

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "references", "GO_fusion"))

from dt_dataset import load_trajectories, compute_returns_to_go, _compute_max_r0, FUSION_TOKEN_ID
from dt_model import DecisionTransformer


def generate_ordering(
    model: DecisionTransformer,
    node_embeds: torch.Tensor,
    target_return: float,
    max_steps: int = 30,
    temperature: float = 0.0,
) -> List[Tuple[int, float]]:
    """Autoregressive rollout: generate a fusion ordering.

    Args:
        model: Trained DecisionTransformer.
        node_embeds: Pre-computed node embeddings [N, D].
        target_return: Normalized target return-to-go (0 to 1).
        max_steps: Maximum fusion steps to generate.
        temperature: Sampling temperature (0 = greedy).

    Returns:
        List of (action_id, logit_value) tuples.
    """
    model.eval()
    device = node_embeds.device
    num_nodes = model.num_nodes

    ordering = []
    fused = set()
    rtg = target_return

    # History tensors (grow each step)
    all_rtg = []
    all_masks = []
    all_actions = []
    all_timesteps = []

    for step in range(max_steps):
        # Build current fused mask
        mask = [1.0 if i in fused else 0.0 for i in range(num_nodes)]

        all_rtg.append(rtg)
        all_masks.append(mask)
        all_timesteps.append(step)

        t = len(all_rtg)
        rtg_t = torch.tensor(all_rtg, dtype=torch.float32, device=device).unsqueeze(0)
        masks_t = torch.tensor(all_masks, dtype=torch.float32, device=device).unsqueeze(0)
        timesteps_t = torch.tensor(all_timesteps, dtype=torch.long, device=device).unsqueeze(0)

        # Actions so far (for steps 0..t-2, step t-1 action is what we're predicting)
        if all_actions:
            actions_t = torch.tensor(all_actions, dtype=torch.long, device=device).unsqueeze(0)
        else:
            actions_t = torch.zeros(1, 0, dtype=torch.long, device=device)

        action = model.predict_action(
            rtg_t, masks_t, actions_t, timesteps_t, node_embeds, temperature,
        )

        # Get logit value for this action
        with torch.no_grad():
            if actions_t.shape[1] < t:
                actions_t = torch.cat([
                    actions_t,
                    torch.zeros(1, 1, dtype=torch.long, device=device),
                ], dim=1)
            attn_mask = torch.ones(1, t, device=device)
            logits = model(rtg_t, masks_t, actions_t, timesteps_t, attn_mask, node_embeds)
            logit_val = logits[0, t - 1, action].item()

        ordering.append((action, logit_val))
        all_actions.append(action)

        if action < num_nodes:
            fused.add(action)

        # Decrease return-to-go (approximate: assume uniform reward per step)
        rtg = max(0.0, rtg - target_return / max_steps)

        # Stop if all eligible nodes are fused
        if len(fused) >= num_nodes:
            break

    return ordering


@torch.no_grad()
def evaluate_accuracy(
    model: DecisionTransformer,
    trajectories: List[dict],
    graph_data,
    max_r0: float,
) -> dict:
    """Evaluate action prediction accuracy on trajectories.

    Returns dict with overall accuracy, per-strategy accuracy, and top-k accuracy.
    """
    model.eval()
    device = next(model.parameters()).device
    node_embeds = model.encode_graph(graph_data.to(device))

    results = {}
    all_correct = 0
    all_total = 0

    for traj in trajectories:
        steps = traj["steps"]
        T = len(steps)
        if T == 0:
            continue

        strategy = traj.get("strategy_detail", traj.get("strategy", "unknown"))
        rtg = compute_returns_to_go(steps, normalize_by=max_r0)
        actions = [s["producer_id"] if s["producer_id"] >= 0 else FUSION_TOKEN_ID for s in steps]

        # Build fused masks
        fused_masks = []
        mask = [0.0] * model.num_nodes
        for t in range(T):
            fused_masks.append(mask.copy())
            a = actions[t]
            if a < model.num_nodes:
                mask[a] = 1.0

        # Run model on full trajectory
        rtg_t = torch.tensor(rtg, dtype=torch.float32, device=device).unsqueeze(0)
        masks_t = torch.tensor(fused_masks, dtype=torch.float32, device=device).unsqueeze(0)
        actions_t = torch.tensor(actions, dtype=torch.long, device=device).unsqueeze(0)
        timesteps_t = torch.arange(T, device=device).unsqueeze(0)
        attn_mask = torch.ones(1, T, device=device)

        logits = model(rtg_t, masks_t, actions_t, timesteps_t, attn_mask, node_embeds)
        preds = logits[0].argmax(dim=-1)  # [T]
        targets = actions_t[0]

        correct = (preds == targets).sum().item()
        all_correct += correct
        all_total += T

        results.setdefault(strategy, {"correct": 0, "total": 0})
        results[strategy]["correct"] += correct
        results[strategy]["total"] += T

    return {
        "overall_accuracy": all_correct / max(all_total, 1),
        "per_strategy": {
            k: v["correct"] / max(v["total"], 1) for k, v in sorted(results.items())
        },
        "total_steps": all_total,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate Decision Transformer")
    parser.add_argument("--checkpoint", default="output/dt_checkpoints/best.pt")
    parser.add_argument("--traj-path", default="output/trajectories_20260420_a100.pt")
    parser.add_argument("--target-return", type=float, default=0.9)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)

    # Load trajectories
    trajectories, graph_data = load_trajectories(args.traj_path)
    max_r0 = _compute_max_r0(trajectories)

    # Load model
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    info = ckpt["info"]

    model = DecisionTransformer(
        num_nodes=info["num_nodes"],
        context_len=info["context_len"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded checkpoint from epoch {ckpt['epoch']} "
          f"(val_loss={ckpt.get('val_loss', '?'):.4f})")

    # Encode graph
    graph_data = graph_data.to(device)
    node_embeds = model.encode_graph(graph_data)

    # 1. Evaluate accuracy on all trajectories
    print("\n=== Action Prediction Accuracy ===")
    acc_results = evaluate_accuracy(model, trajectories, graph_data, max_r0)
    print(f"Overall: {acc_results['overall_accuracy']:.4f} "
          f"({acc_results['total_steps']} steps)")

    print("\nPer strategy:")
    for strategy, acc in acc_results["per_strategy"].items():
        print(f"  {strategy:30s}: {acc:.4f}")

    # 2. Generate orderings at different target returns
    print("\n=== Generated Orderings ===")
    for target_r in [0.3, 0.5, 0.7, 0.9, 1.0]:
        ordering = generate_ordering(
            model, node_embeds, target_return=target_r,
            max_steps=30, temperature=args.temperature,
        )
        actions = [a for a, _ in ordering]
        n_regular = sum(1 for a in actions if a < model.num_nodes)
        n_fusion = sum(1 for a in actions if a == FUSION_TOKEN_ID)
        unique = len(set(a for a in actions if a < model.num_nodes))

        print(f"  target_R={target_r:.1f}: {len(ordering)} steps, "
              f"{n_regular} regular + {n_fusion} fusion.N, "
              f"{unique} unique nodes")

    # 3. Detailed ordering for the target return
    print(f"\n=== Detailed Ordering (target_R={args.target_return}) ===")
    ordering = generate_ordering(
        model, node_embeds, target_return=args.target_return,
        max_steps=30, temperature=args.temperature,
    )

    # Map node IDs to names if available
    node_names = getattr(graph_data, "node_names", None)
    for i, (action, logit) in enumerate(ordering):
        if action == FUSION_TOKEN_ID:
            name = "fusion.N"
        elif node_names and action < len(node_names):
            name = node_names[action]
        else:
            name = f"node_{action}"
        print(f"  Step {i:2d}: {name:30s} (id={action}, logit={logit:.2f})")


if __name__ == "__main__":
    main()
