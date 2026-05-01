#!/usr/bin/env python3
"""Multi-module Decision Transformer evaluation with beam search.

Loads a multi-module checkpoint and generates fusion orderings via
greedy or beam search. Outputs json_plan files for nsys profiling.

Usage:
    # Greedy evaluation on all modules
    PYTHONPATH=~/go_fusion python3 dt_eval_multi.py \
        --checkpoint output/dt_multi_checkpoints/best.pt \
        --traj-dir output/multi_trajectories \
        --output-dir output/dt_plans

    # Beam search (width=5)
    PYTHONPATH=~/go_fusion python3 dt_eval_multi.py \
        --checkpoint output/dt_multi_checkpoints/best.pt \
        --traj-dir output/multi_trajectories \
        --beam-width 5 --output-dir output/dt_plans

    # Single module
    PYTHONPATH=~/go_fusion python3 dt_eval_multi.py \
        --checkpoint output/dt_multi_checkpoints/best.pt \
        --traj-dir output/multi_trajectories \
        --modules "Llama-3-8B-sq4k-BF16-L1/gqa_layer"
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "references", "GO_fusion"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dt_dataset_multi import load_multi_module_data
from dt_model import DecisionTransformer
from dt_train_multi import encode_batch_graphs
from sa_orchestrator import write_plan_file


@dataclass
class BeamState:
    """State of a single beam during beam search."""
    actions: List[int] = field(default_factory=list)
    fused_count: Dict[int, int] = field(default_factory=dict)
    log_prob: float = 0.0
    finished: bool = False


def greedy_generate(
    model: DecisionTransformer,
    node_embeds: torch.Tensor,
    node_mask: torch.Tensor,
    target_rtg: float,
    max_steps: int = 300,
    fusion_penalty: float = 0.0,
    max_consec_fusion: int = 0,
    temperature: float = 0.0,
    repeat_penalty: float = 1.0,
) -> List[int]:
    """Greedy autoregressive generation with constrained decoding.

    Args:
        model: Trained DecisionTransformer.
        node_embeds: [1, max_nodes, D] batched node embeddings.
        node_mask: [1, max_nodes] valid node mask.
        target_rtg: Normalized target RTG (0 to 1).
        max_steps: Maximum fusion steps.
        fusion_penalty: Subtract from fusion token logit (higher = fewer fusion.N).
        max_consec_fusion: Force real node after N consecutive fusion.N (0 = disabled).
        temperature: Sampling temperature (0 = greedy).
        repeat_penalty: Logit penalty per prior selection (replaces -inf blocking).

    Returns:
        List of action IDs.
    """
    model.eval()
    device = node_embeds.device
    num_nodes = model.num_nodes
    fusion_token_id = num_nodes
    num_real_nodes = int(node_mask[0].sum().item())
    context_len = model.context_len

    actions = []
    fused_count = {}  # node_id -> selection count (allows repeated selection)
    consec_fusion = 0

    all_rtg = []
    all_masks = []
    all_actions = []
    all_timesteps = []

    # Clamp timesteps to model's max
    max_timestep = model.timestep_embed.num_embeddings - 1

    with torch.no_grad():
        for step in range(max_steps):
            mask = torch.zeros(num_nodes, device=device)
            for i in fused_count:
                mask[i] = 1.0

            all_rtg.append(target_rtg)
            all_masks.append(mask)
            all_timesteps.append(min(step, max_timestep))

            t = len(all_rtg)
            rtg_t = torch.tensor(all_rtg, dtype=torch.float32, device=device).unsqueeze(0)
            masks_t = torch.stack(all_masks).unsqueeze(0)
            timesteps_t = torch.tensor(all_timesteps, dtype=torch.long, device=device).unsqueeze(0)

            if all_actions:
                actions_t = torch.tensor(all_actions, dtype=torch.long, device=device).unsqueeze(0)
            else:
                actions_t = torch.zeros(1, 0, dtype=torch.long, device=device)

            # Pad actions to length t
            if actions_t.shape[1] < t:
                dummy = torch.zeros(1, t - actions_t.shape[1], dtype=torch.long, device=device)
                actions_padded = torch.cat([actions_t, dummy], dim=1)
            else:
                actions_padded = actions_t

            # Truncate to context_len
            if t > context_len:
                start = t - context_len
                rtg_t = rtg_t[:, start:]
                masks_t = masks_t[:, start:]
                actions_padded = actions_padded[:, start:]
                timesteps_t = timesteps_t[:, start:]
                t_eff = context_len
            else:
                t_eff = t

            attn_mask = torch.ones(1, t_eff, device=device)
            logits = model(rtg_t, masks_t, actions_padded, timesteps_t,
                           attn_mask, node_embeds, node_mask)
            action_logits = logits[0, t_eff - 1]  # [num_actions]

            # Penalize already-selected nodes (soft penalty instead of -inf)
            if repeat_penalty > 0:
                for i, cnt in fused_count.items():
                    action_logits[i] -= repeat_penalty * cnt
            # Mask padding nodes
            action_logits[:num_nodes][node_mask[0][:num_nodes] == 0] = float("-inf")

            # Apply fusion penalty
            if fusion_penalty > 0:
                action_logits[fusion_token_id] -= fusion_penalty

            # Force real node after max consecutive fusion.N
            if max_consec_fusion > 0 and consec_fusion >= max_consec_fusion:
                action_logits[fusion_token_id] = float("-inf")

            # Select action
            if temperature <= 0:
                action = action_logits.argmax().item()
            else:
                probs = F.softmax(action_logits / temperature, dim=-1)
                action = torch.multinomial(probs, 1).item()

            actions.append(action)
            all_actions.append(action)

            if action == fusion_token_id:
                consec_fusion += 1
            else:
                consec_fusion = 0

            if action < num_nodes:
                fused_count[action] = fused_count.get(action, 0) + 1

            if len(fused_count) >= num_real_nodes:
                break

    return actions


def beam_search_generate(
    model: DecisionTransformer,
    node_embeds: torch.Tensor,
    node_mask: torch.Tensor,
    target_rtg: float,
    beam_width: int = 5,
    max_steps: int = 300,
    fusion_penalty: float = 0.0,
    max_consec_fusion: int = 0,
    repeat_penalty: float = 1.0,
) -> List[List[int]]:
    """Beam search over the pointer-network action head.

    Returns top beam_width orderings sorted by cumulative log-prob.
    """
    model.eval()
    device = node_embeds.device
    num_nodes = model.num_nodes
    fusion_token_id = num_nodes  # = max_nodes
    num_real_nodes = int(node_mask[0].sum().item())

    beams = [BeamState()]
    max_ts = model.timestep_embed.num_embeddings - 1

    with torch.no_grad():
        for step in range(max_steps):
            candidates = []

            for beam in beams:
                if beam.finished:
                    candidates.append(beam)
                    continue

                # Build history tensors for this beam
                t = step + 1
                rtg_val = max(0.0, target_rtg - target_rtg * step / max_steps)

                rtg_list = []
                mask_list = []
                fused_so_far = set()
                for s in range(t):
                    rtg_s = max(0.0, target_rtg - target_rtg * s / max_steps)
                    rtg_list.append(rtg_s)
                    m = torch.zeros(num_nodes, device=device)
                    for i in fused_so_far:
                        m[i] = 1.0
                    mask_list.append(m)
                    if s < len(beam.actions):
                        a = beam.actions[s]
                        if a < num_nodes:
                            fused_so_far.add(a)

                rtg_t = torch.tensor(rtg_list, dtype=torch.float32, device=device).unsqueeze(0)
                masks_t = torch.stack(mask_list).unsqueeze(0)
                timesteps_t = torch.clamp(
                    torch.arange(t, dtype=torch.long, device=device), max=max_ts,
                ).unsqueeze(0)

                if beam.actions:
                    actions_t = torch.tensor(beam.actions, dtype=torch.long, device=device).unsqueeze(0)
                else:
                    actions_t = torch.zeros(1, 0, dtype=torch.long, device=device)

                # Get logits at current position
                # Use full forward pass to get logits
                if actions_t.shape[1] < t:
                    actions_padded = torch.cat([
                        actions_t,
                        torch.zeros(1, t - actions_t.shape[1], dtype=torch.long, device=device),
                    ], dim=1)
                else:
                    actions_padded = actions_t

                attn_mask = torch.ones(1, t, device=device)
                logits = model(
                    rtg_t, masks_t, actions_padded, timesteps_t,
                    attn_mask, node_embeds, node_mask,
                )
                step_logits = logits[0, t - 1]  # [num_actions]

                # Penalize already-selected nodes (soft penalty instead of -inf)
                if repeat_penalty > 0:
                    for i, cnt in beam.fused_count.items():
                        if i < num_nodes:
                            step_logits[i] -= repeat_penalty * cnt

                # Mask padding nodes
                action_mask = torch.cat([
                    node_mask[0],
                    torch.ones(1, device=device),  # fusion token always valid
                ])
                step_logits[action_mask == 0] = float("-inf")

                # Apply fusion penalty
                if fusion_penalty > 0:
                    step_logits[fusion_token_id] -= fusion_penalty

                # Force real node after max consecutive fusion.N
                if max_consec_fusion > 0:
                    consec = 0
                    for a in reversed(beam.actions):
                        if a == fusion_token_id:
                            consec += 1
                        else:
                            break
                    if consec >= max_consec_fusion:
                        step_logits[fusion_token_id] = float("-inf")

                # Get top-k candidates
                log_probs = F.log_softmax(step_logits, dim=-1)
                topk_lp, topk_ids = log_probs.topk(min(beam_width, (step_logits > float("-inf")).sum().item()))

                for lp, aid in zip(topk_lp.tolist(), topk_ids.tolist()):
                    new_fused_count = dict(beam.fused_count)
                    if aid < num_nodes:
                        new_fused_count[aid] = new_fused_count.get(aid, 0) + 1

                    finished = len(new_fused_count) >= num_real_nodes

                    candidates.append(BeamState(
                        actions=beam.actions + [aid],
                        fused_count=new_fused_count,
                        log_prob=beam.log_prob + lp,
                        finished=finished,
                    ))

            # Prune to beam_width
            candidates.sort(key=lambda b: b.log_prob, reverse=True)
            beams = candidates[:beam_width]

            # Stop if all beams are finished
            if all(b.finished for b in beams):
                break

    return [b.actions for b in beams]


def actions_to_producer_names(
    actions: List[int],
    node_names: List[str],
    num_nodes: int,
) -> List[str]:
    """Convert action IDs to producer name strings.

    Only real-node actions are included. Fusion token actions (>= num_real_nodes
    or == max_nodes) are stripped — they get priority 0 in XLA anyway since
    the sequential names don't match XLA's dynamic fusion naming.
    Real nodes may appear multiple times (for multi-consumer fusions).
    """
    names = []
    for a in actions:
        if a < len(node_names):
            names.append(node_names[a])
        # else: skip fusion token actions entirely
    return names


def evaluate_accuracy_multi(
    model: DecisionTransformer,
    trajectories: List[dict],
    modules: List[dict],
    info: dict,
    device: torch.device,
) -> dict:
    """Evaluate action prediction accuracy on multi-module trajectories."""
    model.eval()
    max_nodes = info["max_nodes"]
    module_graphs = [m["graph"] for m in modules]

    all_correct = 0
    all_total = 0
    per_module = {}

    # Pre-encode all module graphs
    graph_cache = {}
    for mid, m in enumerate(modules):
        graph = m["graph"].to(device)
        ne = model.encode_graph(graph)
        n = ne.shape[0]
        padded = F.pad(ne, (0, 0, 0, max_nodes - n))
        graph_cache[mid] = padded

        mask = torch.zeros(max_nodes, device=device)
        mask[:n] = 1.0
        graph_cache[f"{mid}_mask"] = mask

    fusion_token_id = max_nodes

    with torch.no_grad():
        for traj in trajectories:
            steps = traj["steps"]
            T = len(steps)
            if T == 0:
                continue

            mid = traj["module_id"]
            num_nodes_actual = traj["num_nodes"]
            module_key = traj["module_key"]
            strategy = traj.get("strategy_detail", traj.get("strategy", "unknown"))

            node_embeds = graph_cache[mid].unsqueeze(0)  # [1, max_nodes, D]
            nm = graph_cache[f"{mid}_mask"].unsqueeze(0)  # [1, max_nodes]

            # Build actions: remap fusion tokens
            actions = []
            for s in steps:
                pid = s["producer_id"]
                if pid < 0 or pid >= num_nodes_actual:
                    actions.append(fusion_token_id)
                else:
                    actions.append(pid)

            # Build fused masks
            fused_masks = []
            mask = [0.0] * max_nodes
            for t in range(T):
                fused_masks.append(mask.copy())
                a = actions[t]
                if a < max_nodes:
                    mask[a] = 1.0

            rtg = [traj.get("reward_gpu", traj.get("reward_cost_model", 1.0))] * T

            rtg_t = torch.tensor(rtg, dtype=torch.float32, device=device).unsqueeze(0)
            masks_t = torch.tensor(fused_masks, dtype=torch.float32, device=device).unsqueeze(0)
            actions_t = torch.tensor(actions, dtype=torch.long, device=device).unsqueeze(0)
            timesteps_t = torch.arange(T, device=device).unsqueeze(0)
            attn_mask = torch.ones(1, T, device=device)

            logits = model(rtg_t, masks_t, actions_t, timesteps_t, attn_mask,
                           node_embeds, nm)
            preds = logits[0].argmax(dim=-1)
            targets = actions_t[0]

            correct = (preds == targets).sum().item()
            all_correct += correct
            all_total += T

            per_module.setdefault(module_key, {"correct": 0, "total": 0})
            per_module[module_key]["correct"] += correct
            per_module[module_key]["total"] += T

    return {
        "overall_accuracy": all_correct / max(all_total, 1),
        "per_module": {
            k: v["correct"] / max(v["total"], 1)
            for k, v in sorted(per_module.items())
        },
        "total_steps": all_total,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Multi-module DT evaluation with beam search"
    )
    parser.add_argument("--checkpoint", required=True,
                        help="Path to best.pt checkpoint")
    parser.add_argument("--traj-dir", default="output/multi_trajectories",
                        help="Directory with .pt trajectory files")
    parser.add_argument("--gpu-csv", default="output/gpu_profiles_all.csv",
                        help="GPU profiling CSV")
    parser.add_argument("--reward-mode", default="gpu_hybrid",
                        choices=["gpu", "gpu_hybrid", "cost_model"],
                        help="Reward mode (must match training)")
    parser.add_argument("--output-dir", default="output/dt_plans",
                        help="Output directory for json_plan files")
    parser.add_argument("--beam-width", type=int, default=1,
                        help="Beam width (1 = greedy)")
    parser.add_argument("--target-rtg", type=float, default=1.0,
                        help="Target return-to-go (normalized)")
    parser.add_argument("--max-steps", type=int, default=300,
                        help="Maximum generation steps")
    parser.add_argument("--modules", nargs="*",
                        help="Specific modules to evaluate")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--eval-accuracy", action="store_true",
                        help="Also run accuracy evaluation on trajectories")
    parser.add_argument("--fusion-penalty", type=float, default=0.0,
                        help="Logit penalty for fusion.N token (higher = fewer fusion.N)")
    parser.add_argument("--max-consec-fusion", type=int, default=0,
                        help="Force real node after N consecutive fusion.N (0 = disabled)")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature for greedy (0 = argmax)")
    parser.add_argument("--repeat-penalty", type=float, default=1.0,
                        help="Logit penalty per prior selection (replaces -inf blocking)")
    parser.add_argument("--max-steps-factor", type=float, default=0.0,
                        help="Compute max_steps = num_real_nodes * factor (0 = use --max-steps)")
    args = parser.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    print("Loading trajectories...")
    trajectories, modules, info = load_multi_module_data(
        args.traj_dir, args.gpu_csv, args.reward_mode
    )
    max_nodes = info["max_nodes"]
    print(f"  {len(modules)} modules, {len(trajectories)} trajectories, "
          f"max_nodes={max_nodes}")

    # Load model
    print("Loading checkpoint...")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    ckpt_info = ckpt["info"]

    model = DecisionTransformer(
        num_nodes=ckpt_info["max_nodes"],
        context_len=ckpt_info.get("context_len", 20),
        max_timestep=ckpt_info.get("max_timestep", 256) + 16,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"  Loaded epoch {ckpt['epoch']}, val_loss={ckpt.get('val_loss', '?'):.4f}, "
          f"val_acc={ckpt.get('val_acc', '?'):.4f}")

    # Accuracy evaluation
    if args.eval_accuracy:
        print("\n=== Action Prediction Accuracy ===")
        acc = evaluate_accuracy_multi(model, trajectories, modules, info, device)
        print(f"Overall: {acc['overall_accuracy']:.4f} ({acc['total_steps']} steps)")
        print("\nPer module:")
        for mod, a in acc["per_module"].items():
            print(f"  {mod:50s}: {a:.4f}")

    # Generate orderings for each module
    print(f"\n=== Generating Orderings (beam_width={args.beam_width}, "
          f"target_rtg={args.target_rtg}) ===")

    module_graphs = [m["graph"] for m in modules]
    results = {}

    for mid, m in enumerate(modules):
        module_key = m["module_key"]
        if args.modules and module_key not in args.modules:
            continue

        num_nodes_actual = m["num_nodes"]
        graph = m["graph"].to(device)
        node_names = getattr(graph, "node_names", None)

        # Encode graph
        ne = model.encode_graph(graph)
        n = ne.shape[0]
        padded_ne = F.pad(ne, (0, 0, 0, max_nodes - n)).unsqueeze(0)  # [1, max_N, D]
        nm = torch.zeros(1, max_nodes, device=device)
        nm[0, :n] = 1.0

        mode_str = "greedy" if args.beam_width <= 1 else f"beam_{args.beam_width}"
        print(f"\n  {module_key} ({num_nodes_actual} nodes):")

        # Compute per-module max_steps
        if args.max_steps_factor > 0:
            effective_max_steps = int(num_nodes_actual * args.max_steps_factor)
        else:
            effective_max_steps = args.max_steps

        if args.beam_width <= 1:
            actions = greedy_generate(
                model, padded_ne, nm, args.target_rtg, effective_max_steps,
                fusion_penalty=args.fusion_penalty,
                max_consec_fusion=args.max_consec_fusion,
                temperature=args.temperature,
                repeat_penalty=args.repeat_penalty,
            )
            all_orderings = [actions]
        else:
            all_orderings = beam_search_generate(
                model, padded_ne, nm, args.target_rtg,
                args.beam_width, effective_max_steps,
                fusion_penalty=args.fusion_penalty,
                max_consec_fusion=args.max_consec_fusion,
                repeat_penalty=args.repeat_penalty,
            )

        for beam_idx, actions in enumerate(all_orderings):
            n_regular = sum(1 for a in actions if a < num_nodes_actual)
            n_fusion = sum(1 for a in actions if a >= num_nodes_actual)
            unique = len(set(a for a in actions if a < num_nodes_actual))

            # Convert to producer names
            if node_names:
                ordering = actions_to_producer_names(
                    actions, node_names, num_nodes_actual,
                )
            else:
                ordering = [f"node_{a}" for a in actions if a < num_nodes_actual]

            # Save json_plan
            if args.beam_width <= 1:
                strategy_name = f"dt_greedy"
                plan_filename = f"{module_key.replace('/', '__')}_dt_greedy.json"
            else:
                strategy_name = f"dt_beam{args.beam_width}_{beam_idx}"
                plan_filename = f"{module_key.replace('/', '__')}_{strategy_name}.json"

            plan_path = os.path.join(args.output_dir, plan_filename)
            os.makedirs(os.path.dirname(plan_path) or args.output_dir, exist_ok=True)
            write_plan_file(ordering, plan_path, metadata={
                "type": mode_str,
                "beam_idx": beam_idx,
                "target_rtg": args.target_rtg,
                "num_steps": len(actions),
            })

            suffix = "" if beam_idx == 0 else f" (beam {beam_idx})"
            print(f"    {strategy_name}{suffix}: {len(actions)} steps, "
                  f"{n_regular} regular + {n_fusion} fusion.N, "
                  f"{unique} unique nodes → {plan_filename}")

            if beam_idx == 0:
                results[module_key] = {
                    "strategy": strategy_name,
                    "num_steps": len(actions),
                    "unique_nodes": unique,
                    "plan_path": plan_path,
                }

    # Summary
    print(f"\n=== Summary ===")
    print(f"Generated plans for {len(results)} modules in {args.output_dir}/")
    if args.beam_width > 1:
        print(f"Beam width: {args.beam_width}, top {args.beam_width} orderings per module")
    print(f"\nTo profile these plans:")
    print(f"  python3 profile_json_plan.py --xla-tool <path> --mode direct \\")
    print(f"    --json-plan <plan.json> --hlo-input <hlo> --module-name <mod> --strategy-name <strat>")


if __name__ == "__main__":
    main()
