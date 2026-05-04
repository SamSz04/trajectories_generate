"""Evaluation for Dynamic Fusion DT.

Supports three evaluation modes:
  - teacher_forced_trace: reconstruct G_t from enriched trajectory, predict a_t
  - remove_only_ablation: greedy generation without real graph mutation (ablation only)
  - xla_in_the_loop: deferred (interface defined but not implemented)

Usage:
    PYTHONPATH=~/go_fusion python3 dt_eval_dynamic.py \
        --enriched-dir output/dynamic_features \
        --checkpoint output/dt_dynamic_checkpoints/best.pt \
        --device cuda
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch_geometric.data import Batch

from dt_model_dynamic import DynamicFusionDT
from dt_dataset_dynamic import load_enriched_modules
from graph_contractor import (
    reconstruct_trajectory_states,
    contract_graph,
)
from dynamic_graph_types import GraphStepMetadata
from reward_builder import RewardConfig, build_returns


# ======================================================================
# Teacher-forcing evaluation
# ======================================================================

@torch.no_grad()
def teacher_forced_eval(
    model: DynamicFusionDT,
    enriched_dir: str,
    gpu_csv_path: Optional[str] = None,
    context_len: int = 20,
    device: torch.device = torch.device("cpu"),
    max_trajectories_per_module: int = 0,
    skip_computation: bool = True,
    verbose: bool = True,
    reward_config: Optional[RewardConfig] = None,
    norm_stats: Optional[Dict[str, dict]] = None,
) -> Dict[str, object]:
    """Run teacher-forced evaluation over all enriched trajectories.

    For each step t, reconstruct G_t, encode it, predict a_t among C_t,
    compare with ground truth.

    Returns:
        Metrics dict with overall and per-module/per-type breakdowns.
    """
    model.eval()

    module_data_list, info = load_enriched_modules(
        enriched_dir=enriched_dir,
        gpu_csv_path=gpu_csv_path,
        skip_computation=skip_computation,
        max_trajectories_per_module=max_trajectories_per_module,
        verbose=verbose,
    )

    # Accumulators
    overall = _MetricsAccum()
    per_module: Dict[str, _MetricsAccum] = defaultdict(lambda: _MetricsAccum())
    per_arch: Dict[str, _MetricsAccum] = defaultdict(lambda: _MetricsAccum())
    per_type = {"original": _MetricsAccum(), "fusion": _MetricsAccum()}
    per_fidelity: Dict[str, _MetricsAccum] = defaultdict(lambda: _MetricsAccum())

    for mi, mdata in enumerate(module_data_list):
        module_key = mdata["module_key"]
        arch = mdata["architecture"]
        g0 = mdata["g0"]

        for ti, traj in enumerate(mdata["trajectories"]):
            # Build cluster features cache
            cluster_cache = {}
            for cname, cdata in traj.get("clusters", {}).items():
                if "features" in cdata and isinstance(cdata["features"], torch.Tensor):
                    cluster_cache[cname] = cdata["features"]

            # Reconstruct states
            states = reconstruct_trajectory_states(
                traj, g0, module_key, ti,
            )

            T = len(states)
            if T == 0:
                continue

            # Compute returns (matching training)
            gpu_reward = traj.get("reward_gpu")
            if reward_config is not None and norm_stats is not None:
                returns_list, _ = build_returns(
                    traj, module_key, norm_stats, reward_config,
                )
                return_dim = reward_config.return_dim
            else:
                # Legacy: flat gpu_reward
                rtg_val = gpu_reward if gpu_reward is not None else 1.0
                returns_list = [[rtg_val]] * T
                return_dim = 1

            # Slide window and evaluate
            for start in range(0, T, context_len):
                end = min(start + context_len, T)
                K = end - start

                # Build graphs for this window
                graphs = []
                metadatas = []
                actions_list = []
                rtgs_list = []
                timesteps_list = []

                for k in range(K):
                    state = states[start + k]
                    data, metadata = contract_graph(g0, state, cluster_cache)
                    graphs.append(data)
                    metadatas.append(metadata)
                    actions_list.append(
                        metadata.selected_idx if metadata.selected_idx is not None else -1
                    )
                    rtgs_list.append(returns_list[start + k])
                    timesteps_list.append(start + k)

                # Pad to context_len if needed
                while len(graphs) < context_len:
                    dummy = _make_dummy_graph(g0.x.shape[1])
                    graphs.append(dummy)
                    metadatas.append(None)
                    actions_list.append(-1)
                    rtgs_list.append([0.0] * return_dim)
                    timesteps_list.append(0)

                # Batch
                pyg_batch = Batch.from_data_list(graphs).to(device)
                graph_sizes = torch.tensor(
                    [g.num_nodes for g in graphs], dtype=torch.long, device=device,
                )
                actions = torch.tensor(
                    [actions_list], dtype=torch.long, device=device,
                )  # [1, K_padded]
                rtgs = torch.tensor(
                    [rtgs_list], dtype=torch.float32, device=device,
                )
                timesteps = torch.tensor(
                    [timesteps_list], dtype=torch.long, device=device,
                )
                attn_mask = torch.zeros(1, context_len, dtype=torch.long, device=device)
                attn_mask[0, :K] = 1

                max_vt = max(g.num_nodes for g in graphs)
                candidate_masks = torch.zeros(
                    1, context_len, max_vt, dtype=torch.bool, device=device,
                )
                for k, g in enumerate(graphs):
                    n = g.candidate_mask.shape[0]
                    candidate_masks[0, k, :n] = g.candidate_mask.to(device)

                # Forward
                logits = model(
                    pyg_batch, graph_sizes, actions, rtgs, timesteps,
                    attn_mask, candidate_masks,
                )  # [1, context_len, max_vt]

                # Score each valid step
                for k in range(K):
                    meta = metadatas[k]
                    if meta is None or meta.selected_idx is None:
                        continue

                    step_logits = logits[0, k]  # [max_vt]
                    target = actions_list[k]
                    if target < 0 or target >= step_logits.shape[0]:
                        continue

                    pred = step_logits.argmax().item()
                    is_correct = (pred == target)

                    # Rank of correct answer
                    sorted_idx = step_logits.argsort(descending=True)
                    rank = (sorted_idx == target).nonzero(as_tuple=True)[0]
                    rank = rank[0].item() + 1 if len(rank) > 0 else len(sorted_idx)

                    # Top-3
                    _, top3 = step_logits.topk(min(3, step_logits.shape[0]))
                    in_top3 = target in top3.tolist()

                    # Record
                    record = {
                        "correct": is_correct,
                        "rank": rank,
                        "in_top3": in_top3,
                        "n_candidates": meta.num_candidates,
                    }

                    overall.add(record)
                    per_module[module_key].add(record)
                    per_arch[arch].add(record)
                    per_fidelity[meta.graph_fidelity].add(record)

                    name = meta.idx_to_name[target] if target < len(meta.idx_to_name) else ""
                    if name.startswith("fusion."):
                        per_type["fusion"].add(record)
                    else:
                        per_type["original"].add(record)

        if verbose:
            m = per_module[module_key]
            print(f"  {module_key}: {m.top1_acc():.3f} top1, "
                  f"{m.mrr():.3f} MRR ({m.count} steps)")

    # Compile results
    results = {
        "overall": overall.summary(),
        "per_module": {k: v.summary() for k, v in per_module.items()},
        "per_architecture": {k: v.summary() for k, v in per_arch.items()},
        "per_type": {k: v.summary() for k, v in per_type.items()},
        "per_fidelity": {k: v.summary() for k, v in per_fidelity.items()},
    }

    if verbose:
        print(f"\n=== Overall ===")
        s = results["overall"]
        print(f"  Steps: {s['count']}")
        print(f"  Top-1: {s['top1_acc']:.4f}")
        print(f"  Top-3: {s['top3_acc']:.4f}")
        print(f"  MRR:   {s['mrr']:.4f}")
        print(f"  Mean rank: {s['mean_rank']:.2f}")

        print(f"\n=== By action type ===")
        for atype, sm in results["per_type"].items():
            print(f"  {atype}: top1={sm['top1_acc']:.4f}, "
                  f"MRR={sm['mrr']:.4f} ({sm['count']} steps)")

        print(f"\n=== By fidelity ===")
        for fid, sm in results["per_fidelity"].items():
            print(f"  {fid}: top1={sm['top1_acc']:.4f} ({sm['count']} steps)")

    return results


# ======================================================================
# Dynamic scorer evaluation
# ======================================================================

@torch.no_grad()
def score_candidates(
    model: DynamicFusionDT,
    g0,
    state,
    cluster_cache: dict,
    history_states: list,
    history_actions: list,
    target_rtg: float,
    device: torch.device,
    context_len: int = 20,
    target_cm_rtg: float = 5.0,
    target_real_return: float = 1.5,
) -> torch.Tensor:
    """Score all candidates at a given step.

    For free scoring (deployment), uses target RTG prompts.
    Ground-truth RTG_cm_t requires future information and cannot be used here.

    Args:
        model: Trained DynamicFusionDT
        g0: Initial graph G_0
        state: GraphStepState at current step
        cluster_cache: Pre-computed cluster features
        history_states: Previous GraphStepStates
        history_actions: Previous action indices
        target_rtg: Target return-to-go (for return_dim=1)
        device: Torch device
        context_len: Context window size
        target_cm_rtg: Target cost-model RTG (for return_dim=2, channel 0)
        target_real_return: Target real return (for return_dim=2, channel 1)

    Returns:
        scores: [num_candidates] tensor of scores
    """
    model.eval()

    # Build window: history + current step
    all_states = list(history_states) + [state]
    T = len(all_states)
    start = max(0, T - context_len)
    window_states = all_states[start:]
    window_actions = (list(history_actions) + [-1])[start:]
    K = len(window_states)

    graphs = []
    metadatas_local = []
    for s in window_states:
        data, metadata = contract_graph(g0, s, cluster_cache)
        graphs.append(data)
        metadatas_local.append(metadata)

    # Pad to context_len
    while len(graphs) < context_len:
        dummy = _make_dummy_graph(g0.x.shape[1])
        graphs.append(dummy)
        metadatas_local.append(None)
        window_actions.append(-1)

    pyg_batch = Batch.from_data_list(graphs).to(device)
    graph_sizes = torch.tensor([g.num_nodes for g in graphs], dtype=torch.long, device=device)
    actions = torch.tensor([window_actions[:context_len]], dtype=torch.long, device=device)

    # Build RTG tensor matching model's return_dim
    return_dim = getattr(model, 'return_dim', 1)
    if return_dim == 1:
        rtgs = torch.full((1, context_len, 1), target_rtg, device=device)
    else:
        # Dual channel: [target_cm_rtg, target_real_return]
        rtgs = torch.zeros(1, context_len, return_dim, device=device)
        rtgs[:, :, 0] = target_cm_rtg
        rtgs[:, :, 1] = target_real_return

    timesteps = torch.arange(context_len, device=device).unsqueeze(0)
    attn_mask = torch.zeros(1, context_len, dtype=torch.long, device=device)
    attn_mask[0, :K] = 1

    max_vt = max(g.num_nodes for g in graphs)
    candidate_masks = torch.zeros(1, context_len, max_vt, dtype=torch.bool, device=device)
    for k, g in enumerate(graphs):
        n = g.candidate_mask.shape[0]
        candidate_masks[0, k, :n] = g.candidate_mask.to(device)

    logits = model(pyg_batch, graph_sizes, actions, rtgs, timesteps, attn_mask, candidate_masks)

    # Return scores for the last valid step
    last_step = K - 1
    step_logits = logits[0, last_step]  # [max_vt]

    # Extract scores for candidates only
    meta = metadatas_local[last_step]
    if meta is not None:
        candidate_scores = step_logits[meta.candidate_indices]
        return candidate_scores
    return step_logits


# ======================================================================
# Helpers
# ======================================================================

class _MetricsAccum:
    """Accumulator for evaluation metrics."""

    def __init__(self):
        self.count = 0
        self.correct = 0
        self.top3 = 0
        self.rank_sum = 0.0
        self.rr_sum = 0.0

    def add(self, record: dict):
        self.count += 1
        self.correct += int(record["correct"])
        self.top3 += int(record["in_top3"])
        self.rank_sum += record["rank"]
        self.rr_sum += 1.0 / record["rank"]

    def top1_acc(self) -> float:
        return self.correct / max(self.count, 1)

    def top3_acc(self) -> float:
        return self.top3 / max(self.count, 1)

    def mrr(self) -> float:
        return self.rr_sum / max(self.count, 1)

    def mean_rank(self) -> float:
        return self.rank_sum / max(self.count, 1)

    def summary(self) -> dict:
        return {
            "count": self.count,
            "top1_acc": self.top1_acc(),
            "top3_acc": self.top3_acc(),
            "mrr": self.mrr(),
            "mean_rank": self.mean_rank(),
        }


def _make_dummy_graph(feature_dim: int):
    """Create a single-node dummy graph for padding."""
    from torch_geometric.data import Data
    return Data(
        x=torch.zeros(1, feature_dim),
        edge_index=torch.zeros(2, 0, dtype=torch.long),
        opcode_ids=torch.zeros(1, dtype=torch.long),
        cluster_features=torch.zeros(1, 34),
        is_cluster=torch.zeros(1, dtype=torch.bool),
        candidate_mask=torch.zeros(1, dtype=torch.bool),
        num_nodes=1,
    )


# ======================================================================
# CLI
# ======================================================================

def main():
    parser = argparse.ArgumentParser(description="Evaluate Dynamic Fusion DT")
    parser.add_argument("--enriched-dir", default="output/dynamic_features")
    parser.add_argument("--gpu-csv", default="output/gpu_profiles_merged.csv")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--context-len", type=int, default=20)
    parser.add_argument("--max-trajs-per-module", type=int, default=0)
    parser.add_argument("--skip-computation", action="store_true", default=True)
    parser.add_argument("--output", default=None, help="Save results JSON")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()
    device = torch.device(args.device)

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    ckpt_args = ckpt.get("args", {})

    # Load return conditioning config from checkpoint
    return_dim = ckpt.get("return_dim", 1)
    return_mode = ckpt.get("return_mode", None)
    norm_mode = ckpt.get("norm_mode", "module_p95")
    real_return_norm = ckpt.get("real_return_norm", "legacy_max")
    norm_stats = ckpt.get("norm_stats", {})
    max_gpu_reward = ckpt.get("max_gpu_reward", 1.0)

    reward_config = None
    if return_mode is not None:
        reward_config = RewardConfig(
            return_mode=return_mode,
            norm_mode=norm_mode,
            real_return_norm=real_return_norm,
            max_gpu_reward=max_gpu_reward,
        )
        print(f"Return mode: {return_mode}, return_dim: {return_dim}, "
              f"norm: {norm_mode}, real_norm: {real_return_norm}")

    model = DynamicFusionDT(
        embed_dim=ckpt_args.get("embed_dim", 64),
        num_heads=ckpt_args.get("num_heads", 4),
        num_layers=ckpt_args.get("num_layers", 3),
        d_ff=ckpt_args.get("d_ff", 128),
        context_len=args.context_len,
        gnn_hidden_dim=ckpt_args.get("gnn_hidden_dim"),
        dropout=0.0,
        return_dim=return_dim,
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')}")
    print(f"Model: {model.count_parameters():,} parameters")

    results = teacher_forced_eval(
        model=model,
        enriched_dir=args.enriched_dir,
        gpu_csv_path=args.gpu_csv,
        context_len=args.context_len,
        device=device,
        max_trajectories_per_module=args.max_trajs_per_module,
        skip_computation=args.skip_computation,
        reward_config=reward_config,
        norm_stats=norm_stats,
    )

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
