"""Multi-module dataset for Decision Transformer with real GPU rewards.

Loads trajectory .pt files from multiple HLO modules and maps strategies
to real GPU kernel times from nsys profiling CSV.

Usage:
    from dt_dataset_multi import create_multi_dataloaders
    train_loader, val_loader, modules, info = create_multi_dataloaders(
        traj_dir="output/multi_trajectories",
        gpu_csv_path="output/gpu_profiles_all.csv",
        reward_mode="gpu",
    )
"""

from __future__ import annotations

import csv
import glob
import math
import os
import random
import sys
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "references", "GO_fusion"))

from src.hlo_parser.feature_encoder import encode_features, FEATURE_DIM


def load_gpu_rewards(csv_path: str) -> Dict[Tuple[str, str], float]:
    """Load real GPU kernel times from profiling CSV.

    Returns: {(module_key, strategy): avg_kernel_per_iter_us}
    where module_key is like "Llama-3-8B-sq4k-BF16-L1/ffn_layer".
    """
    times = {}
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["module"], row["strategy"])
            times[key] = float(row["avg_kernel_per_iter_us"])
    return times


def load_multi_module_data(
    traj_dir: str,
    gpu_csv_path: Optional[str] = None,
    reward_mode: str = "gpu",
) -> Tuple[List[dict], List[dict], Dict]:
    """Load trajectories from all .pt files with GPU reward mapping.

    Args:
        traj_dir: Directory containing <Model>__<module>.pt files.
        gpu_csv_path: Path to gpu_profiles_all.csv.
        reward_mode: "gpu", "gpu_hybrid", or "cost_model".

    Returns:
        trajectories: list of trajectory dicts with module info
        modules: list of {module_key, graph, num_nodes}
        info: summary dict
    """
    gpu_times = {}
    if gpu_csv_path and os.path.exists(gpu_csv_path):
        gpu_times = load_gpu_rewards(gpu_csv_path)

    modules = []
    trajectories = []

    for pt_file in sorted(glob.glob(os.path.join(traj_dir, "*.pt"))):
        data = torch.load(pt_file, weights_only=False)
        filename = os.path.basename(pt_file).replace(".pt", "")
        # Filename: Model__module → CSV key: Model/module
        module_key = filename.replace("__", "/")

        # Skip whole_computation modules (too large for sub-module DT)
        if "whole_computation" in filename:
            continue

        graph = data["initial_graph"]

        # Re-encode features if graph was saved with older feature dim
        if graph.x.shape[1] != FEATURE_DIM and hasattr(graph, 'instructions'):
            graph = encode_features(graph)

        num_nodes = graph.num_nodes

        module_id = len(modules)
        modules.append({
            "module_key": module_key,
            "graph": graph,
            "num_nodes": num_nodes,
        })

        # XLA default GPU time for this module (denominator of speedup)
        xla_default_us = gpu_times.get((module_key, "xla_default"))

        for traj in data["trajectories"]:
            strategy_detail = traj.get("strategy_detail", "")

            # Compute GPU speedup: xla_default / strategy (> 1.0 = faster)
            gpu_speedup = None
            if xla_default_us is not None:
                strat_us = gpu_times.get((module_key, strategy_detail))
                if strat_us and strat_us > 0:
                    gpu_speedup = xla_default_us / strat_us

            # Filter trajectories without GPU reward in gpu modes
            if reward_mode in ("gpu", "gpu_hybrid", "gpu_progressive") and gpu_speedup is None:
                continue

            # Remap fusion token: -1 → num_nodes (module-specific)
            steps = traj["steps"]
            for step in steps:
                if step["producer_id"] == -1:
                    step["producer_id"] = num_nodes

            trajectories.append({
                "module_key": module_key,
                "module_id": module_id,
                "num_nodes": num_nodes,
                "strategy": traj.get("strategy", "unknown"),
                "strategy_detail": strategy_detail,
                "steps": steps,
                "reward_cost_model": traj.get("reward_cost_model", 1.0),
                "reward_gpu": gpu_speedup,
            })

    max_nodes = max(m["num_nodes"] for m in modules)

    gpu_rewards = [t["reward_gpu"] for t in trajectories if t["reward_gpu"] is not None]
    info = {
        "num_modules": len(modules),
        "num_trajectories": len(trajectories),
        "max_nodes": max_nodes,
        "num_actions": max_nodes + 1,
        "module_sizes": {m["module_key"]: m["num_nodes"] for m in modules},
        "reward_mode": reward_mode,
    }
    if gpu_rewards:
        info["gpu_reward_range"] = (min(gpu_rewards), max(gpu_rewards))
        info["gpu_reward_mean"] = sum(gpu_rewards) / len(gpu_rewards)

    return trajectories, modules, info


class MultiModuleFusionDTDataset(Dataset):
    """Sliding-window dataset across multiple HLO modules.

    Each item includes module_id for graph selection and fused_masks/actions
    padded to max_nodes (the largest graph across all modules).

    Reward modes:
      - "gpu": Flat RTG = gpu_speedup (same for all steps). No credit assignment.
      - "gpu_hybrid": Per-step cost-model shape scaled to GPU speedup.
        R_t = (cost_rtg_t / cost_rtg_0) * gpu_speedup. Combines per-step
        credit assignment from cost model with real GPU terminal reward.
      - "gpu_progressive": GPU-anchored progressive decay. Uses cost-model
        shape where signal exists (allows negative rewards); falls back to
        sqrt decay for degenerate modules (all-zero cost-model priorities).
      - "cost_model": Per-step cost-model RTG only (no GPU data needed).
    """

    def __init__(
        self,
        trajectories: List[dict],
        max_nodes: int,
        context_len: int = 20,
        reward_mode: str = "gpu",
        max_gpu_reward: float = 1.0,
        max_cost_r0: float = 1.0,
    ):
        self.max_nodes = max_nodes
        self.context_len = context_len
        self.fusion_token_id = max_nodes  # universal fusion token index
        self.windows: List[dict] = []

        for traj in trajectories:
            module_id = traj["module_id"]
            num_nodes = traj["num_nodes"]
            steps = traj["steps"]
            T = len(steps)
            if T == 0:
                continue

            # Compute RTG
            # Per-step cost-model rewards (allow negatives for gpu_progressive)
            if reward_mode == "gpu_progressive":
                rewards_raw = [s["us_unfused"] - s["us_fused"] for s in steps]
            else:
                rewards_raw = [max(0.0, s["us_unfused"] - s["us_fused"]) for s in steps]
            cost_rtg = [0.0] * T
            cost_rtg[T - 1] = rewards_raw[T - 1]
            for t in range(T - 2, -1, -1):
                cost_rtg[t] = rewards_raw[t] + cost_rtg[t + 1]
            cost_r0 = cost_rtg[0] if cost_rtg[0] != 0 else 1.0

            if reward_mode == "gpu" and traj["reward_gpu"] is not None:
                # Flat GPU speedup, normalized by max observed
                gpu_r = traj["reward_gpu"] / max_gpu_reward
                rtg = [gpu_r] * T
            elif reward_mode == "gpu_hybrid" and traj["reward_gpu"] is not None:
                # Hybrid: cost-model shape × GPU terminal reward
                # R_t = (cost_rtg_t / cost_r0) × gpu_speedup / max_gpu_reward
                gpu_r = traj["reward_gpu"] / max_gpu_reward
                rtg = [(cr / cost_r0) * gpu_r for cr in cost_rtg]
            elif reward_mode == "gpu_progressive" and traj["reward_gpu"] is not None:
                # Progressive: cost-model shape if signal exists,
                # synthetic sqrt decay for degenerate modules
                gpu_r = traj["reward_gpu"] / max_gpu_reward
                abs_cost_r0 = abs(cost_r0)
                if abs_cost_r0 > 1.0:  # meaningful cost-model signal
                    # Normalize by abs to keep RTG in [0, gpu_r] range
                    # (cost_rtg monotonically decreases, so cost_rtg[0] is largest)
                    rtg = [(cr / abs_cost_r0) * gpu_r for cr in cost_rtg]
                else:
                    # Degenerate: synthetic sqrt decay
                    rtg = [gpu_r * math.sqrt(max(0.0, 1.0 - t / max(T - 1, 1)))
                           for t in range(T)]
            else:
                # Pure cost-model RTG, normalized by max_cost_r0
                if max_cost_r0 > 0:
                    rtg = [r / max_cost_r0 for r in cost_rtg]
                else:
                    rtg = cost_rtg

            # Actions: remap module-specific fusion token → universal
            actions = []
            for s in steps:
                pid = s["producer_id"]
                if pid >= num_nodes:  # fusion token
                    actions.append(self.fusion_token_id)
                else:
                    actions.append(pid)

            # Fused masks padded to max_nodes
            fused_masks = []
            mask = [0.0] * max_nodes
            for t in range(T):
                fused_masks.append(mask.copy())
                a = actions[t]
                if a < max_nodes:
                    mask[a] = 1.0

            # Sliding windows with stride 1
            for start in range(max(1, T - context_len + 1)):
                end = min(start + context_len, T)
                self.windows.append({
                    "module_id": module_id,
                    "num_nodes": num_nodes,
                    "returns_to_go": rtg[start:end],
                    "fused_masks": fused_masks[start:end],
                    "actions": actions[start:end],
                    "timesteps": list(range(start, end)),
                })

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        w = self.windows[idx]
        K = self.context_len
        L = len(w["actions"])

        rtg = torch.zeros(K)
        rtg[:L] = torch.tensor(w["returns_to_go"], dtype=torch.float32)

        masks = torch.zeros(K, self.max_nodes)
        masks[:L] = torch.tensor(w["fused_masks"], dtype=torch.float32)

        actions = torch.zeros(K, dtype=torch.long)
        actions[:L] = torch.tensor(w["actions"], dtype=torch.long)

        timesteps = torch.zeros(K, dtype=torch.long)
        timesteps[:L] = torch.tensor(w["timesteps"], dtype=torch.long)

        attn_mask = torch.zeros(K)
        attn_mask[:L] = 1.0

        return {
            "module_id": w["module_id"],
            "num_nodes": w["num_nodes"],
            "returns_to_go": rtg,
            "fused_masks": masks,
            "actions": actions,
            "timesteps": timesteps,
            "attn_mask": attn_mask,
        }


def create_multi_dataloaders(
    traj_dir: str,
    gpu_csv_path: Optional[str] = None,
    reward_mode: str = "gpu",
    context_len: int = 20,
    batch_size: int = 64,
    train_ratio: float = 0.8,
    seed: int = 42,
    holdout_arch: Optional[str] = None,
) -> Tuple[DataLoader, DataLoader, List[dict], Dict]:
    """Create train/val dataloaders for multi-module DT.

    Args:
        holdout_arch: If set, hold out all modules whose key starts with this
                      prefix for validation (leave-one-architecture-out).
                      E.g., "Griffin-2B" holds out all Griffin modules.

    Returns:
        (train_loader, val_loader, modules, info)
    """
    trajectories, modules, info = load_multi_module_data(
        traj_dir, gpu_csv_path, reward_mode,
    )

    max_nodes = info["max_nodes"]
    rng = random.Random(seed)

    if holdout_arch:
        train_trajs = [t for t in trajectories
                       if not t["module_key"].startswith(holdout_arch)]
        val_trajs = [t for t in trajectories
                     if t["module_key"].startswith(holdout_arch)]
    else:
        indices = list(range(len(trajectories)))
        rng.shuffle(indices)
        n_train = int(len(trajectories) * train_ratio)
        train_trajs = [trajectories[i] for i in indices[:n_train]]
        val_trajs = [trajectories[i] for i in indices[n_train:]]

    # Normalization from training data only
    if reward_mode in ("gpu", "gpu_hybrid", "gpu_progressive"):
        gpu_rewards = [t["reward_gpu"] for t in train_trajs
                       if t["reward_gpu"] is not None]
        max_gpu_reward = max(gpu_rewards) if gpu_rewards else 1.0
        max_cost_r0 = 1.0
    else:
        max_gpu_reward = 1.0
        max_cost_r0 = 0.0
        for t in train_trajs:
            steps = t["steps"]
            if steps:
                r0 = sum(max(0.0, s["us_unfused"] - s["us_fused"]) for s in steps)
                max_cost_r0 = max(max_cost_r0, r0)
        if max_cost_r0 == 0:
            max_cost_r0 = 1.0

    train_ds = MultiModuleFusionDTDataset(
        train_trajs, max_nodes, context_len, reward_mode,
        max_gpu_reward, max_cost_r0,
    )
    val_ds = MultiModuleFusionDTDataset(
        val_trajs, max_nodes, context_len, reward_mode,
        max_gpu_reward, max_cost_r0,
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # Compute max timestep from data
    max_timestep = 0
    for t in trajectories:
        max_timestep = max(max_timestep, len(t["steps"]))

    info.update({
        "num_train_trajs": len(train_trajs),
        "num_val_trajs": len(val_trajs),
        "num_train_windows": len(train_ds),
        "num_val_windows": len(val_ds),
        "context_len": context_len,
        "max_gpu_reward": max_gpu_reward,
        "max_cost_r0": max_cost_r0,
        "max_timestep": max_timestep,
    })

    return train_loader, val_loader, modules, info


if __name__ == "__main__":
    traj_dir = "output/multi_trajectories"
    gpu_csv = "output/gpu_profiles_all.csv"

    train_loader, val_loader, modules, info = create_multi_dataloaders(
        traj_dir, gpu_csv, reward_mode="gpu",
    )

    print("Dataset info:")
    for k, v in info.items():
        if k != "module_sizes":
            print(f"  {k}: {v}")
    print(f"\nModules ({info['num_modules']}):")
    for k, v in info["module_sizes"].items():
        print(f"  {k}: {v} nodes")

    batch = next(iter(train_loader))
    print(f"\nBatch shapes:")
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: {v.shape}")
        else:
            print(f"  {k}: {v}")
