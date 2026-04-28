"""Dataset and dataloaders for Decision Transformer training on fusion trajectories."""

from __future__ import annotations

import os
import random
import sys
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "references", "GO_fusion"))

from src.hlo_parser.parser import parse_hlo_file
from src.hlo_parser.graph_builder import build_graph
from src.hlo_parser.feature_encoder import encode_features

NUM_NODES = 56
FUSION_TOKEN_ID = NUM_NODES  # special index for dynamically-created fusion.N producers
NUM_ACTIONS = NUM_NODES + 1  # 57


def load_trajectories(
    traj_path: str,
) -> Tuple[List[dict], object]:
    """Load trajectories and graph from .pt file.

    The .pt file must have been generated with the correct HLO
    (before_priority-fusion.txt, 56 nodes).

    Returns:
        (trajectories, graph_data)
    """
    data = torch.load(traj_path, weights_only=False)
    trajectories = data["trajectories"]
    graph_data = data["initial_graph"]

    # Remap any remaining -1 producer_ids (fusion.N nodes) to FUSION_TOKEN_ID
    for traj in trajectories:
        for step in traj["steps"]:
            if step["producer_id"] == -1:
                step["producer_id"] = FUSION_TOKEN_ID

    return trajectories, graph_data


def compute_returns_to_go(steps: List[dict], normalize_by: float = 1.0) -> List[float]:
    """Compute return-to-go for each step.

    Per-step reward: r_t = max(0, us_unfused - us_fused)
    Return-to-go:   R_t = sum(r_{t:T}) / normalize_by
    """
    rewards = [max(0.0, s["us_unfused"] - s["us_fused"]) for s in steps]
    T = len(rewards)
    rtg = [0.0] * T
    rtg[T - 1] = rewards[T - 1]
    for t in range(T - 2, -1, -1):
        rtg[t] = rewards[t] + rtg[t + 1]
    if normalize_by > 0:
        rtg = [r / normalize_by for r in rtg]
    return rtg


def _compute_max_r0(trajectories: List[dict]) -> float:
    """Compute max initial return-to-go across trajectories (unnormalized)."""
    max_r0 = 0.0
    for traj in trajectories:
        steps = traj["steps"]
        if steps:
            rtg = compute_returns_to_go(steps, normalize_by=1.0)
            max_r0 = max(max_r0, rtg[0])
    return max_r0 if max_r0 > 0 else 1.0


class FusionDTDataset(Dataset):
    """Sliding-window dataset over fusion trajectories for Decision Transformer.

    Each item is a window of up to context_len timesteps containing:
        returns_to_go: [K]             normalized return-to-go
        fused_masks:   [K, num_nodes]  binary mask of nodes fused before each step
        actions:       [K]             action taken (node index 0..56)
        timesteps:     [K]             absolute step index within trajectory
        attn_mask:     [K]             1.0 for real steps, 0.0 for padding
    """

    def __init__(
        self,
        trajectories: List[dict],
        num_nodes: int = NUM_NODES,
        context_len: int = 20,
        max_r0: float = 1.0,
    ):
        self.num_nodes = num_nodes
        self.context_len = context_len
        self.max_r0 = max_r0
        self.windows: List[dict] = []

        for traj in trajectories:
            steps = traj["steps"]
            T = len(steps)
            if T == 0:
                continue

            rtg = compute_returns_to_go(steps, normalize_by=max_r0)

            actions = []
            for s in steps:
                pid = s["producer_id"]
                actions.append(pid if pid >= 0 else FUSION_TOKEN_ID)

            # fused_masks[t] = which nodes were fused before step t
            fused_masks = []
            mask = [0.0] * num_nodes
            for t in range(T):
                fused_masks.append(mask.copy())
                a = actions[t]
                if a < num_nodes:
                    mask[a] = 1.0

            # Sliding windows with stride 1
            for start in range(max(1, T - context_len + 1)):
                end = min(start + context_len, T)
                self.windows.append(
                    {
                        "returns_to_go": rtg[start:end],
                        "fused_masks": fused_masks[start:end],
                        "actions": actions[start:end],
                        "timesteps": list(range(start, end)),
                    }
                )

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        w = self.windows[idx]
        K = self.context_len
        L = len(w["actions"])

        rtg = torch.zeros(K)
        rtg[:L] = torch.tensor(w["returns_to_go"], dtype=torch.float32)

        masks = torch.zeros(K, self.num_nodes)
        masks[:L] = torch.tensor(w["fused_masks"], dtype=torch.float32)

        actions = torch.zeros(K, dtype=torch.long)
        actions[:L] = torch.tensor(w["actions"], dtype=torch.long)

        timesteps = torch.zeros(K, dtype=torch.long)
        timesteps[:L] = torch.tensor(w["timesteps"], dtype=torch.long)

        attn_mask = torch.zeros(K)
        attn_mask[:L] = 1.0

        return {
            "returns_to_go": rtg,
            "fused_masks": masks,
            "actions": actions,
            "timesteps": timesteps,
            "attn_mask": attn_mask,
        }


def create_dataloaders(
    traj_path: str,
    context_len: int = 20,
    batch_size: int = 64,
    train_ratio: float = 0.8,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, object, Dict]:
    """Create train and validation dataloaders.

    Splits by trajectory (not by window) to avoid data leakage.
    Normalizes return-to-go using max R_0 from training trajectories only.

    Returns:
        (train_loader, val_loader, graph_data, info_dict)
    """
    trajectories, graph_data = load_trajectories(traj_path)
    num_nodes = graph_data.num_nodes

    # Shuffle trajectories with seed for reproducible stratified split
    rng = random.Random(seed)
    indices = list(range(len(trajectories)))
    rng.shuffle(indices)

    n_train = int(len(trajectories) * train_ratio)
    train_trajs = [trajectories[i] for i in indices[:n_train]]
    val_trajs = [trajectories[i] for i in indices[n_train:]]

    # Normalize using training data only
    max_r0 = _compute_max_r0(train_trajs)

    train_ds = FusionDTDataset(train_trajs, num_nodes, context_len, max_r0)
    val_ds = FusionDTDataset(val_trajs, num_nodes, context_len, max_r0)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    info = {
        "num_trajectories": len(trajectories),
        "num_train_trajs": len(train_trajs),
        "num_val_trajs": len(val_trajs),
        "num_train_windows": len(train_ds),
        "num_val_windows": len(val_ds),
        "num_nodes": num_nodes,
        "num_actions": num_nodes + 1,
        "context_len": context_len,
        "max_r0": max_r0,
    }
    return train_loader, val_loader, graph_data, info


if __name__ == "__main__":
    traj_path = "output/trajectories_20260420_a100.pt"
    train_loader, val_loader, graph, info = create_dataloaders(traj_path)

    print("Dataset info:")
    for k, v in info.items():
        print(f"  {k}: {v}")

    batch = next(iter(train_loader))
    print(f"\nBatch shapes:")
    for k, v in batch.items():
        print(f"  {k}: {v.shape}")

    # Verify return-to-go is monotonically non-increasing per window
    rtg = batch["returns_to_go"]
    mask = batch["attn_mask"]
    for i in range(min(3, rtg.shape[0])):
        valid = mask[i].sum().int().item()
        vals = rtg[i, :valid].tolist()
        mono = all(vals[j] >= vals[j + 1] - 1e-6 for j in range(len(vals) - 1))
        print(f"  Window {i}: R_0={vals[0]:.4f}, R_T={vals[-1]:.4f}, "
              f"len={valid}, monotonic={mono}")
