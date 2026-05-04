"""Dynamic graph dataset for Decision Transformer with mutation-aware encoding.

Loads enriched .pt files and constructs per-step contracted graphs G_t.
Each training window contains K consecutive (Data, Metadata) pairs with
variable graph sizes.

Usage:
    from dt_dataset_dynamic import create_dynamic_dataloaders

    train_loader, val_loader, info = create_dynamic_dataloaders(
        enriched_dir="output/dynamic_features",
        gpu_csv_path="output/gpu_profiles_merged.csv",
        context_len=20,
    )
"""

from __future__ import annotations

import csv
import glob
import math
import os
import sys
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch

from dynamic_graph_types import GraphStepMetadata, FidelityReport
from graph_contractor import (
    reconstruct_and_contract_trajectory,
    reconstruct_trajectory_states,
    contract_graph,
)
from precompute_graphs import _dict_to_metadata

# ---------------------------------------------------------------------------
# GPU reward loading (reused from dt_dataset_multi.py)
# ---------------------------------------------------------------------------


def _load_gpu_rewards(csv_path: str) -> Dict[Tuple[str, str], float]:
    """Load GPU kernel times from profiling CSV."""
    times = {}
    if not csv_path or not os.path.exists(csv_path):
        return times
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["module"], row["strategy"])
            times[key] = float(row["avg_kernel_per_iter_us"])
    return times


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_enriched_modules(
    enriched_dir: str,
    gpu_csv_path: Optional[str] = None,
    skip_computation: bool = True,
    skip_unresolved: bool = True,
    include_approximate: bool = True,
    max_trajectories_per_module: int = 0,
    verbose: bool = True,
) -> Tuple[List[dict], Dict]:
    """Load enriched .pt files and prepare trajectory metadata.

    Args:
        enriched_dir: Directory with enriched .pt files
        gpu_csv_path: Path to GPU profiles CSV
        skip_computation: Skip whole_computation modules
        skip_unresolved: Skip trajectories with unresolved graph states
        include_approximate: Include approximate graph states
        max_trajectories_per_module: Limit per module (0 = all)
        verbose: Print progress

    Returns:
        (module_data_list, info_dict)
    """
    gpu_times = _load_gpu_rewards(gpu_csv_path) if gpu_csv_path else {}

    pt_files = sorted(glob.glob(os.path.join(enriched_dir, "*.pt")))
    if verbose:
        print(f"Found {len(pt_files)} enriched .pt files in {enriched_dir}")

    module_data_list = []
    total_trajs = 0
    total_steps = 0
    skipped_modules = 0

    for pt_file in pt_files:
        filename = os.path.basename(pt_file).replace(".pt", "")

        if skip_computation and "computation" in filename.lower():
            skipped_modules += 1
            continue

        data = torch.load(pt_file, map_location="cpu", weights_only=False)
        module_key = data.get("module_key", filename)
        g0 = data["graph"]
        enriched_trajs = data["enriched_trajectories"]

        if max_trajectories_per_module > 0:
            enriched_trajs = enriched_trajs[:max_trajectories_per_module]

        # CSV module key format: Model/module
        csv_module_key = filename.replace("__", "/")

        # Get XLA default time for reward computation
        xla_default_us = gpu_times.get((csv_module_key, "xla_default"))

        valid_trajs = []
        for ti, traj in enumerate(enriched_trajs):
            strategy_detail = traj.get("strategy_detail", "")

            # Compute GPU reward
            gpu_reward = traj.get("reward_gpu")
            if gpu_reward is None and xla_default_us is not None:
                strat_us = gpu_times.get((csv_module_key, strategy_detail))
                if strat_us and strat_us > 0:
                    gpu_reward = xla_default_us / strat_us

            valid_trajs.append({
                "traj_idx": ti,
                "strategy": traj.get("strategy", ""),
                "strategy_detail": strategy_detail,
                "steps": traj["steps"],
                "clusters": traj["clusters"],
                "reward_gpu": gpu_reward,
                "n_steps": len(traj["steps"]),
            })
            total_steps += len(traj["steps"])

        total_trajs += len(valid_trajs)

        # Infer architecture from module key
        arch = filename.split("__")[0].rsplit("-", 1)[0] if "__" in filename else ""

        module_data_list.append({
            "module_key": module_key,
            "filename": filename,
            "csv_module_key": csv_module_key,
            "architecture": arch,
            "g0": g0,
            "trajectories": valid_trajs,
            "num_original_nodes": g0.num_nodes,
            "pt_path": pt_file,
        })

        if verbose:
            print(f"  {module_key}: {g0.num_nodes} nodes, "
                  f"{len(valid_trajs)} trajs, {sum(t['n_steps'] for t in valid_trajs)} steps")

    info = {
        "num_modules": len(module_data_list),
        "num_trajectories": total_trajs,
        "total_steps": total_steps,
        "skipped_modules": skipped_modules,
    }

    if verbose:
        print(f"\nLoaded {info['num_modules']} modules, "
              f"{total_trajs} trajectories, {total_steps} steps")

    return module_data_list, info


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class DynamicFusionDTDataset(Dataset):
    """Sliding-window dataset with per-step contracted graphs.

    Each item is a window of K consecutive steps, where each step has:
    - A contracted PyG Data (G_t)
    - Action index (into contracted graph)
    - RTG value
    - GraphStepMetadata (kept outside Data)

    Graphs are constructed lazily and cached.
    """

    def __init__(
        self,
        module_data_list: List[dict],
        context_len: int = 20,
        reward_mode: str = "gpu",
        max_gpu_reward: float = 1.0,
        include_approximate: bool = True,
        include_unresolved: bool = False,
        precomputed_dir: Optional[str] = None,
    ):
        self.context_len = context_len
        self.reward_mode = reward_mode
        self.max_gpu_reward = max(max_gpu_reward, 1e-6)
        self.include_approximate = include_approximate
        self.include_unresolved = include_unresolved

        # Load pre-computed graphs if available
        self._precomputed: Dict[str, dict] = {}  # filename -> {graphs, metadatas, traj_map}
        if precomputed_dir:
            self._load_precomputed(precomputed_dir, module_data_list)

        # Build flat list of (module_idx, traj_idx, start_pos) windows
        self.windows: List[Tuple[int, int, int]] = []
        self.module_data = module_data_list

        # Pre-compute per-trajectory RTGs
        self._rtg_cache: Dict[Tuple[int, int], List[float]] = {}

        for mi, mdata in enumerate(module_data_list):
            for ti, traj in enumerate(mdata["trajectories"]):
                T = traj["n_steps"]
                if T == 0:
                    continue

                # Compute RTG for this trajectory
                rtg = self._compute_rtg(traj, reward_mode)
                self._rtg_cache[(mi, ti)] = rtg

                # Create sliding windows
                for start in range(T):
                    self.windows.append((mi, ti, start))

        # Graph contraction cache: (module_key, traj_idx, step_pos) -> (Data, Metadata)
        self._graph_cache: Dict[Tuple[str, int, int], Tuple[Data, GraphStepMetadata]] = {}
        self._cache_max_size = 50000

    def _load_precomputed(self, precomputed_dir: str, module_data_list: List[dict]):
        """Load pre-computed contracted graphs from disk."""
        for mdata in module_data_list:
            filename = mdata["filename"]
            pc_path = os.path.join(precomputed_dir, f"{filename}.pt")
            if os.path.exists(pc_path):
                self._precomputed[filename] = torch.load(
                    pc_path, map_location="cpu", weights_only=False
                )

    def _compute_rtg(self, traj: dict, reward_mode: str) -> List[float]:
        """Compute return-to-go for each step."""
        steps = traj["steps"]
        T = len(steps)

        # Cost-model per-step rewards
        rewards_raw = [s.get("us_unfused", 0) - s.get("us_fused", 0) for s in steps]
        cost_rtg = [0.0] * T
        cost_rtg[T - 1] = rewards_raw[T - 1]
        for t in range(T - 2, -1, -1):
            cost_rtg[t] = rewards_raw[t] + cost_rtg[t + 1]
        cost_r0 = cost_rtg[0] if cost_rtg[0] != 0 else 1.0

        gpu_r = traj.get("reward_gpu")

        if reward_mode == "gpu" and gpu_r is not None:
            r = gpu_r / self.max_gpu_reward
            return [r] * T
        elif reward_mode == "gpu_progressive" and gpu_r is not None:
            r = gpu_r / self.max_gpu_reward
            abs_r0 = abs(cost_r0)
            if abs_r0 > 1.0:
                return [(cr / abs_r0) * r for cr in cost_rtg]
            else:
                return [r * math.sqrt(max(0, T - t) / T) for t in range(T)]
        elif reward_mode == "cost_model":
            return [cr / abs(cost_r0) if cost_r0 != 0 else 0 for cr in cost_rtg]
        else:
            # Fallback: flat reward = 1.0
            return [1.0] * T

    def _get_contracted_graph(
        self, mi: int, ti: int, step_pos: int,
    ) -> Tuple[Data, GraphStepMetadata]:
        """Get or build contracted graph for a specific step."""
        mdata = self.module_data[mi]
        cache_key = (mdata["module_key"], ti, step_pos)

        if cache_key in self._graph_cache:
            return self._graph_cache[cache_key]

        # Try pre-computed data first
        filename = mdata["filename"]
        if filename in self._precomputed:
            pc = self._precomputed[filename]
            traj_map = pc["traj_map"]
            # traj_map keys may be strings after torch.save/load
            ti_key = str(ti) if str(ti) in traj_map else ti
            if ti_key in traj_map:
                step_map = traj_map[ti_key]
                sp_key = str(step_pos) if str(step_pos) in step_map else step_pos
                if sp_key in step_map:
                    flat_idx = step_map[sp_key]
                    data = pc["graphs"][flat_idx]
                    metadata = _dict_to_metadata(pc["metadatas"][flat_idx])
                    self._graph_cache[cache_key] = (data, metadata)
                    return data, metadata

        traj = mdata["trajectories"][ti]
        g0 = mdata["g0"]

        # Build cluster features cache
        cluster_cache = {}
        for cname, cdata in traj.get("clusters", {}).items():
            if "features" in cdata and isinstance(cdata["features"], torch.Tensor):
                cluster_cache[cname] = cdata["features"]

        # Reconstruct state at this step
        states = reconstruct_trajectory_states(
            traj, g0, mdata["module_key"], ti,
        )

        # Contract all steps and cache them
        for state in states:
            sk = (mdata["module_key"], ti, state.step_position)
            if sk not in self._graph_cache:
                data, metadata = contract_graph(g0, state, cluster_cache)
                if len(self._graph_cache) < self._cache_max_size:
                    self._graph_cache[sk] = (data, metadata)
                if state.step_position == step_pos:
                    result = (data, metadata)

        return self._graph_cache.get(cache_key, result)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx: int) -> dict:
        mi, ti, start = self.windows[idx]
        mdata = self.module_data[mi]
        traj = mdata["trajectories"][ti]
        T = traj["n_steps"]
        K = self.context_len
        rtg_list = self._rtg_cache[(mi, ti)]

        # Collect K steps starting from 'start'
        graphs: List[Data] = []
        metadatas: List[GraphStepMetadata] = []
        actions: List[int] = []
        rtgs: List[float] = []
        timesteps: List[int] = []
        attn_mask: List[int] = []

        for k in range(K):
            pos = start + k
            if pos >= T:
                # Padding: use a dummy single-node graph
                dummy = Data(
                    x=torch.zeros(1, mdata["g0"].x.shape[1]),
                    edge_index=torch.zeros(2, 0, dtype=torch.long),
                    opcode_ids=torch.zeros(1, dtype=torch.long),
                    cluster_features=torch.zeros(1, 34),
                    is_cluster=torch.zeros(1, dtype=torch.bool),
                    candidate_mask=torch.zeros(1, dtype=torch.bool),
                    num_nodes=1,
                )
                dummy_meta = GraphStepMetadata(
                    module_key=mdata["module_key"],
                    trajectory_idx=ti,
                    step_idx=-1,
                    step_position=-1,
                    idx_to_name=["PAD"],
                    name_to_idx={"PAD": 0},
                    cluster_members={},
                    candidate_names=[],
                    selected_name=None,
                    selected_idx=None,
                    candidate_indices=[],
                    graph_fidelity="pad",
                    notes=[],
                    is_cluster=[False],
                )
                graphs.append(dummy)
                metadatas.append(dummy_meta)
                actions.append(-1)  # will be masked
                rtgs.append(0.0)
                timesteps.append(0)
                attn_mask.append(0)
            else:
                data, metadata = self._get_contracted_graph(mi, ti, pos)
                graphs.append(data)
                metadatas.append(metadata)

                # Action: index of selected node in contracted graph
                sel_idx = metadata.selected_idx if metadata.selected_idx is not None else -1
                actions.append(sel_idx)
                rtgs.append(rtg_list[pos])
                timesteps.append(pos)
                attn_mask.append(1)

        return {
            "graphs": graphs,
            "metadatas": metadatas,
            "actions": torch.tensor(actions, dtype=torch.long),
            "rtgs": torch.tensor(rtgs, dtype=torch.float32),
            "timesteps": torch.tensor(timesteps, dtype=torch.long),
            "attn_mask": torch.tensor(attn_mask, dtype=torch.long),
            "module_key": mdata["module_key"],
            "architecture": mdata["architecture"],
            "trajectory_idx": ti,
            "reward_gpu": traj.get("reward_gpu"),
        }


# ---------------------------------------------------------------------------
# Collation
# ---------------------------------------------------------------------------


def dynamic_collate_fn(batch: List[dict]) -> dict:
    """Collate dynamic graph windows into a batch.

    Flattens B*K PyG Data objects into a single Batch.
    Keeps metadata as List[List[GraphStepMetadata]].
    Pads actions and candidate masks to [B, K].
    """
    B = len(batch)
    K = batch[0]["actions"].shape[0]

    # 1. Flatten all B*K PyG graphs into one Batch
    all_graphs = []
    graph_sizes = []  # num_nodes per graph
    for item in batch:
        for g in item["graphs"]:
            all_graphs.append(g)
            graph_sizes.append(g.num_nodes)

    pyg_batch = Batch.from_data_list(all_graphs)
    graph_sizes_tensor = torch.tensor(graph_sizes, dtype=torch.long)  # [B*K]

    # 2. Keep metadata outside Batch
    all_metadatas = [item["metadatas"] for item in batch]  # [B][K]

    # 3. Stack actions, rtgs, timesteps, attn_mask → [B, K]
    actions = torch.stack([item["actions"] for item in batch])
    rtgs = torch.stack([item["rtgs"] for item in batch])
    timesteps = torch.stack([item["timesteps"] for item in batch])
    attn_mask = torch.stack([item["attn_mask"] for item in batch])

    # 4. Build per-graph candidate masks padded to max_Vt
    max_vt = max(graph_sizes)
    candidate_masks = torch.zeros(B * K, max_vt, dtype=torch.bool)
    for i, g in enumerate(all_graphs):
        if hasattr(g, "candidate_mask"):
            n = g.candidate_mask.shape[0]
            candidate_masks[i, :n] = g.candidate_mask

    candidate_masks = candidate_masks.view(B, K, max_vt)

    # 5. Module and trajectory info
    module_keys = [item["module_key"] for item in batch]
    architectures = [item["architecture"] for item in batch]
    gpu_rewards = [item.get("reward_gpu") for item in batch]

    return {
        "pyg_batch": pyg_batch,
        "graph_sizes": graph_sizes_tensor,  # [B*K]
        "metadatas": all_metadatas,         # List[List[GraphStepMetadata]]
        "actions": actions,                 # [B, K]
        "rtgs": rtgs,                       # [B, K]
        "timesteps": timesteps,             # [B, K]
        "attn_mask": attn_mask,             # [B, K]
        "candidate_masks": candidate_masks, # [B, K, max_Vt]
        "max_vt": max_vt,
        "module_keys": module_keys,
        "architectures": architectures,
        "gpu_rewards": gpu_rewards,
    }


# ---------------------------------------------------------------------------
# Dataloader creation
# ---------------------------------------------------------------------------


def create_dynamic_dataloaders(
    enriched_dir: str,
    gpu_csv_path: Optional[str] = None,
    context_len: int = 20,
    reward_mode: str = "gpu",
    batch_size: int = 32,
    val_split: float = 0.15,
    holdout_arch: Optional[str] = None,
    skip_computation: bool = True,
    include_approximate: bool = True,
    include_unresolved: bool = False,
    max_trajectories_per_module: int = 0,
    num_workers: int = 0,
    precomputed_dir: Optional[str] = None,
    verbose: bool = True,
) -> Tuple[DataLoader, DataLoader, Dict]:
    """Create train/val dataloaders with dynamic graph contraction.

    Args:
        enriched_dir: Directory with enriched .pt files
        gpu_csv_path: Path to GPU profiles CSV
        context_len: Sliding window size K
        reward_mode: "gpu", "gpu_progressive", or "cost_model"
        batch_size: Batch size
        val_split: Validation split fraction (by trajectory)
        holdout_arch: Hold out all modules from this architecture for validation
        skip_computation: Skip whole_computation modules
        include_approximate: Include approximate graph fidelity states
        include_unresolved: Include unresolved graph fidelity states
        max_trajectories_per_module: Limit per module (0 = all)
        num_workers: DataLoader workers
        verbose: Print progress

    Returns:
        (train_loader, val_loader, info)
    """
    module_data_list, info = load_enriched_modules(
        enriched_dir=enriched_dir,
        gpu_csv_path=gpu_csv_path,
        skip_computation=skip_computation,
        include_approximate=include_approximate,
        max_trajectories_per_module=max_trajectories_per_module,
        verbose=verbose,
    )

    if not module_data_list:
        raise ValueError(f"No modules loaded from {enriched_dir}")

    # Compute max GPU reward for normalization
    all_gpu_rewards = []
    for mdata in module_data_list:
        for traj in mdata["trajectories"]:
            if traj.get("reward_gpu") is not None:
                all_gpu_rewards.append(traj["reward_gpu"])
    max_gpu_reward = max(all_gpu_rewards) if all_gpu_rewards else 1.0

    # Split into train/val by architecture holdout or random trajectory split
    if holdout_arch:
        train_modules = [m for m in module_data_list
                         if holdout_arch.lower() not in m["architecture"].lower()]
        val_modules = [m for m in module_data_list
                       if holdout_arch.lower() in m["architecture"].lower()]
        if verbose:
            print(f"\nHoldout {holdout_arch}: "
                  f"{len(train_modules)} train modules, {len(val_modules)} val modules")
    else:
        # Split trajectories within each module
        train_modules = []
        val_modules = []
        for mdata in module_data_list:
            trajs = mdata["trajectories"]
            n_val = max(1, int(len(trajs) * val_split))
            n_train = len(trajs) - n_val

            train_m = dict(mdata)
            train_m["trajectories"] = trajs[:n_train]
            val_m = dict(mdata)
            val_m["trajectories"] = trajs[n_train:]

            if train_m["trajectories"]:
                train_modules.append(train_m)
            if val_m["trajectories"]:
                val_modules.append(val_m)

    train_ds = DynamicFusionDTDataset(
        train_modules,
        context_len=context_len,
        reward_mode=reward_mode,
        max_gpu_reward=max_gpu_reward,
        include_approximate=include_approximate,
        include_unresolved=include_unresolved,
        precomputed_dir=precomputed_dir,
    )
    val_ds = DynamicFusionDTDataset(
        val_modules,
        context_len=context_len,
        reward_mode=reward_mode,
        max_gpu_reward=max_gpu_reward,
        include_approximate=include_approximate,
        include_unresolved=include_unresolved,
        precomputed_dir=precomputed_dir,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=dynamic_collate_fn,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=dynamic_collate_fn,
        num_workers=num_workers,
    )

    info.update({
        "context_len": context_len,
        "reward_mode": reward_mode,
        "max_gpu_reward": max_gpu_reward,
        "train_windows": len(train_ds),
        "val_windows": len(val_ds),
        "holdout_arch": holdout_arch,
    })

    if verbose:
        print(f"\nTrain: {len(train_ds)} windows, Val: {len(val_ds)} windows")

    return train_loader, val_loader, info
