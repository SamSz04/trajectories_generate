"""Dataset and collation for the dynamic priority scorer.

Each sample is one MDP decision step: a candidate set with features,
a selected candidate index, and optional advantage weight.

Loads enriched .pt files produced by cluster_features.py and assembles
per-step training samples with variable-size candidate sets, padded
per batch via a custom collate function.

Usage:
    # Test with one enriched .pt file
    python scorer_dataset.py --data-dir output/dynamic_features --max-cands 50
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cluster_features import CLUSTER_FEATURE_DIM
from scorer_model import ORIGINAL_FEATURE_DIM, CONTEXT_INPUT_DIM


# ======================================================================
# Constants
# ======================================================================

MAX_CANDIDATES = 64    # max candidates per step (subsample if larger)
NEG_SAMPLE_K = 32      # negatives to sample when candidate set too large


# ======================================================================
# Dataset
# ======================================================================

class ScorerDataset(Dataset):
    """Per-step candidate-ranking dataset.

    Each item is a dict with:
      - orig_features:    [N_orig, 27]
      - orig_opcode_ids:  [N_orig] long
      - orig_mask:        [N_orig] bool
      - cluster_features: [N_clust, 34]
      - cluster_mask:     [N_clust] bool
      - context:          [5]
      - target_idx:       long scalar — index of selected candidate
      - weight:           float scalar — advantage weight (for weighted training)
      - is_fusion_selected: bool — whether the selected candidate is fusion.N
      - module_key:       str
    """

    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        holdout_arch: Optional[str] = None,
        max_candidates: int = MAX_CANDIDATES,
        training_mode: str = 'imitation_all',
        top_quantile: float = 0.25,
    ):
        """
        Args:
            data_dir: Directory with enriched .pt files from cluster_features.py
            split: 'train' or 'val' (80/20 split by strategy index)
            holdout_arch: Architecture name to hold out (for LOO evaluation)
            max_candidates: Maximum candidate set size (subsample if larger)
            training_mode: 'imitation_all', 'advantage_weighted', or 'top_quantile'
            top_quantile: Fraction of best strategies to keep (for top_quantile mode)
        """
        self.max_candidates = max_candidates
        self.training_mode = training_mode

        # Load all enriched .pt files
        data_dir = Path(data_dir)
        pt_files = sorted(data_dir.glob("*.pt"))
        if not pt_files:
            raise FileNotFoundError(f"No .pt files in {data_dir}")

        self.samples: List[dict] = []
        self._load_all(pt_files, split, holdout_arch, training_mode, top_quantile)

    def _load_all(
        self,
        pt_files: List[Path],
        split: str,
        holdout_arch: Optional[str],
        training_mode: str,
        top_quantile: float,
    ):
        """Load and flatten all modules into per-step samples."""
        for pt_file in pt_files:
            module_key = pt_file.stem
            if module_key in ('summary',):
                continue

            # LOO holdout
            if holdout_arch:
                model_name = module_key.split('__')[0]
                is_holdout = holdout_arch.lower() in model_name.lower()
                if split == 'train' and is_holdout:
                    continue
                if split == 'val' and not is_holdout:
                    continue

            data = torch.load(str(pt_file), weights_only=False)
            graph = data['graph']
            trajectories = data.get('enriched_trajectories', [])

            if not trajectories:
                continue

            # Build name → idx mapping for original nodes
            name_to_idx = {
                n: i for i, n in enumerate(graph.node_names)
            }

            # Filter trajectories by training mode
            if training_mode == 'top_quantile':
                trajectories = self._filter_top_quantile(
                    trajectories, top_quantile)

            # Split trajectories by index (80/20)
            if not holdout_arch:
                n = len(trajectories)
                split_idx = int(0.8 * n)
                if split == 'train':
                    trajectories = trajectories[:split_idx]
                else:
                    trajectories = trajectories[split_idx:]

            # Compute advantage weights for the module
            rewards = [
                t.get('reward_gpu') for t in trajectories
                if t.get('reward_gpu') is not None
            ]
            xla_reward = None
            for t in trajectories:
                if t.get('strategy') == 'xla_default' and t.get('reward_gpu'):
                    xla_reward = t['reward_gpu']
                    break
            if xla_reward is None and rewards:
                xla_reward = 1.0  # xla_default reward = xla_us / xla_us = 1.0

            # Extract per-step samples
            for traj in trajectories:
                weight = self._compute_weight(
                    traj, xla_reward, training_mode)

                self._extract_steps(
                    traj, graph, name_to_idx, module_key, weight)

    def _filter_top_quantile(
        self, trajectories: list, quantile: float,
    ) -> list:
        """Keep only the top `quantile` fraction by GPU reward."""
        with_reward = [t for t in trajectories if t.get('reward_gpu')]
        if not with_reward:
            return trajectories

        rewards = sorted([t['reward_gpu'] for t in with_reward], reverse=True)
        cutoff = rewards[max(0, int(len(rewards) * quantile) - 1)]
        return [t for t in trajectories if
                t.get('reward_gpu') is not None and t['reward_gpu'] >= cutoff]

    def _compute_weight(
        self,
        traj: dict,
        xla_reward: Optional[float],
        training_mode: str,
    ) -> float:
        """Compute sample weight based on training mode."""
        if training_mode != 'advantage_weighted':
            return 1.0

        reward = traj.get('reward_gpu')
        if reward is None or xla_reward is None:
            return 1.0

        # advantage = (xla_us - strat_us) / xla_us
        # = 1 - 1/reward  (since reward = xla_us / strat_us)
        advantage = 1.0 - 1.0 / reward if reward > 0 else 0.0
        return min(max(math.exp(3.0 * advantage), 0.1), 5.0)

    def _extract_steps(
        self,
        traj: dict,
        graph,
        name_to_idx: Dict[str, int],
        module_key: str,
        weight: float,
    ):
        """Convert enriched trajectory steps into training samples."""
        clusters = traj.get('clusters', {})
        steps = traj.get('steps', [])
        total_steps = len(steps)

        if total_steps == 0:
            return

        n_original = graph.num_nodes
        n_fused_so_far = 0
        n_fusions_created = 0

        for step in steps:
            cand_names = step['candidate_names']
            selected_name = step['selected_name']

            # Separate original vs cluster candidates
            orig_names = [c for c in cand_names if not c.startswith('fusion.')]
            clust_names = [c for c in cand_names if c.startswith('fusion.')]

            # Subsample if too many candidates
            if len(cand_names) > self.max_candidates:
                orig_names, clust_names = self._subsample(
                    orig_names, clust_names, selected_name,
                    self.max_candidates,
                )

            # Build original node features
            orig_features = []
            orig_opcode_ids = []
            for name in orig_names:
                idx = name_to_idx.get(name, -1)
                if idx >= 0:
                    orig_features.append(graph.x[idx])
                    orig_opcode_ids.append(graph.opcode_ids[idx].item())
                else:
                    orig_features.append(torch.zeros(ORIGINAL_FEATURE_DIM))
                    orig_opcode_ids.append(0)

            # Build cluster features
            clust_feats = []
            for name in clust_names:
                cd = clusters.get(name, {})
                feat = cd.get('features', torch.zeros(CLUSTER_FEATURE_DIM))
                # Update steps_since_creation (index 33)
                if isinstance(feat, torch.Tensor):
                    feat = feat.clone()
                    creation_step = cd.get('creation_step', 0)
                    feat[33] = float(max(0, step['step_idx'] - creation_step))
                clust_feats.append(feat)

            # Build context features
            n_alive = len(orig_names) + len(clust_names)
            n_fus_cands = len(clust_names)
            context = torch.tensor([
                step['step_idx'] / max(total_steps, 1),      # step_progress
                n_fused_so_far / max(n_original, 1),          # frac_fused
                n_alive / max(n_original, 1),                 # frac_alive
                n_fus_cands / max(n_alive, 1),                # frac_fusion_cands
                math.log2(n_fusions_created + 1),             # log2(n_fusions+1)
            ], dtype=torch.float32)

            # Find target index in combined list
            all_names = orig_names + clust_names
            try:
                target_idx = all_names.index(selected_name)
            except ValueError:
                continue  # skip if selected not in subsampled set

            is_fusion_sel = selected_name.startswith('fusion.')

            # Stack tensors
            if orig_features:
                orig_feat_t = torch.stack(orig_features)
                orig_opcode_t = torch.tensor(orig_opcode_ids, dtype=torch.long)
            else:
                orig_feat_t = torch.zeros(0, ORIGINAL_FEATURE_DIM)
                orig_opcode_t = torch.zeros(0, dtype=torch.long)

            if clust_feats:
                clust_feat_t = torch.stack(clust_feats)
            else:
                clust_feat_t = torch.zeros(0, CLUSTER_FEATURE_DIM)

            self.samples.append({
                'orig_features': orig_feat_t,
                'orig_opcode_ids': orig_opcode_t,
                'cluster_features': clust_feat_t,
                'context': context,
                'target_idx': target_idx,
                'weight': weight,
                'is_fusion_selected': is_fusion_sel,
                'module_key': module_key,
                'n_orig': len(orig_names),
                'n_clust': len(clust_names),
            })

            # Update running state
            n_fused_so_far += 1
            if is_fusion_sel:
                pass  # existing fusion dequeued
            # Track new fusions (selected node might create one)
            # This is approximate — we just count fusion candidates
            n_fusions_created = len([c for c in cand_names
                                     if c.startswith('fusion.')])

    def _subsample(
        self,
        orig_names: List[str],
        clust_names: List[str],
        selected_name: str,
        max_total: int,
    ) -> Tuple[List[str], List[str]]:
        """Subsample candidates to max_total, always keeping selected."""
        # Split budget roughly proportionally
        total = len(orig_names) + len(clust_names)
        orig_budget = max(1, int(max_total * len(orig_names) / total))
        clust_budget = max_total - orig_budget

        def subsample_list(names: list, budget: int, sel: str) -> list:
            if len(names) <= budget:
                return names
            # Always keep selected
            keep = [sel] if sel in names else []
            remaining = [n for n in names if n != sel]
            # Random sample
            perm = torch.randperm(len(remaining))[:budget - len(keep)]
            keep.extend(remaining[i] for i in perm.tolist())
            return keep

        orig_sub = subsample_list(orig_names, orig_budget, selected_name)
        clust_sub = subsample_list(clust_names, clust_budget, selected_name)
        return orig_sub, clust_sub

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        return self.samples[idx]


# ======================================================================
# Collate function
# ======================================================================

def scorer_collate_fn(batch: List[dict]) -> dict:
    """Collate variable-size candidate sets into padded batch tensors.

    Returns a dict with:
      - orig_features:    [B, max_N_orig, 27]
      - orig_opcode_ids:  [B, max_N_orig] long
      - orig_mask:        [B, max_N_orig] bool
      - cluster_features: [B, max_N_clust, 34]
      - cluster_mask:     [B, max_N_clust] bool
      - context:          [B, 5]
      - target_idx:       [B] long
      - weight:           [B] float
      - is_fusion_selected: [B] bool
    """
    B = len(batch)
    max_orig = max(s['n_orig'] for s in batch) or 1
    max_clust = max(s['n_clust'] for s in batch) or 1

    orig_features = torch.zeros(B, max_orig, ORIGINAL_FEATURE_DIM)
    orig_opcode_ids = torch.zeros(B, max_orig, dtype=torch.long)
    orig_mask = torch.zeros(B, max_orig, dtype=torch.bool)

    cluster_features = torch.zeros(B, max_clust, CLUSTER_FEATURE_DIM)
    cluster_mask = torch.zeros(B, max_clust, dtype=torch.bool)

    context = torch.zeros(B, CONTEXT_INPUT_DIM)
    target_idx = torch.zeros(B, dtype=torch.long)
    weight = torch.zeros(B)
    is_fusion = torch.zeros(B, dtype=torch.bool)

    for i, s in enumerate(batch):
        no = s['n_orig']
        nc = s['n_clust']

        if no > 0:
            orig_features[i, :no] = s['orig_features']
            orig_opcode_ids[i, :no] = s['orig_opcode_ids']
            orig_mask[i, :no] = True

        if nc > 0:
            cluster_features[i, :nc] = s['cluster_features']
            cluster_mask[i, :nc] = True

        context[i] = s['context']
        target_idx[i] = s['target_idx']
        weight[i] = s['weight']
        is_fusion[i] = s['is_fusion_selected']

    return {
        'orig_features': orig_features,
        'orig_opcode_ids': orig_opcode_ids,
        'orig_mask': orig_mask,
        'cluster_features': cluster_features,
        'cluster_mask': cluster_mask,
        'context': context,
        'target_idx': target_idx,
        'weight': weight,
        'is_fusion_selected': is_fusion,
    }


# ======================================================================
# Utilities
# ======================================================================

def build_dataloaders(
    data_dir: str,
    batch_size: int = 64,
    holdout_arch: Optional[str] = None,
    training_mode: str = 'imitation_all',
    max_candidates: int = MAX_CANDIDATES,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    """Build train and val dataloaders."""
    train_ds = ScorerDataset(
        data_dir, split='train',
        holdout_arch=holdout_arch,
        max_candidates=max_candidates,
        training_mode=training_mode,
    )
    val_ds = ScorerDataset(
        data_dir, split='val',
        holdout_arch=holdout_arch,
        max_candidates=max_candidates,
        training_mode='imitation_all',  # val always unweighted
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=scorer_collate_fn, num_workers=num_workers,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        collate_fn=scorer_collate_fn, num_workers=num_workers,
    )

    return train_loader, val_loader


# ======================================================================
# CLI: test loading
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Test scorer dataset loading")
    parser.add_argument("--data-dir", required=True,
                        help="Directory with enriched .pt files")
    parser.add_argument("--max-cands", type=int, default=MAX_CANDIDATES)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--training-mode", default='imitation_all',
                        choices=['imitation_all', 'advantage_weighted',
                                 'top_quantile'])
    args = parser.parse_args()

    print(f"Loading dataset from {args.data_dir}...")
    try:
        train_ds = ScorerDataset(
            args.data_dir, split='train',
            max_candidates=args.max_cands,
            training_mode=args.training_mode,
        )
        val_ds = ScorerDataset(
            args.data_dir, split='val',
            max_candidates=args.max_cands,
        )
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        return

    print(f"Train samples: {len(train_ds)}")
    print(f"Val samples:   {len(val_ds)}")

    if not train_ds.samples:
        print("No samples loaded. Check data directory.")
        return

    # Stats
    n_fusion_sel = sum(s['is_fusion_selected'] for s in train_ds.samples)
    n_total = len(train_ds.samples)
    print(f"Fusion selections: {n_fusion_sel}/{n_total} "
          f"({100*n_fusion_sel/n_total:.1f}%)")

    # Candidate set size distribution
    orig_sizes = [s['n_orig'] for s in train_ds.samples]
    clust_sizes = [s['n_clust'] for s in train_ds.samples]
    total_sizes = [o + c for o, c in zip(orig_sizes, clust_sizes)]
    print(f"Candidate sets: mean={sum(total_sizes)/len(total_sizes):.1f}, "
          f"max={max(total_sizes)}, min={min(total_sizes)}")
    print(f"  Original: mean={sum(orig_sizes)/len(orig_sizes):.1f}")
    print(f"  Clusters: mean={sum(clust_sizes)/len(clust_sizes):.1f}")

    # Test batch
    loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=scorer_collate_fn,
    )
    batch = next(iter(loader))
    print(f"\nBatch shapes:")
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k:25s}: {v.shape} {v.dtype}")

    # Verify target indices are valid
    n_total_cands = batch['orig_mask'].sum(dim=1) + batch['cluster_mask'].sum(dim=1)
    for i in range(args.batch_size):
        assert batch['target_idx'][i] < n_total_cands[i], \
            f"target_idx {batch['target_idx'][i]} >= n_cands {n_total_cands[i]}"
    print(f"  Target indices valid!")

    # Per-module breakdown
    modules = {}
    for s in train_ds.samples:
        modules.setdefault(s['module_key'], 0)
        modules[s['module_key']] += 1
    print(f"\nModules ({len(modules)}):")
    for m, c in sorted(modules.items(), key=lambda x: -x[1]):
        print(f"  {m}: {c} samples")

    print("\nAll checks passed!")


if __name__ == "__main__":
    main()
