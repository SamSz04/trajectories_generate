"""Learn a parametric priority formula for XLA fusion.

Trains a simple model (linear or tiny MLP) on enriched trajectory data
to predict which candidate should be selected at each fusion step.
Exports learned weights to JSON for deployment in XLA C++.

The key insight: the formula scores ALL candidates (including fusion.N
clusters) dynamically at every step, unlike the static json_plan interface.

Usage:
    # Train linear model
    python learn_priority.py train \
        --data-dir output/dynamic_features \
        --model-type linear \
        --output output/learned_weights.json

    # Train tiny MLP
    python learn_priority.py train \
        --data-dir output/dynamic_features \
        --model-type mlp --hidden-dim 16 \
        --output output/learned_weights.json

    # Evaluate learned weights
    python learn_priority.py eval \
        --data-dir output/dynamic_features \
        --weights output/learned_weights.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cluster_features import CLUSTER_FEATURE_DIM


# ======================================================================
# Constants
# ======================================================================

UNIVERSAL_FEATURE_DIM = 10

# Indices into data.x (19-dim)
_X_BYTES = 12       # log2(total_bytes)
_X_ELEMENTS = 11    # log2(total_elements)
_X_RANK = 6         # tensor rank
_X_FAN_IN = 13      # operand count (raw)
_X_FAN_OUT = 14     # user count (raw)
_X_IS_FUSION = 16   # is fusion node
_X_IS_FUSABLE = 18  # is fusable

# Cluster feature indices (34-dim)
_C_SIZE = 0         # log2(cluster_size)
_C_BYTES = 11       # log2(total_bytes)
_C_ELEMENTS = 12    # log2(total_elements)
_C_RANK = 13        # max_rank
_C_FAN_IN = 14      # log2(cluster_fan_in + 1)
_C_FAN_OUT = 15     # log2(cluster_fan_out + 1)
_C_REDUCE = 17      # contains_reduce
_C_ELEMENTWISE = 20 # contains_elementwise

# Elementwise opcodes (from feature_encoder.py)
ELEMENTWISE_OPCODES = {
    'add', 'subtract', 'multiply', 'divide', 'maximum', 'minimum',
    'select', 'compare', 'and', 'or', 'xor', 'negate', 'abs',
    'ceil', 'floor', 'clamp', 'convert', 'copy', 'exp', 'log',
    'sqrt', 'rsqrt', 'tanh', 'sine', 'cosine', 'sign',
}

# XLA opcode vocabulary (from feature_encoder.py)
OPCODE_VOCAB = [
    "abs", "acos", "acosh", "add", "add-dependency", "after-all",
    "all-gather", "all-gather-done", "all-gather-start", "all-reduce",
    "all-reduce-done", "all-reduce-start", "all-to-all", "and",
    "asin", "asinh", "async-done", "async-start", "async-update",
    "atan", "atan2", "atanh", "batch-norm-grad", "batch-norm-inference",
    "batch-norm-training", "bitcast", "bitcast-convert", "broadcast",
    "call", "cbrt", "ceil", "cholesky", "clamp", "collective-broadcast",
    "collective-permute", "collective-permute-done", "collective-permute-start",
    "compare", "complex", "concatenate", "conditional", "constant",
    "convert", "convolution", "copy", "copy-done", "copy-start", "cos",
    "custom-call", "divide", "domain", "dot", "dynamic-reshape",
    "dynamic-slice", "dynamic-update-slice", "erf", "exp", "expm1",
    "fft", "floor", "fusion", "gather", "get-dimension-size",
    "get-tuple-element", "imag", "infeed", "iota", "is-finite",
    "log", "log1p", "logistic", "map", "maximum", "minimum",
    "multiply", "negate", "not", "opt-barrier", "or", "outfeed",
    "pad", "parameter", "partition-id", "popcnt", "power", "real",
    "recv", "recv-done", "reduce", "reduce-precision", "reduce-scatter",
    "reduce-window", "remainder", "replica-id", "reshape", "reverse",
    "rng", "rng-bit-generator", "rng-get-and-update-state",
    "round-nearest-afz", "round-nearest-even", "rsqrt", "scatter",
    "select", "send", "send-done", "set-dimension-size", "shift-left",
    "shift-right-arithmetic", "shift-right-logical", "sign", "sin",
    "slice", "sort", "sqrt", "stochastic-convert", "subtract", "tan",
    "tanh", "top-k", "transpose", "triangular-solve", "tuple",
    "while", "xor",
]
OPCODE_TO_IDX = {op: i for i, op in enumerate(OPCODE_VOCAB)}


# ======================================================================
# Feature extraction
# ======================================================================

def extract_features_orig(
    data_x_row: torch.Tensor,  # [19]
    opcode_id: int,
) -> torch.Tensor:
    """Extract 10-dim universal features from an original HLO node."""
    feat = torch.zeros(UNIVERSAL_FEATURE_DIM)
    feat[0] = data_x_row[_X_BYTES].item()                           # log2(bytes) already
    feat[1] = data_x_row[_X_ELEMENTS].item()                        # log2(elements) already
    feat[2] = data_x_row[_X_RANK].item()                            # rank
    feat[3] = math.log2(max(data_x_row[_X_FAN_OUT].item(), 0) + 1)  # log2(fan_out+1)
    feat[4] = math.log2(max(data_x_row[_X_FAN_IN].item(), 0) + 1)   # log2(fan_in+1)

    # is_elementwise: check opcode name
    opname = OPCODE_VOCAB[opcode_id] if 0 <= opcode_id < len(OPCODE_VOCAB) else ''
    feat[5] = 1.0 if opname in ELEMENTWISE_OPCODES else 0.0
    feat[6] = data_x_row[_X_IS_FUSION].item()                       # is_fusion
    feat[7] = data_x_row[_X_ELEMENTS].item()                        # approx flops ≈ elements
    feat[8] = 1.0 if opname == 'reduce' else 0.0                    # is_reduce
    feat[9] = 1.0                                                    # fused_instr_count = 1
    return feat


def extract_features_cluster(
    cluster_feat: torch.Tensor,  # [34]
) -> torch.Tensor:
    """Extract 10-dim universal features from a fusion.N cluster."""
    feat = torch.zeros(UNIVERSAL_FEATURE_DIM)
    feat[0] = cluster_feat[_C_BYTES].item()       # log2(total_bytes)
    feat[1] = cluster_feat[_C_ELEMENTS].item()     # log2(total_elements)
    feat[2] = cluster_feat[_C_RANK].item()         # max_rank
    feat[3] = cluster_feat[_C_FAN_OUT].item()      # log2(fan_out+1) already
    feat[4] = cluster_feat[_C_FAN_IN].item()       # log2(fan_in+1) already
    feat[5] = cluster_feat[_C_ELEMENTWISE].item()  # contains_elementwise
    feat[6] = 1.0                                  # is_fusion (always)
    feat[7] = cluster_feat[_C_ELEMENTS].item()     # approx flops ≈ elements
    feat[8] = cluster_feat[_C_REDUCE].item()       # contains_reduce
    feat[9] = 2.0 ** cluster_feat[_C_SIZE].item()  # cluster_size = 2^log2(size)
    return feat


# ======================================================================
# Dataset
# ======================================================================

class PriorityDataset(Dataset):
    """Per-step pairwise ranking dataset for learning priority formula.

    Each sample: (positive_features, negative_features, weight)
    where positive = selected candidate, negative = random non-selected.
    """

    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        neg_samples_per_step: int = 4,
        top_quantile: float = 0.25,
        min_reward: float = 0.0,
        gpu_csv: Optional[str] = None,
    ):
        self.pairs: List[Tuple[torch.Tensor, torch.Tensor, float]] = []
        # Load GPU rewards from CSV if available
        self.gpu_rewards: Dict[str, Dict[str, float]] = {}
        if gpu_csv:
            self._load_gpu_csv(gpu_csv)
        self._load(Path(data_dir), split, neg_samples_per_step,
                   top_quantile, min_reward)

    def _load_gpu_csv(self, csv_path: str):
        """Load GPU kernel times from merged CSV for reward computation."""
        import csv
        xla_times = {}  # module → xla_default time
        strat_times = {}  # (module, strategy) → time

        with open(csv_path) as f:
            for row in csv.DictReader(f):
                module = row['module']
                strategy = row['strategy']
                us = float(row['avg_kernel_per_iter_us'])
                if us <= 0:
                    continue
                if strategy == 'xla_default':
                    xla_times[module] = us
                strat_times[(module, strategy)] = us

        # Compute rewards: xla_us / strategy_us
        for (module, strategy), us in strat_times.items():
            xla_us = xla_times.get(module)
            if xla_us and us > 0:
                reward = xla_us / us
                self.gpu_rewards.setdefault(
                    module.replace('/', '__'), {})[strategy] = reward

    def _load(
        self,
        data_dir: Path,
        split: str,
        neg_per_step: int,
        top_quantile: float,
        min_reward: float,
    ):
        pt_files = sorted(data_dir.glob("*.pt"))
        pt_files = [f for f in pt_files if f.stem != 'summary']

        for pt_file in pt_files:
            data = torch.load(str(pt_file), weights_only=False)
            graph = data['graph']
            trajectories = data.get('enriched_trajectories', [])
            if not trajectories:
                continue

            # Name → idx mapping
            name_to_idx = {n: i for i, n in enumerate(graph.node_names)}

            # Filter to top-quantile trajectories
            # Try to get rewards from GPU CSV or from trajectory data
            module_rewards = self.gpu_rewards.get(pt_file.stem, {})
            for t in trajectories:
                if t.get('reward_gpu') is None:
                    strat = t.get('strategy', '')
                    r = module_rewards.get(strat)
                    if r is not None:
                        t['reward_gpu'] = r

            rewards = [t.get('reward_gpu') for t in trajectories
                       if t.get('reward_gpu') is not None]
            if rewards:
                rewards_sorted = sorted(rewards, reverse=True)
                cutoff = rewards_sorted[max(0, int(len(rewards_sorted) * top_quantile) - 1)]
                has_rewards = True
            else:
                cutoff = min_reward
                has_rewards = False

            # Split trajectories 80/20
            n = len(trajectories)
            split_idx = int(0.8 * n)
            if split == 'train':
                trajs = trajectories[:split_idx]
            else:
                trajs = trajectories[split_idx:]

            # Get cluster features
            for traj in trajs:
                reward = traj.get('reward_gpu')

                # When rewards available: filter and weight
                if has_rewards:
                    if reward is None:
                        continue
                    if reward < cutoff:
                        continue
                    advantage = 1.0 - 1.0 / reward if reward > 0 else 0.0
                    weight = min(max(math.exp(3.0 * advantage), 0.1), 5.0)
                else:
                    # No rewards: use all trajectories with equal weight
                    weight = 1.0

                clusters = traj.get('clusters', {})
                steps = traj.get('steps', [])

                for step in steps:
                    cand_names = step['candidate_names']
                    selected = step['selected_name']

                    if selected not in cand_names:
                        continue
                    if len(cand_names) < 2:
                        continue

                    # Extract features for selected
                    pos_feat = self._get_features(
                        selected, graph, name_to_idx, clusters)
                    if pos_feat is None:
                        continue

                    # Sample negatives
                    non_selected = [c for c in cand_names if c != selected]
                    if not non_selected:
                        continue

                    # Random sample up to neg_per_step
                    n_neg = min(neg_per_step, len(non_selected))
                    perm = torch.randperm(len(non_selected))[:n_neg]
                    for ni in perm.tolist():
                        neg_name = non_selected[ni]
                        neg_feat = self._get_features(
                            neg_name, graph, name_to_idx, clusters)
                        if neg_feat is not None:
                            self.pairs.append((pos_feat, neg_feat, weight))

    def _get_features(
        self,
        name: str,
        graph,
        name_to_idx: Dict[str, int],
        clusters: dict,
    ) -> Optional[torch.Tensor]:
        """Get universal features for a candidate."""
        if name.startswith('fusion.'):
            cd = clusters.get(name, {})
            feat = cd.get('features')
            if feat is None or not isinstance(feat, torch.Tensor):
                return None
            return extract_features_cluster(feat)
        else:
            idx = name_to_idx.get(name, -1)
            if idx < 0:
                return None
            return extract_features_orig(
                graph.x[idx],
                graph.opcode_ids[idx].item(),
            )

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        pos, neg, w = self.pairs[idx]
        return pos, neg, torch.tensor(w, dtype=torch.float32)


# ======================================================================
# Models
# ======================================================================

class LinearPriority(nn.Module):
    """Linear scoring: score = w^T * features + bias."""

    def __init__(self, input_dim: int = UNIVERSAL_FEATURE_DIM):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).squeeze(-1)

    def export_weights(self, feat_mean, feat_std) -> dict:
        w = self.linear.weight.data[0].tolist()
        b = self.linear.bias.data[0].item()
        return {
            'model_type': 'linear',
            'feature_names': [
                'log2_output_bytes', 'log2_total_elements', 'output_rank',
                'log2_fan_out', 'log2_fan_in', 'is_elementwise',
                'is_fusion', 'log2_flop_count', 'is_reduce',
                'fused_instr_count',
            ],
            'weights': w,
            'bias': b,
            'cm_weight': 0.5,
            'feature_stats': {
                'mean': feat_mean.tolist(),
                'std': feat_std.tolist(),
            },
        }


class MLPPriority(nn.Module):
    """Tiny MLP scoring: features -> hidden -> score."""

    def __init__(
        self,
        input_dim: int = UNIVERSAL_FEATURE_DIM,
        hidden_dim: int = 16,
    ):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.layer1(x))
        return self.layer2(h).squeeze(-1)

    def export_weights(self, feat_mean, feat_std) -> dict:
        return {
            'model_type': 'mlp',
            'feature_names': [
                'log2_output_bytes', 'log2_total_elements', 'output_rank',
                'log2_fan_out', 'log2_fan_in', 'is_elementwise',
                'is_fusion', 'log2_flop_count', 'is_reduce',
                'fused_instr_count',
            ],
            'hidden_dim': self.layer1.out_features,
            'layer1_weights': self.layer1.weight.data.tolist(),
            'layer1_bias': self.layer1.bias.data.tolist(),
            'layer2_weights': self.layer2.weight.data[0].tolist(),
            'layer2_bias': self.layer2.bias.data[0].item(),
            'cm_weight': 0.5,
            'feature_stats': {
                'mean': feat_mean.tolist(),
                'std': feat_std.tolist(),
            },
        }


# ======================================================================
# Training
# ======================================================================

def compute_feature_stats(dataset: PriorityDataset) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute mean and std of features across all samples."""
    all_feats = []
    for pos, neg, _ in dataset.pairs:
        all_feats.append(pos)
        all_feats.append(neg)
    if not all_feats:
        return torch.zeros(UNIVERSAL_FEATURE_DIM), torch.ones(UNIVERSAL_FEATURE_DIM)
    stacked = torch.stack(all_feats)
    mean = stacked.mean(dim=0)
    std = stacked.std(dim=0).clamp(min=1e-6)
    return mean, std


def train_model(
    model: nn.Module,
    train_ds: PriorityDataset,
    val_ds: PriorityDataset,
    feat_mean: torch.Tensor,
    feat_std: torch.Tensor,
    epochs: int = 100,
    lr: float = 1e-3,
    batch_size: int = 256,
    margin: float = 1.0,
    device: str = 'cpu',
) -> dict:
    """Train with margin ranking loss."""
    device = torch.device(device)
    model = model.to(device)
    feat_mean = feat_mean.to(device)
    feat_std = feat_std.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              drop_last=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=0)

    best_val_acc = 0.0
    best_state = None
    patience = 20
    no_improve = 0

    for epoch in range(epochs):
        # Train
        model.train()
        total_loss = 0.0
        n_batches = 0
        for pos_feat, neg_feat, weight in train_loader:
            pos_feat = (pos_feat.to(device) - feat_mean) / feat_std
            neg_feat = (neg_feat.to(device) - feat_mean) / feat_std
            weight = weight.to(device)

            pos_score = model(pos_feat)
            neg_score = model(neg_feat)

            # Margin ranking loss: pos should be > neg by margin
            target = torch.ones_like(pos_score)
            loss = F.margin_ranking_loss(
                pos_score, neg_score, target, margin=margin, reduction='none')
            loss = (loss * weight).mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = total_loss / max(n_batches, 1)

        # Validate
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for pos_feat, neg_feat, _ in val_loader:
                pos_feat = (pos_feat.to(device) - feat_mean) / feat_std
                neg_feat = (neg_feat.to(device) - feat_mean) / feat_std
                pos_score = model(pos_feat)
                neg_score = model(neg_feat)
                correct += (pos_score > neg_score).sum().item()
                total += pos_score.size(0)

        val_acc = correct / max(total, 1)

        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"  Epoch {epoch:3d}: loss={avg_loss:.4f}, "
                  f"val_pairwise_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  Early stopping at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return {'best_val_acc': best_val_acc, 'final_epoch': epoch}


# ======================================================================
# Evaluation
# ======================================================================

def evaluate_ranking(
    model: nn.Module,
    data_dir: str,
    feat_mean: torch.Tensor,
    feat_std: torch.Tensor,
    device: str = 'cpu',
):
    """Evaluate per-step ranking accuracy on val split.

    For each step, score all candidates and check if selected is ranked #1.
    """
    device_t = torch.device(device)
    model = model.to(device_t).eval()
    feat_mean = feat_mean.to(device_t)
    feat_std = feat_std.to(device_t)

    data_dir_p = Path(data_dir)
    pt_files = sorted(data_dir_p.glob("*.pt"))
    pt_files = [f for f in pt_files if f.stem != 'summary']

    total_steps = 0
    top1_correct = 0
    top3_correct = 0
    mrr_sum = 0.0

    orig_steps = 0
    orig_top1 = 0
    fusion_steps = 0
    fusion_top1 = 0

    per_module = {}

    for pt_file in pt_files:
        module_key = pt_file.stem
        data = torch.load(str(pt_file), weights_only=False)
        graph = data['graph']
        trajectories = data.get('enriched_trajectories', [])
        if not trajectories:
            continue

        name_to_idx = {n: i for i, n in enumerate(graph.node_names)}

        # Val split (last 20% of trajectories)
        n = len(trajectories)
        trajs = trajectories[int(0.8 * n):]

        m_total = 0
        m_top1 = 0

        for traj in trajs:
            clusters = traj.get('clusters', {})
            steps = traj.get('steps', [])

            for step in steps:
                cand_names = step['candidate_names']
                selected = step['selected_name']
                if selected not in cand_names or len(cand_names) < 2:
                    continue

                # Extract features for all candidates
                feats = []
                valid_names = []
                for name in cand_names:
                    if name.startswith('fusion.'):
                        cd = clusters.get(name, {})
                        f = cd.get('features')
                        if f is None or not isinstance(f, torch.Tensor):
                            continue
                        feats.append(extract_features_cluster(f))
                    else:
                        idx = name_to_idx.get(name, -1)
                        if idx < 0:
                            continue
                        feats.append(extract_features_orig(
                            graph.x[idx], graph.opcode_ids[idx].item()))
                    valid_names.append(name)

                if selected not in valid_names or len(valid_names) < 2:
                    continue

                sel_idx = valid_names.index(selected)
                feat_tensor = torch.stack(feats).to(device_t)
                feat_tensor = (feat_tensor - feat_mean) / feat_std

                with torch.no_grad():
                    scores = model(feat_tensor)

                # Ranking
                sorted_indices = scores.argsort(descending=True).tolist()
                rank = sorted_indices.index(sel_idx) + 1

                total_steps += 1
                m_total += 1
                if rank == 1:
                    top1_correct += 1
                    m_top1 += 1
                if rank <= 3:
                    top3_correct += 1
                mrr_sum += 1.0 / rank

                # Orig vs fusion breakdown
                is_fusion = selected.startswith('fusion.')
                if is_fusion:
                    fusion_steps += 1
                    if rank == 1:
                        fusion_top1 += 1
                else:
                    orig_steps += 1
                    if rank == 1:
                        orig_top1 += 1

        if m_total > 0:
            per_module[module_key] = {
                'steps': m_total,
                'top1_acc': m_top1 / m_total,
            }

    print(f"\n=== Ranking Evaluation (val split) ===")
    print(f"Total steps: {total_steps}")
    if total_steps > 0:
        print(f"Top-1 accuracy: {top1_correct/total_steps:.4f}")
        print(f"Top-3 accuracy: {top3_correct/total_steps:.4f}")
        print(f"MRR:            {mrr_sum/total_steps:.4f}")
        print(f"\nOriginal nodes: {orig_top1}/{orig_steps} = "
              f"{orig_top1/max(orig_steps,1):.4f}")
        print(f"Fusion.N:       {fusion_top1}/{fusion_steps} = "
              f"{fusion_top1/max(fusion_steps,1):.4f}")

    # Feature importance (for linear model)
    if hasattr(model, 'linear'):
        w = model.linear.weight.data[0]
        names = [
            'log2_bytes', 'log2_elems', 'rank', 'log2_fanout',
            'log2_fanin', 'is_ewise', 'is_fusion', 'log2_flops',
            'is_reduce', 'fused_count',
        ]
        print(f"\n=== Feature Importance (linear weights) ===")
        # Sort by absolute weight
        sorted_w = sorted(zip(names, w.tolist()), key=lambda x: abs(x[1]), reverse=True)
        for name, weight in sorted_w:
            print(f"  {name:15s}: {weight:+.4f}")

    return {
        'total_steps': total_steps,
        'top1_acc': top1_correct / max(total_steps, 1),
        'top3_acc': top3_correct / max(total_steps, 1),
        'mrr': mrr_sum / max(total_steps, 1),
        'orig_acc': orig_top1 / max(orig_steps, 1),
        'fusion_acc': fusion_top1 / max(fusion_steps, 1),
        'per_module': per_module,
    }


# ======================================================================
# Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Learn a parametric priority formula for XLA fusion")
    sub = parser.add_subparsers(dest="command")

    # Train
    p_train = sub.add_parser("train", help="Train priority formula")
    p_train.add_argument("--data-dir", required=True,
                         help="Directory with enriched .pt files")
    p_train.add_argument("--model-type", default="linear",
                         choices=["linear", "mlp"])
    p_train.add_argument("--hidden-dim", type=int, default=16,
                         help="Hidden dim for MLP model")
    p_train.add_argument("--epochs", type=int, default=100)
    p_train.add_argument("--lr", type=float, default=1e-3)
    p_train.add_argument("--batch-size", type=int, default=256)
    p_train.add_argument("--margin", type=float, default=1.0)
    p_train.add_argument("--neg-samples", type=int, default=4,
                         help="Negative samples per step")
    p_train.add_argument("--top-quantile", type=float, default=0.25,
                         help="Keep top fraction of trajectories by reward")
    p_train.add_argument("--gpu-csv", default=None,
                         help="GPU profiles CSV for reward computation")
    p_train.add_argument("--device", default="cpu")
    p_train.add_argument("--output", default="output/learned_weights.json")

    # Eval
    p_eval = sub.add_parser("eval", help="Evaluate learned weights")
    p_eval.add_argument("--data-dir", required=True)
    p_eval.add_argument("--weights", required=True,
                        help="Path to learned_weights.json")
    p_eval.add_argument("--device", default="cpu")

    args = parser.parse_args()

    if args.command == "train":
        print(f"Loading training data from {args.data_dir}...")
        train_ds = PriorityDataset(
            args.data_dir, split='train',
            neg_samples_per_step=args.neg_samples,
            top_quantile=args.top_quantile,
            gpu_csv=args.gpu_csv,
        )
        val_ds = PriorityDataset(
            args.data_dir, split='val',
            neg_samples_per_step=args.neg_samples,
            top_quantile=1.0,  # val uses all trajectories
            gpu_csv=args.gpu_csv,
        )
        print(f"Train pairs: {len(train_ds)}")
        print(f"Val pairs:   {len(val_ds)}")

        if len(train_ds) == 0:
            print("No training data. Check data directory.")
            return

        # Compute feature normalization stats
        feat_mean, feat_std = compute_feature_stats(train_ds)
        print(f"Feature mean: {feat_mean.tolist()}")
        print(f"Feature std:  {feat_std.tolist()}")

        # Build model
        if args.model_type == "linear":
            model = LinearPriority()
        else:
            model = MLPPriority(hidden_dim=args.hidden_dim)

        n_params = sum(p.numel() for p in model.parameters())
        print(f"\nModel: {args.model_type}, {n_params} params")

        # Train
        print(f"Training for up to {args.epochs} epochs...")
        result = train_model(
            model, train_ds, val_ds,
            feat_mean, feat_std,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            margin=args.margin,
            device=args.device,
        )
        print(f"\nBest val pairwise accuracy: {result['best_val_acc']:.4f}")

        # Evaluate full ranking
        metrics = evaluate_ranking(
            model, args.data_dir, feat_mean, feat_std, device=args.device)

        # Export weights
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        weights = model.export_weights(feat_mean.cpu(), feat_std.cpu())
        weights['training'] = {
            'epochs': result['final_epoch'],
            'val_pairwise_acc': result['best_val_acc'],
            'top1_acc': metrics['top1_acc'],
            'mrr': metrics['mrr'],
            'train_pairs': len(train_ds),
            'val_pairs': len(val_ds),
        }
        output_path.write_text(json.dumps(weights, indent=2))
        print(f"\nWeights exported to {output_path}")

    elif args.command == "eval":
        # Load weights
        weights = json.loads(Path(args.weights).read_text())
        feat_mean = torch.tensor(weights['feature_stats']['mean'])
        feat_std = torch.tensor(weights['feature_stats']['std'])

        if weights['model_type'] == 'linear':
            model = LinearPriority()
            model.linear.weight.data = torch.tensor(
                [weights['weights']], dtype=torch.float32)
            model.linear.bias.data = torch.tensor(
                [weights['bias']], dtype=torch.float32)
        else:
            model = MLPPriority(hidden_dim=weights['hidden_dim'])
            model.layer1.weight.data = torch.tensor(
                weights['layer1_weights'], dtype=torch.float32)
            model.layer1.bias.data = torch.tensor(
                weights['layer1_bias'], dtype=torch.float32)
            model.layer2.weight.data = torch.tensor(
                [weights['layer2_weights']], dtype=torch.float32)
            model.layer2.bias.data = torch.tensor(
                [weights['layer2_bias']], dtype=torch.float32)

        evaluate_ranking(model, args.data_dir, feat_mean, feat_std,
                         device=args.device)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
