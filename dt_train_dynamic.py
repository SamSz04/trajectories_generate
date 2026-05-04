"""Training script for Dynamic Fusion DT.

Supports multiple loss modes:
  - ce: standard cross-entropy
  - advantage_weighted: CE weighted by trajectory advantage
  - top_quantile: only train on top-Q% trajectories per module
  - awbc_filtered: advantage-weighted + bottom-quantile filtering

Usage:
    PYTHONPATH=~/go_fusion python3 dt_train_dynamic.py \
        --enriched-dir output/dynamic_features \
        --gpu-csv output/gpu_profiles_merged.csv \
        --epochs 200 --patience 30 --device cuda
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dt_model_dynamic import DynamicFusionDT
from dt_dataset_dynamic import (
    create_dynamic_dataloaders,
    DynamicFusionDTDataset,
    dynamic_collate_fn,
)


# ======================================================================
# Loss functions
# ======================================================================

def compute_loss(
    logits: torch.Tensor,
    actions: torch.Tensor,
    attn_mask: torch.Tensor,
    candidate_masks: torch.Tensor,
    loss_mode: str = "ce",
    gpu_rewards: Optional[List[float]] = None,
    advantage_beta: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute training loss with optional trajectory weighting.

    Args:
        logits: [B, K, max_Vt] action logits
        actions: [B, K] target action indices
        attn_mask: [B, K] valid step mask (1 = valid)
        candidate_masks: [B, K, max_Vt] candidate masks
        loss_mode: "ce", "advantage_weighted", "top_quantile", "awbc_filtered"
        gpu_rewards: [B] GPU reward per trajectory (used for weighting)
        advantage_beta: scaling factor for advantage weights

    Returns:
        (loss, metrics_dict)
    """
    B, K, V = logits.shape
    device = logits.device

    # Flatten for cross-entropy
    valid_mask = (attn_mask == 1) & (actions >= 0)  # [B, K]
    valid_mask = valid_mask & (actions < V)

    if not valid_mask.any():
        return torch.tensor(0.0, device=device, requires_grad=True), {"loss": 0.0}

    # Gather valid logits and targets
    flat_logits = logits[valid_mask]     # [N_valid, V]
    flat_targets = actions[valid_mask]   # [N_valid]

    # Base CE loss per sample
    ce_per_sample = F.cross_entropy(flat_logits, flat_targets, reduction="none")

    if loss_mode == "ce":
        loss = ce_per_sample.mean()
    elif loss_mode == "advantage_weighted" and gpu_rewards is not None:
        # Weight each trajectory's steps by exp(beta * advantage)
        # advantage = gpu_reward - 1.0 (relative to XLA default)
        weights = torch.ones(B, K, device=device)
        for b in range(B):
            if gpu_rewards[b] is not None:
                adv = gpu_rewards[b] - 1.0
                w = min(max(math.exp(advantage_beta * adv), 0.1), 5.0)
                weights[b] = w
        flat_weights = weights[valid_mask]
        loss = (ce_per_sample * flat_weights).sum() / flat_weights.sum()
    else:
        loss = ce_per_sample.mean()

    # Metrics
    preds = flat_logits.argmax(dim=-1)
    correct = (preds == flat_targets).float()
    top1_acc = correct.mean().item()

    # Top-3 accuracy
    _, top3_idx = flat_logits.topk(min(3, V), dim=-1)
    top3_hits = (top3_idx == flat_targets.unsqueeze(-1)).any(dim=-1).float()
    top3_acc = top3_hits.mean().item()

    # MRR
    sorted_idx = flat_logits.argsort(dim=-1, descending=True)
    ranks = (sorted_idx == flat_targets.unsqueeze(-1)).nonzero(as_tuple=True)[1] + 1
    mrr = (1.0 / ranks.float()).mean().item()

    metrics = {
        "loss": loss.item(),
        "top1_acc": top1_acc,
        "top3_acc": top3_acc,
        "mrr": mrr,
        "n_valid": valid_mask.sum().item(),
    }

    return loss, metrics


# ======================================================================
# Training loop
# ======================================================================

def train_epoch(
    model: DynamicFusionDT,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    loss_mode: str = "ce",
    advantage_beta: float = 1.0,
    grad_clip: float = 1.0,
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    total_top3 = 0.0
    total_mrr = 0.0
    total_steps = 0
    n_batches = 0

    for batch in loader:
        pyg_batch = batch["pyg_batch"].to(device)
        graph_sizes = batch["graph_sizes"].to(device)
        actions = batch["actions"].to(device)
        rtgs = batch["rtgs"].to(device)
        timesteps = batch["timesteps"].to(device)
        attn_mask = batch["attn_mask"].to(device)
        candidate_masks = batch["candidate_masks"].to(device)
        gpu_rewards = batch.get("gpu_rewards")

        logits = model(
            pyg_batch, graph_sizes, actions, rtgs, timesteps,
            attn_mask, candidate_masks,
        )

        loss, metrics = compute_loss(
            logits, actions, attn_mask, candidate_masks,
            loss_mode=loss_mode,
            gpu_rewards=gpu_rewards,
            advantage_beta=advantage_beta,
        )

        optimizer.zero_grad()
        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += metrics["loss"] * metrics["n_valid"]
        total_acc += metrics["top1_acc"] * metrics["n_valid"]
        total_top3 += metrics["top3_acc"] * metrics["n_valid"]
        total_mrr += metrics["mrr"] * metrics["n_valid"]
        total_steps += metrics["n_valid"]
        n_batches += 1

    n = max(total_steps, 1)
    return {
        "loss": total_loss / n,
        "top1_acc": total_acc / n,
        "top3_acc": total_top3 / n,
        "mrr": total_mrr / n,
        "n_batches": n_batches,
        "n_steps": total_steps,
    }


@torch.no_grad()
def eval_epoch(
    model: DynamicFusionDT,
    loader: DataLoader,
    device: torch.device,
    loss_mode: str = "ce",
) -> Dict[str, float]:
    """Evaluate for one epoch."""
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    total_top3 = 0.0
    total_mrr = 0.0
    total_steps = 0

    # Per-action-type accuracy
    n_orig_correct = 0
    n_orig_total = 0
    n_fusion_correct = 0
    n_fusion_total = 0

    for batch in loader:
        pyg_batch = batch["pyg_batch"].to(device)
        graph_sizes = batch["graph_sizes"].to(device)
        actions = batch["actions"].to(device)
        rtgs = batch["rtgs"].to(device)
        timesteps = batch["timesteps"].to(device)
        attn_mask = batch["attn_mask"].to(device)
        candidate_masks = batch["candidate_masks"].to(device)

        logits = model(
            pyg_batch, graph_sizes, actions, rtgs, timesteps,
            attn_mask, candidate_masks,
        )

        loss, metrics = compute_loss(
            logits, actions, attn_mask, candidate_masks,
            loss_mode="ce",
        )

        total_loss += metrics["loss"] * metrics["n_valid"]
        total_acc += metrics["top1_acc"] * metrics["n_valid"]
        total_top3 += metrics["top3_acc"] * metrics["n_valid"]
        total_mrr += metrics["mrr"] * metrics["n_valid"]
        total_steps += metrics["n_valid"]

        # Breakdown by action type (original vs fusion.N)
        B, K = actions.shape
        valid_mask = (attn_mask == 1) & (actions >= 0)
        metadatas = batch["metadatas"]  # [B][K]

        for b in range(B):
            for k in range(K):
                if not valid_mask[b, k]:
                    continue
                meta = metadatas[b][k]
                a_idx = actions[b, k].item()
                if a_idx < 0 or a_idx >= len(meta.idx_to_name):
                    continue

                name = meta.idx_to_name[a_idx]
                pred = logits[b, k].argmax().item()
                is_correct = (pred == a_idx)

                if name.startswith("fusion."):
                    n_fusion_total += 1
                    n_fusion_correct += int(is_correct)
                else:
                    n_orig_total += 1
                    n_orig_correct += int(is_correct)

    n = max(total_steps, 1)
    result = {
        "loss": total_loss / n,
        "top1_acc": total_acc / n,
        "top3_acc": total_top3 / n,
        "mrr": total_mrr / n,
        "n_steps": total_steps,
    }

    if n_orig_total > 0:
        result["orig_acc"] = n_orig_correct / n_orig_total
        result["n_orig"] = n_orig_total
    if n_fusion_total > 0:
        result["fusion_acc"] = n_fusion_correct / n_fusion_total
        result["n_fusion"] = n_fusion_total

    return result


# ======================================================================
# Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser(description="Train Dynamic Fusion DT")
    # Data
    parser.add_argument("--enriched-dir", default="output/dynamic_features")
    parser.add_argument("--gpu-csv", default="output/gpu_profiles_merged.csv")
    parser.add_argument("--reward-mode", default="gpu",
                        choices=["gpu", "gpu_progressive", "cost_model"])
    parser.add_argument("--context-len", type=int, default=20)
    parser.add_argument("--skip-computation", action="store_true", default=True)
    parser.add_argument("--max-trajs-per-module", type=int, default=0)
    parser.add_argument("--precomputed-dir", default=None,
                        help="Directory with pre-computed contracted graphs")

    # Model
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--d-ff", type=int, default=128)
    parser.add_argument("--gnn-hidden-dim", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=0.1)

    # Training
    parser.add_argument("--loss-mode", default="ce",
                        choices=["ce", "advantage_weighted", "top_quantile", "awbc_filtered"])
    parser.add_argument("--advantage-beta", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=30)

    # Holdout
    parser.add_argument("--holdout-arch", default=None)
    parser.add_argument("--val-split", type=float, default=0.15)

    # Output
    parser.add_argument("--checkpoint-dir", default="output/dt_dynamic_checkpoints")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    print(f"=== Dynamic Fusion DT Training ===")
    print(f"Device: {device}")
    print(f"Loss mode: {args.loss_mode}")
    print(f"Reward mode: {args.reward_mode}")

    # Load data
    train_loader, val_loader, info = create_dynamic_dataloaders(
        enriched_dir=args.enriched_dir,
        gpu_csv_path=args.gpu_csv,
        context_len=args.context_len,
        reward_mode=args.reward_mode,
        batch_size=args.batch_size,
        val_split=args.val_split,
        holdout_arch=args.holdout_arch,
        skip_computation=args.skip_computation,
        max_trajectories_per_module=args.max_trajs_per_module,
        precomputed_dir=args.precomputed_dir,
    )

    # Build model
    model = DynamicFusionDT(
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        d_ff=args.d_ff,
        context_len=args.context_len,
        gnn_hidden_dim=args.gnn_hidden_dim,
        dropout=args.dropout,
    ).to(device)

    print(f"\nModel: {model.count_parameters():,} parameters")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6,
    )

    best_val_acc = 0.0
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_metrics = train_epoch(
            model, train_loader, optimizer, device,
            loss_mode=args.loss_mode,
            advantage_beta=args.advantage_beta,
            grad_clip=args.grad_clip,
        )
        val_metrics = eval_epoch(model, val_loader, device)

        scheduler.step()
        elapsed = time.time() - t0

        # Print progress
        print(f"Epoch {epoch:3d} [{elapsed:.0f}s] | "
              f"Train loss={train_metrics['loss']:.4f} acc={train_metrics['top1_acc']:.3f} | "
              f"Val loss={val_metrics['loss']:.4f} "
              f"acc={val_metrics['top1_acc']:.3f} "
              f"top3={val_metrics.get('top3_acc', 0):.3f} "
              f"mrr={val_metrics.get('mrr', 0):.3f}", end="")

        if "orig_acc" in val_metrics and "fusion_acc" in val_metrics:
            print(f" | orig={val_metrics['orig_acc']:.3f}"
                  f"({val_metrics.get('n_orig', 0)}) "
                  f"fus={val_metrics['fusion_acc']:.3f}"
                  f"({val_metrics.get('n_fusion', 0)})", end="")
        print()

        # Checkpointing
        val_acc = val_metrics["top1_acc"]
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            ckpt = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_metrics": val_metrics,
                "train_metrics": train_metrics,
                "args": vars(args),
                "info": info,
            }
            torch.save(ckpt, os.path.join(args.checkpoint_dir, "best.pt"))
            print(f"  -> New best: {best_val_acc:.4f}")
        else:
            patience_counter += 1

        if patience_counter >= args.patience:
            print(f"\nEarly stopping at epoch {epoch} (patience={args.patience})")
            break

    print(f"\nBest val accuracy: {best_val_acc:.4f}")
    print(f"Checkpoint: {args.checkpoint_dir}/best.pt")


if __name__ == "__main__":
    main()
