"""Training loop for the dynamic priority scorer.

Trains a PriorityScorer on enriched trajectory data from cluster_features.py.
Supports three training modes: imitation_all, advantage_weighted, top_quantile.

Usage:
    # Basic training
    python scorer_train.py \\
        --data-dir output/dynamic_features \\
        --epochs 100 --device cuda

    # Advantage-weighted training
    python scorer_train.py \\
        --data-dir output/dynamic_features \\
        --training-mode advantage_weighted \\
        --epochs 100 --device cuda

    # Leave-one-architecture-out
    python scorer_train.py \\
        --data-dir output/dynamic_features \\
        --holdout-arch Griffin-2B \\
        --epochs 100 --device cuda
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scorer_model import PriorityScorer, build_scorer
from scorer_dataset import (
    ScorerDataset, scorer_collate_fn, build_dataloaders,
    MAX_CANDIDATES,
)


# ======================================================================
# Evaluation
# ======================================================================

@torch.no_grad()
def evaluate(
    model: PriorityScorer,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate model on a dataloader.

    Returns dict with:
      - loss: mean cross-entropy
      - top1_acc: fraction where selected = highest scored
      - top3_acc: fraction where selected in top 3
      - mrr: mean reciprocal rank
      - top1_orig: accuracy on original-node selections
      - top1_fusion: accuracy on fusion.N selections
      - n_samples: total samples evaluated
    """
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_top3 = 0
    total_rr = 0.0
    n_samples = 0

    # Per-type accumulators
    orig_correct = 0
    orig_total = 0
    fusion_correct = 0
    fusion_total = 0

    for batch in loader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

        scores = model(
            orig_features=batch['orig_features'],
            orig_opcode_ids=batch['orig_opcode_ids'],
            orig_mask=batch['orig_mask'],
            cluster_features=batch['cluster_features'],
            cluster_mask=batch['cluster_mask'],
            context=batch['context'],
        )

        target = batch['target_idx']
        loss = model.compute_loss(scores, target)
        total_loss += loss.item() * target.size(0)

        pred = model.predict(scores)
        correct = (pred == target)
        total_correct += correct.sum().item()

        # Top-3
        _, top3 = scores.topk(min(3, scores.size(1)), dim=1)
        in_top3 = (top3 == target.unsqueeze(1)).any(dim=1)
        total_top3 += in_top3.sum().item()

        # MRR
        sorted_indices = scores.argsort(dim=1, descending=True)
        for i in range(target.size(0)):
            rank = (sorted_indices[i] == target[i]).nonzero(as_tuple=True)[0]
            if len(rank) > 0:
                total_rr += 1.0 / (rank[0].item() + 1)

        # Per-type accuracy
        is_fus = batch['is_fusion_selected']
        orig_mask_b = ~is_fus
        fusion_mask_b = is_fus

        orig_correct += (correct & orig_mask_b).sum().item()
        orig_total += orig_mask_b.sum().item()
        fusion_correct += (correct & fusion_mask_b).sum().item()
        fusion_total += fusion_mask_b.sum().item()

        n_samples += target.size(0)

    if n_samples == 0:
        return {'loss': 0, 'top1_acc': 0, 'top3_acc': 0, 'mrr': 0,
                'top1_orig': 0, 'top1_fusion': 0, 'n_samples': 0}

    return {
        'loss': total_loss / n_samples,
        'top1_acc': total_correct / n_samples,
        'top3_acc': total_top3 / n_samples,
        'mrr': total_rr / n_samples,
        'top1_orig': orig_correct / max(orig_total, 1),
        'top1_fusion': fusion_correct / max(fusion_total, 1),
        'n_orig': orig_total,
        'n_fusion': fusion_total,
        'n_samples': n_samples,
    }


# ======================================================================
# Training loop
# ======================================================================

def train(
    model: PriorityScorer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 100,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    patience: int = 20,
    checkpoint_dir: str = 'output/scorer_checkpoints',
    use_advantage_weights: bool = False,
) -> Dict[str, list]:
    """Train the scorer with early stopping.

    Returns history dict with per-epoch metrics.
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': [],
        'val_top3': [], 'val_mrr': [],
        'val_orig_acc': [], 'val_fusion_acc': [],
        'lr': [],
    }

    best_val_loss = float('inf')
    best_epoch = 0
    no_improve = 0

    print(f"\nTraining: {model.param_count():,} params, "
          f"{len(train_loader.dataset)} train, {len(val_loader.dataset)} val")
    print(f"{'='*80}")

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        # --- Train ---
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_samples = 0

        for batch_idx, batch in enumerate(train_loader):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            scores = model(
                orig_features=batch['orig_features'],
                orig_opcode_ids=batch['orig_opcode_ids'],
                orig_mask=batch['orig_mask'],
                cluster_features=batch['cluster_features'],
                cluster_mask=batch['cluster_mask'],
                context=batch['context'],
            )

            target = batch['target_idx']
            weight = batch['weight'] if use_advantage_weights else None
            loss = model.compute_loss(scores, target, sample_weight=weight)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item() * target.size(0)
            epoch_correct += (model.predict(scores) == target).sum().item()
            epoch_samples += target.size(0)

        scheduler.step()

        train_loss = epoch_loss / max(epoch_samples, 1)
        train_acc = epoch_correct / max(epoch_samples, 1)

        # --- Validate ---
        val_metrics = evaluate(model, val_loader, device)

        # Record
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_metrics['loss'])
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_metrics['top1_acc'])
        history['val_top3'].append(val_metrics['top3_acc'])
        history['val_mrr'].append(val_metrics['mrr'])
        history['val_orig_acc'].append(val_metrics['top1_orig'])
        history['val_fusion_acc'].append(val_metrics['top1_fusion'])
        history['lr'].append(optimizer.param_groups[0]['lr'])

        dt = time.time() - t0

        # Print
        print(f"  {epoch:3d}  "
              f"train={train_loss:.4f}/{train_acc:.4f}  "
              f"val={val_metrics['loss']:.4f}/{val_metrics['top1_acc']:.4f}  "
              f"top3={val_metrics['top3_acc']:.4f}  "
              f"mrr={val_metrics['mrr']:.4f}  "
              f"orig={val_metrics['top1_orig']:.4f}  "
              f"fus={val_metrics['top1_fusion']:.4f}  "
              f"lr={optimizer.param_groups[0]['lr']:.1e}  "
              f"{dt:.1f}s")

        # Early stopping
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            best_epoch = epoch
            no_improve = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_metrics,
                'history': history,
            }, str(checkpoint_dir / 'best.pt'))
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"\n  Early stopping at epoch {epoch} "
                      f"(best: epoch {best_epoch}, "
                      f"val_loss={best_val_loss:.4f})")
                break

    # Save final
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'history': history,
    }, str(checkpoint_dir / 'final.pt'))

    # Save history as JSON
    history_json = {k: [float(v) for v in vs] for k, vs in history.items()}
    (checkpoint_dir / 'history.json').write_text(
        json.dumps(history_json, indent=2))

    print(f"\nBest: epoch {best_epoch}, val_loss={best_val_loss:.4f}")
    print(f"Checkpoints: {checkpoint_dir}")

    return history


# ======================================================================
# Baselines
# ======================================================================

@torch.no_grad()
def evaluate_baselines(
    val_loader: DataLoader,
    device: torch.device,
) -> Dict[str, Dict[str, float]]:
    """Evaluate simple baselines for comparison.

    Baselines:
      - random: random candidate selection
      - first: always pick first candidate
    """
    results = {}

    # Random baseline
    total = 0
    correct = 0
    for batch in val_loader:
        B = batch['target_idx'].size(0)
        n_cands = (batch['orig_mask'].sum(dim=1) +
                   batch['cluster_mask'].sum(dim=1))
        for i in range(B):
            if n_cands[i] > 0:
                correct += 1.0 / n_cands[i].item()
            total += 1
    results['random'] = {
        'top1_acc': correct / max(total, 1),
        'n_samples': total,
    }

    # First-candidate baseline
    total = 0
    correct = 0
    for batch in val_loader:
        target = batch['target_idx']
        correct += (target == 0).sum().item()
        total += target.size(0)
    results['first'] = {
        'top1_acc': correct / max(total, 1),
        'n_samples': total,
    }

    return results


# ======================================================================
# Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Train dynamic priority scorer")
    parser.add_argument("--data-dir", required=True,
                        help="Directory with enriched .pt files")
    parser.add_argument("--checkpoint-dir",
                        default="output/scorer_checkpoints")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--max-cands", type=int, default=MAX_CANDIDATES)
    parser.add_argument("--training-mode", default='imitation_all',
                        choices=['imitation_all', 'advantage_weighted',
                                 'top_quantile'])
    parser.add_argument("--holdout-arch", default=None,
                        help="Architecture to hold out for LOO eval")
    parser.add_argument("--device", default='cuda' if torch.cuda.is_available()
                        else 'cpu')
    parser.add_argument("--dropout", type=float, default=0.1)
    args = parser.parse_args()

    device = torch.device(args.device)

    # Build data
    print(f"Loading data from {args.data_dir}...")
    train_loader, val_loader = build_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        holdout_arch=args.holdout_arch,
        training_mode=args.training_mode,
        max_candidates=args.max_cands,
    )

    if len(train_loader.dataset) == 0:
        print("ERROR: No training samples. Check data directory.")
        return

    print(f"Train: {len(train_loader.dataset)} samples, "
          f"{len(train_loader)} batches")
    print(f"Val: {len(val_loader.dataset)} samples, "
          f"{len(val_loader)} batches")

    # Baselines
    print("\nBaselines:")
    baselines = evaluate_baselines(val_loader, device)
    for name, metrics in baselines.items():
        print(f"  {name:10s}: top1={metrics['top1_acc']:.4f}")

    # Build model
    model = build_scorer(dropout=args.dropout)
    print(f"\nModel: {model.param_count():,} params")

    # Train
    use_adv = args.training_mode == 'advantage_weighted'
    history = train(
        model, train_loader, val_loader, device,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        checkpoint_dir=args.checkpoint_dir,
        use_advantage_weights=use_adv,
    )

    # Final evaluation with best checkpoint
    ckpt = torch.load(
        str(Path(args.checkpoint_dir) / 'best.pt'),
        weights_only=False, map_location=device,
    )
    model.load_state_dict(ckpt['model_state_dict'])
    final_metrics = evaluate(model, val_loader, device)

    print(f"\nFinal eval (best checkpoint, epoch {ckpt['epoch']}):")
    print(f"  Loss:       {final_metrics['loss']:.4f}")
    print(f"  Top-1:      {final_metrics['top1_acc']:.4f}")
    print(f"  Top-3:      {final_metrics['top3_acc']:.4f}")
    print(f"  MRR:        {final_metrics['mrr']:.4f}")
    print(f"  Orig acc:   {final_metrics['top1_orig']:.4f} "
          f"({final_metrics['n_orig']} samples)")
    print(f"  Fusion acc: {final_metrics['top1_fusion']:.4f} "
          f"({final_metrics['n_fusion']} samples)")

    # Save summary
    summary = {
        'args': vars(args),
        'baselines': baselines,
        'best_epoch': ckpt['epoch'],
        'final_metrics': {k: float(v) if isinstance(v, (int, float)) else v
                          for k, v in final_metrics.items()},
        'model_params': model.param_count(),
    }
    (Path(args.checkpoint_dir) / 'summary.json').write_text(
        json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
