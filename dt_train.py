"""Training loop for Decision Transformer on fusion trajectories.

Usage:
    python dt_train.py [--traj-path PATH] [--epochs N] [--lr LR]

Logs to TensorBoard in output/dt_runs/ and saves checkpoints to output/dt_checkpoints/.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "references", "GO_fusion"))

from dt_dataset import create_dataloaders
from dt_model import DecisionTransformer


def train_epoch(model, loader, optimizer, graph, device):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for batch in loader:
        rtg = batch["returns_to_go"].to(device)
        masks = batch["fused_masks"].to(device)
        actions = batch["actions"].to(device)
        timesteps = batch["timesteps"].to(device)
        attn_mask = batch["attn_mask"].to(device)

        # Re-encode graph each batch (fresh computation graph for backward)
        node_embeds = model.encode_graph(graph)
        logits = model(rtg, masks, actions, timesteps, attn_mask, node_embeds)

        # Cross-entropy loss on valid (non-padded) positions
        valid = attn_mask.bool()  # [B, K]
        logits_flat = logits[valid]  # [V, 57]
        targets_flat = actions[valid]  # [V]

        loss = F.cross_entropy(logits_flat, targets_flat)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * targets_flat.shape[0]
        total_correct += (logits_flat.argmax(dim=-1) == targets_flat).sum().item()
        total_count += targets_flat.shape[0]

    return total_loss / total_count, total_correct / total_count


@torch.no_grad()
def eval_epoch(model, loader, graph, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    node_embeds = model.encode_graph(graph)

    for batch in loader:
        rtg = batch["returns_to_go"].to(device)
        masks = batch["fused_masks"].to(device)
        actions = batch["actions"].to(device)
        timesteps = batch["timesteps"].to(device)
        attn_mask = batch["attn_mask"].to(device)

        logits = model(rtg, masks, actions, timesteps, attn_mask, node_embeds)

        valid = attn_mask.bool()
        logits_flat = logits[valid]
        targets_flat = actions[valid]

        loss = F.cross_entropy(logits_flat, targets_flat)

        total_loss += loss.item() * targets_flat.shape[0]
        total_correct += (logits_flat.argmax(dim=-1) == targets_flat).sum().item()
        total_count += targets_flat.shape[0]

    return total_loss / total_count, total_correct / total_count


def main():
    parser = argparse.ArgumentParser(description="Train Decision Transformer")
    parser.add_argument("--traj-path", default="output/trajectories_20260420_a100.pt")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--context-len", type=int, default=20)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--checkpoint-dir", default="output/dt_checkpoints")
    parser.add_argument("--log-dir", default="output/dt_runs")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # TensorBoard (optional)
    writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter
        os.makedirs(args.log_dir, exist_ok=True)
        writer = SummaryWriter(args.log_dir)
    except ImportError:
        print("TensorBoard not available, logging to stdout only")

    # Data
    train_loader, val_loader, graph, info = create_dataloaders(
        args.traj_path,
        context_len=args.context_len,
        batch_size=args.batch_size,
    )
    print(f"Dataset: {info['num_train_trajs']} train / {info['num_val_trajs']} val trajectories")
    print(f"         {info['num_train_windows']} train / {info['num_val_windows']} val windows")
    print(f"         {info['num_nodes']} nodes, {info['num_actions']} actions")

    # Model
    model = DecisionTransformer(
        num_nodes=info["num_nodes"],
        context_len=info["context_len"],
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {total_params:,} parameters")

    # Encode graph (fixed for all epochs)
    graph = graph.to(device)

    # Optimizer + scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs,
    )

    # Training loop
    best_val_loss = float("inf")
    patience_counter = 0

    print(f"\nTraining for {args.epochs} epochs (patience={args.patience})...")
    print(f"{'Epoch':>5} {'Train Loss':>11} {'Train Acc':>10} {'Val Loss':>10} "
          f"{'Val Acc':>9} {'LR':>10} {'Time':>6}")
    print("-" * 70)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # Re-encode graph each epoch (GNN weights update)
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, graph, device,
        )
        val_loss, val_acc = eval_epoch(model, val_loader, graph, device)
        scheduler.step()

        elapsed = time.time() - t0
        lr = scheduler.get_last_lr()[0]

        # Log
        if writer:
            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("Loss/val", val_loss, epoch)
            writer.add_scalar("Accuracy/train", train_acc, epoch)
            writer.add_scalar("Accuracy/val", val_acc, epoch)
            writer.add_scalar("LR", lr, epoch)

        if epoch <= 10 or epoch % 10 == 0 or val_loss < best_val_loss:
            print(f"{epoch:5d} {train_loss:11.4f} {train_acc:10.4f} {val_loss:10.4f} "
                  f"{val_acc:9.4f} {lr:10.2e} {elapsed:5.1f}s")

        # Checkpoint
        if epoch % 20 == 0:
            path = os.path.join(args.checkpoint_dir, f"epoch_{epoch}.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "info": info,
            }, path)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_loss": val_loss,
                "val_acc": val_acc,
                "info": info,
            }, os.path.join(args.checkpoint_dir, "best.pt"))
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping at epoch {epoch} (patience={args.patience})")
                break

    print(f"\nBest val loss: {best_val_loss:.4f}")
    print(f"Best checkpoint: {args.checkpoint_dir}/best.pt")

    if writer:
        writer.close()


if __name__ == "__main__":
    main()
