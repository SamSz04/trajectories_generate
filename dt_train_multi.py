"""Multi-module Decision Transformer training with real GPU rewards.

Trains on trajectories from all 22 HLO modules simultaneously, using
real GPU kernel times from nsys profiling as rewards.

Usage:
    PYTHONPATH=~/go_fusion python dt_train_multi.py \
        --traj-dir output/multi_trajectories \
        --gpu-csv output/gpu_profiles_all.csv \
        --reward-mode gpu \
        --epochs 300 --patience 40 --device cuda
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

from dt_dataset_multi import create_multi_dataloaders
from dt_model import DecisionTransformer


def encode_batch_graphs(model, module_graphs, module_ids, max_nodes, device):
    """Encode graphs for a batch, with proper autograd support.

    For each unique module in the batch, encodes its graph through the GNN,
    pads to max_nodes, and gathers into a batched tensor.

    Returns:
        node_embeds: [B, max_nodes, D] — with grad_fn for GNN training
        node_mask:   [B, max_nodes] — 1.0 for valid nodes, 0.0 for padding
    """
    B = module_ids.shape[0]
    D = model.embed_dim

    # Encode unique modules (preserves autograd)
    unique_ids = module_ids.unique()
    padded_cache = {}
    mask_cache = {}

    for mid in unique_ids:
        mid_val = mid.item()
        graph = module_graphs[mid_val].to(device)
        ne = model.encode_graph(graph)  # [N_i, D]
        n = ne.shape[0]
        padded = F.pad(ne, (0, 0, 0, max_nodes - n))  # [max_nodes, D]
        padded_cache[mid_val] = padded

        mask = torch.zeros(max_nodes, device=device)
        mask[:n] = 1.0
        mask_cache[mid_val] = mask

    # Gather by module_id: stack preserves autograd
    node_embeds = torch.stack(
        [padded_cache[mid.item()] for mid in module_ids]
    )  # [B, max_nodes, D]
    node_mask = torch.stack(
        [mask_cache[mid.item()] for mid in module_ids]
    )  # [B, max_nodes]

    return node_embeds, node_mask


def train_epoch(model, loader, optimizer, module_graphs, max_nodes, device):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    num_batches = len(loader)

    for batch_idx, batch in enumerate(loader):
        if batch_idx % 500 == 0:
            print(f"    train batch {batch_idx}/{num_batches}", flush=True)
        rtg = batch["returns_to_go"].to(device)
        masks = batch["fused_masks"].to(device)
        actions = batch["actions"].to(device)
        timesteps = batch["timesteps"].to(device)
        attn_mask = batch["attn_mask"].to(device)
        module_ids = batch["module_id"]

        # Encode graphs for this batch (with gradients)
        node_embeds, node_mask = encode_batch_graphs(
            model, module_graphs, module_ids, max_nodes, device,
        )

        logits = model(rtg, masks, actions, timesteps, attn_mask,
                       node_embeds, node_mask)

        # Cross-entropy on valid (non-padded) positions
        valid = attn_mask.bool()
        logits_flat = logits[valid]
        targets_flat = actions[valid]

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
def eval_epoch(model, loader, module_graphs, max_nodes, device, modules_info):
    """Evaluate with per-module accuracy breakdown."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    module_correct = {}
    module_count = {}

    for batch in loader:
        rtg = batch["returns_to_go"].to(device)
        masks = batch["fused_masks"].to(device)
        actions = batch["actions"].to(device)
        timesteps = batch["timesteps"].to(device)
        attn_mask = batch["attn_mask"].to(device)
        module_ids = batch["module_id"]

        node_embeds, node_mask = encode_batch_graphs(
            model, module_graphs, module_ids, max_nodes, device,
        )

        logits = model(rtg, masks, actions, timesteps, attn_mask,
                       node_embeds, node_mask)

        valid = attn_mask.bool()
        logits_flat = logits[valid]
        targets_flat = actions[valid]

        loss = F.cross_entropy(logits_flat, targets_flat)
        preds = logits_flat.argmax(dim=-1)
        correct = (preds == targets_flat)

        total_loss += loss.item() * targets_flat.shape[0]
        total_correct += correct.sum().item()
        total_count += targets_flat.shape[0]

        # Per-module tracking (on CPU to avoid device issues)
        B, K = attn_mask.shape
        valid_cpu = valid.cpu()
        correct_cpu = correct.cpu()
        mid_expanded = module_ids.unsqueeze(1).expand(B, K)[valid_cpu]
        for mid in mid_expanded.unique():
            mid_val = mid.item()
            m = mid_expanded == mid
            module_correct[mid_val] = module_correct.get(mid_val, 0) + correct_cpu[m].sum().item()
            module_count[mid_val] = module_count.get(mid_val, 0) + m.sum().item()

    avg_loss = total_loss / total_count if total_count > 0 else 0
    avg_acc = total_correct / total_count if total_count > 0 else 0

    per_module_acc = {}
    for mid_val in sorted(module_correct):
        key = modules_info[mid_val]["module_key"]
        acc = module_correct[mid_val] / module_count[mid_val]
        per_module_acc[key] = acc

    return avg_loss, avg_acc, per_module_acc


def main():
    parser = argparse.ArgumentParser(description="Multi-module DT Training")
    parser.add_argument("--traj-dir", default="output/multi_trajectories")
    parser.add_argument("--gpu-csv", default="output/gpu_profiles_all.csv")
    parser.add_argument("--reward-mode", default="gpu_hybrid",
                        choices=["gpu", "gpu_hybrid", "gpu_progressive", "cost_model"])
    parser.add_argument("--holdout-arch", default=None,
                        help="Architecture prefix to hold out for LOO eval")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--context-len", type=int, default=20)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--embed-dim", type=int, default=64,
                        help="Transformer hidden dimension (default: 64)")
    parser.add_argument("--num-heads", type=int, default=4,
                        help="Number of attention heads (default: 4)")
    parser.add_argument("--num-layers", type=int, default=3,
                        help="Number of transformer layers (default: 3)")
    parser.add_argument("--d-ff", type=int, default=128,
                        help="Feed-forward inner dimension (default: 128)")
    parser.add_argument("--gnn-dim", type=int, default=64,
                        help="GNN hidden dimension (default: 64, matches embed-dim)")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout rate (default: 0.1, try 0.3 for large models)")
    parser.add_argument("--checkpoint-dir", default="output/dt_multi_checkpoints")
    parser.add_argument("--resume", default=None,
                        help="Resume from checkpoint (path to .pt file, or 'auto' to resume from checkpoint-dir/best.pt)")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Data
    train_loader, val_loader, modules, info = create_multi_dataloaders(
        args.traj_dir, args.gpu_csv, args.reward_mode,
        context_len=args.context_len, batch_size=args.batch_size,
        holdout_arch=args.holdout_arch,
    )

    max_nodes = info["max_nodes"]
    max_timestep = info["max_timestep"] + 16  # small buffer

    print(f"=== Multi-Module DT Training ===")
    print(f"Modules: {info['num_modules']}, Max nodes: {max_nodes}")
    print(f"Trajectories: {info['num_train_trajs']} train / {info['num_val_trajs']} val")
    print(f"Windows: {info['num_train_windows']} train / {info['num_val_windows']} val")
    print(f"Actions: {info['num_actions']} (max_nodes + fusion_token)")
    print(f"Reward mode: {info['reward_mode']}")
    if "gpu_reward_range" in info:
        lo, hi = info["gpu_reward_range"]
        print(f"GPU reward range: [{lo:.4f}, {hi:.4f}], mean: {info['gpu_reward_mean']:.4f}")
    if args.holdout_arch:
        print(f"Holdout architecture: {args.holdout_arch}")
    print()

    # Extract graph list for encoding
    module_graphs = [m["graph"] for m in modules]

    # Model
    gnn_dim = args.gnn_dim if args.gnn_dim != 64 else args.embed_dim
    model = DecisionTransformer(
        num_nodes=max_nodes,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        d_ff=args.d_ff,
        context_len=args.context_len,
        max_timestep=max_timestep,
        gnn_hidden_dim=gnn_dim,
        dropout=args.dropout,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {total_params:,} parameters")
    print(f"  embed_dim={args.embed_dim}, heads={args.num_heads}, "
          f"layers={args.num_layers}, d_ff={args.d_ff}, gnn_dim={gnn_dim}")
    print(f"  state_embed input: {max_nodes} + {model.embed_dim} = {max_nodes + model.embed_dim}")
    print(f"  max_timestep: {max_timestep}")

    model_config = {
        "embed_dim": args.embed_dim,
        "num_heads": args.num_heads,
        "num_layers": args.num_layers,
        "d_ff": args.d_ff,
        "gnn_hidden_dim": gnn_dim,
        "num_nodes": max_nodes,
        "context_len": args.context_len,
        "max_timestep": max_timestep,
    }

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs,
    )

    # Resume from checkpoint
    start_epoch = 1
    best_val_acc = 0.0
    best_val_loss = float("inf")
    patience_counter = 0

    if args.resume:
        ckpt_path = args.resume
        if ckpt_path == "auto":
            ckpt_path = os.path.join(args.checkpoint_dir, "best.pt")
        if os.path.exists(ckpt_path):
            print(f"\nResuming from: {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            if "optimizer_state_dict" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            if "scheduler_state_dict" in ckpt:
                scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            if "best_val_acc" in ckpt:
                best_val_acc = ckpt["best_val_acc"]
                best_val_loss = ckpt.get("best_val_loss", float("inf"))
                patience_counter = ckpt.get("patience_counter", 0)
            elif "val_acc" in ckpt:
                best_val_acc = ckpt["val_acc"]
                best_val_loss = ckpt.get("val_loss", float("inf"))
            start_epoch = ckpt.get("epoch", 0) + 1
            print(f"  Resuming at epoch {start_epoch}, best_val_acc={best_val_acc:.4f}, "
                  f"patience={patience_counter}/{args.patience}")
        else:
            print(f"\nWARNING: checkpoint not found: {ckpt_path}, training from scratch")

    print(f"\nTraining for {args.epochs} epochs (patience={args.patience})...")
    print(f"{'Epoch':>5} {'TrLoss':>8} {'TrAcc':>7} {'VlLoss':>8} "
          f"{'VlAcc':>7} {'LR':>10} {'Time':>6}")
    print("-" * 60)

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        print(f"  [epoch {epoch}] starting train...", flush=True)

        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, module_graphs, max_nodes, device,
        )
        print(f"  [epoch {epoch}] train done, starting eval...", flush=True)
        val_loss, val_acc, per_module_acc = eval_epoch(
            model, val_loader, module_graphs, max_nodes, device, modules,
        )
        scheduler.step()

        elapsed = time.time() - t0
        lr = scheduler.get_last_lr()[0]

        if epoch <= 10 or epoch % 10 == 0 or val_loss < best_val_loss:
            print(f"{epoch:5d} {train_loss:8.4f} {train_acc:7.4f} {val_loss:8.4f} "
                  f"{val_acc:7.4f} {lr:10.2e} {elapsed:5.1f}s")

        # Periodic per-module accuracy
        if epoch % 50 == 0:
            print("  Per-module val accuracy:")
            for key, acc in per_module_acc.items():
                print(f"    {key}: {acc:.4f}")

        # Checkpoint
        if epoch % 50 == 0:
            path = os.path.join(args.checkpoint_dir, f"epoch_{epoch}.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_val_acc": best_val_acc,
                "best_val_loss": best_val_loss,
                "patience_counter": patience_counter,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "info": info,
                "model_config": model_config,
            }, path)

        # Early stopping on val accuracy (not loss)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_val_acc": best_val_acc,
                "best_val_loss": best_val_loss,
                "patience_counter": patience_counter,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "per_module_acc": per_module_acc,
                "info": info,
                "model_config": model_config,
            }, os.path.join(args.checkpoint_dir, "best.pt"))
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping at epoch {epoch} (patience={args.patience})")
                break

    # Final per-module breakdown
    print(f"\n{'='*60}")
    print(f"Best val accuracy: {best_val_acc:.4f} (loss: {best_val_loss:.4f})")
    print(f"\nFinal per-module validation accuracy:")
    for key, acc in per_module_acc.items():
        print(f"  {key}: {acc:.4f}")
    print(f"\nCheckpoint: {args.checkpoint_dir}/best.pt")


if __name__ == "__main__":
    main()
