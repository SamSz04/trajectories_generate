"""Tests for DynamicFusionDT model with variable return_dim.

Tests:
  1. Forward pass with return_dim=1 (single channel, backward compat)
  2. Forward pass with return_dim=2 (dual channel, hybrid mode)
  3. Backward compat: [B, K] rtgs (2D) auto-unsqueezed to [B, K, 1]
  4. MLP return embed produces correct output shape
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch_geometric.data import Data, Batch

from dt_model_dynamic import DynamicFusionDT


def _make_graph(num_nodes: int, num_edges: int, num_candidates: int) -> Data:
    """Create a synthetic contracted graph."""
    x = torch.randn(num_nodes, 19)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    opcode_ids = torch.randint(0, 132, (num_nodes,))
    cluster_features = torch.randn(num_nodes, 34)
    is_cluster = torch.zeros(num_nodes, dtype=torch.bool)
    n_cluster = max(1, num_nodes // 4)
    is_cluster[-n_cluster:] = True
    cluster_features[~is_cluster] = 0.0
    candidate_mask = torch.zeros(num_nodes, dtype=torch.bool)
    cand_idx = torch.randperm(num_nodes)[:num_candidates]
    candidate_mask[cand_idx] = True
    return Data(
        x=x, edge_index=edge_index, opcode_ids=opcode_ids,
        cluster_features=cluster_features, is_cluster=is_cluster,
        candidate_mask=candidate_mask, num_nodes=num_nodes,
    )


def _build_batch(B=2, K=4):
    """Build a test batch."""
    sizes = [10, 8, 12, 6, 15, 7, 9, 11][:B * K]
    graphs = [_make_graph(s, s, max(1, s // 3)) for s in sizes]
    pyg_batch = Batch.from_data_list(graphs)
    graph_sizes = torch.tensor([g.num_nodes for g in graphs], dtype=torch.long)
    max_vt = max(sizes)

    actions = torch.stack([
        torch.tensor([min(1, s - 1) for s in sizes[b * K:(b + 1) * K]])
        for b in range(B)
    ])

    candidate_masks = torch.zeros(B, K, max_vt, dtype=torch.bool)
    for i, g in enumerate(graphs):
        b, k = i // K, i % K
        n = g.candidate_mask.shape[0]
        candidate_masks[b, k, :n] = g.candidate_mask

    timesteps = torch.arange(K).unsqueeze(0).expand(B, -1)
    attn_mask = torch.ones(B, K, dtype=torch.long)

    return pyg_batch, graph_sizes, actions, timesteps, attn_mask, candidate_masks, max_vt


def test_return_dim_1():
    """Forward pass with return_dim=1 produces correct output shape."""
    model = DynamicFusionDT(
        embed_dim=32, num_heads=2, num_layers=1, d_ff=64,
        context_len=4, return_dim=1,
    )
    model.eval()

    B, K = 2, 4
    pyg_batch, graph_sizes, actions, timesteps, attn_mask, candidate_masks, max_vt = _build_batch(B, K)
    rtgs = torch.randn(B, K, 1)  # [B, K, 1]

    with torch.no_grad():
        logits = model(pyg_batch, graph_sizes, actions, rtgs, timesteps, attn_mask, candidate_masks)

    assert logits.shape == (B, K, max_vt), f"Expected {(B, K, max_vt)}, got {logits.shape}"
    print("  PASS  test_return_dim_1")


def test_return_dim_2():
    """Forward pass with return_dim=2 (hybrid mode)."""
    model = DynamicFusionDT(
        embed_dim=32, num_heads=2, num_layers=1, d_ff=64,
        context_len=4, return_dim=2,
    )
    model.eval()

    B, K = 2, 4
    pyg_batch, graph_sizes, actions, timesteps, attn_mask, candidate_masks, max_vt = _build_batch(B, K)
    rtgs = torch.randn(B, K, 2)  # [B, K, 2]

    with torch.no_grad():
        logits = model(pyg_batch, graph_sizes, actions, rtgs, timesteps, attn_mask, candidate_masks)

    assert logits.shape == (B, K, max_vt), f"Expected {(B, K, max_vt)}, got {logits.shape}"
    print("  PASS  test_return_dim_2")


def test_backward_compat_2d_rtgs():
    """Old-style [B, K] rtgs should be auto-unsqueezed to [B, K, 1]."""
    model = DynamicFusionDT(
        embed_dim=32, num_heads=2, num_layers=1, d_ff=64,
        context_len=4, return_dim=1,
    )
    model.eval()

    B, K = 2, 4
    pyg_batch, graph_sizes, actions, timesteps, attn_mask, candidate_masks, max_vt = _build_batch(B, K)
    rtgs_2d = torch.randn(B, K)  # [B, K] — old format

    with torch.no_grad():
        logits = model(pyg_batch, graph_sizes, actions, rtgs_2d, timesteps, attn_mask, candidate_masks)

    assert logits.shape == (B, K, max_vt), f"Expected {(B, K, max_vt)}, got {logits.shape}"
    print("  PASS  test_backward_compat_2d_rtgs")


def test_mlp_return_embed_shape():
    """MLP return embed transforms return_dim → embed_dim correctly."""
    for rd in [1, 2]:
        model = DynamicFusionDT(
            embed_dim=32, num_heads=2, num_layers=1, d_ff=64,
            context_len=4, return_dim=rd,
        )
        assert model.return_dim == rd

        # Check MLP layers
        layers = list(model.return_embed)
        assert len(layers) == 3  # Linear, GELU, Linear
        assert layers[0].in_features == rd
        assert layers[0].out_features == 32
        assert layers[2].in_features == 32
        assert layers[2].out_features == 32

    print("  PASS  test_mlp_return_embed_shape")


def test_gradient_flow_return_dim_2():
    """Gradients flow through MLP return embed with return_dim=2."""
    model = DynamicFusionDT(
        embed_dim=32, num_heads=2, num_layers=1, d_ff=64,
        context_len=4, return_dim=2,
    )
    model.train()

    B, K = 1, 4
    pyg_batch, graph_sizes, actions, timesteps, attn_mask, candidate_masks, max_vt = _build_batch(B, K)
    rtgs = torch.randn(B, K, 2, requires_grad=True)

    logits = model(pyg_batch, graph_sizes, actions, rtgs, timesteps, attn_mask, candidate_masks)
    loss = logits.sum()
    loss.backward()

    # Check gradient flows to return_embed
    for name, p in model.return_embed.named_parameters():
        assert p.grad is not None, f"No gradient for return_embed.{name}"
        assert p.grad.abs().sum() > 0, f"Zero gradient for return_embed.{name}"

    print("  PASS  test_gradient_flow_return_dim_2")


if __name__ == "__main__":
    test_return_dim_1()
    test_return_dim_2()
    test_backward_compat_2d_rtgs()
    test_mlp_return_embed_shape()
    test_gradient_flow_return_dim_2()
    print("\nAll tests passed!")
