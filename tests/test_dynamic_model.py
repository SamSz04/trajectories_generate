"""Unit tests for DynamicFusionDT model.

Tests:
  1. Forward pass with variable graph sizes
  2. Same-step action leakage prevention
  3. Candidate masking correctness
  4. Gradient flow through GNN and Transformer
  5. Fusion.N nodes receive distinct embeddings
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
    # Mark last few nodes as clusters
    n_cluster = max(1, num_nodes // 4)
    is_cluster[-n_cluster:] = True
    # Zero out cluster features for non-cluster nodes
    cluster_features[~is_cluster] = 0.0

    candidate_mask = torch.zeros(num_nodes, dtype=torch.bool)
    cand_idx = torch.randperm(num_nodes)[:num_candidates]
    candidate_mask[cand_idx] = True

    return Data(
        x=x,
        edge_index=edge_index,
        opcode_ids=opcode_ids,
        cluster_features=cluster_features,
        is_cluster=is_cluster,
        candidate_mask=candidate_mask,
        num_nodes=num_nodes,
    )


def test_forward_variable_sizes():
    """Test forward pass with variable graph sizes in a batch."""
    model = DynamicFusionDT(embed_dim=32, num_heads=2, num_layers=1, d_ff=64, context_len=4)
    model.eval()

    B, K = 2, 4
    # Create graphs with different sizes
    sizes = [10, 8, 12, 6, 15, 7, 9, 11]  # B*K=8 graphs
    graphs = [_make_graph(s, s, max(1, s // 3)) for s in sizes]

    pyg_batch = Batch.from_data_list(graphs)
    graph_sizes = torch.tensor(sizes, dtype=torch.long)
    max_vt = max(sizes)

    actions = torch.randint(0, 5, (B, K))
    rtgs = torch.ones(B, K)
    timesteps = torch.arange(K).unsqueeze(0).expand(B, -1)
    attn_mask = torch.ones(B, K, dtype=torch.long)

    candidate_masks = torch.zeros(B, K, max_vt, dtype=torch.bool)
    for i, g in enumerate(graphs):
        n = g.candidate_mask.shape[0]
        candidate_masks[i // K, i % K, :n] = g.candidate_mask

    logits = model(pyg_batch, graph_sizes, actions, rtgs, timesteps, attn_mask, candidate_masks)

    assert logits.shape == (B, K, max_vt), f"Expected ({B}, {K}, {max_vt}), got {logits.shape}"
    # Non-candidate logits should be -inf
    for b in range(B):
        for k in range(K):
            non_cand = ~candidate_masks[b, k]
            assert (logits[b, k][non_cand] == float("-inf")).all(), \
                "Non-candidate logits should be -inf"
    print("PASS: test_forward_variable_sizes")


def test_no_action_leakage():
    """Verify a_t cannot be used to predict a_t (no same-step leakage).

    The model uses [R_t, s_t, a_{t-1}]. The logits for a_t come from the
    hidden state at s_t, which should NOT attend to a_t.
    """
    model = DynamicFusionDT(embed_dim=32, num_heads=2, num_layers=2, d_ff=64, context_len=4)

    B, K = 1, 4
    graphs = [_make_graph(8, 6, 4) for _ in range(K)]
    pyg_batch = Batch.from_data_list(graphs)
    graph_sizes = torch.tensor([8] * K, dtype=torch.long)

    actions = torch.tensor([[0, 1, 2, 3]])
    rtgs = torch.ones(1, K)
    timesteps = torch.arange(K).unsqueeze(0)
    attn_mask = torch.ones(1, K, dtype=torch.long)

    candidate_masks = torch.zeros(1, K, 8, dtype=torch.bool)
    for k in range(K):
        candidate_masks[0, k, :4] = True

    # Run with original actions
    model.eval()
    logits1 = model(pyg_batch, graph_sizes, actions, rtgs, timesteps, attn_mask, candidate_masks)

    # Change current-step actions (should NOT affect predictions at same step)
    # In [R_t, s_t, a_{t-1}], changing a_t should only affect predictions at step t+1
    actions2 = torch.tensor([[3, 2, 1, 0]])
    logits2 = model(pyg_batch, graph_sizes, actions2, rtgs, timesteps, attn_mask, candidate_masks)

    # Logits at step 0 should be identical (a_{-1} is BOS in both cases)
    assert torch.allclose(logits1[0, 0], logits2[0, 0], atol=1e-5), \
        "Step 0 logits should be identical (both use BOS)"

    # Logits at step 1 may differ (a_0 changed), but that's expected
    # because a_{t-1} legitimately influences step t's prediction

    print("PASS: test_no_action_leakage")


def test_candidate_masking():
    """Verify candidate mask correctly restricts action choices."""
    model = DynamicFusionDT(embed_dim=32, num_heads=2, num_layers=1, d_ff=64, context_len=2)
    model.eval()

    graphs = [_make_graph(10, 8, 3) for _ in range(2)]
    pyg_batch = Batch.from_data_list(graphs)
    graph_sizes = torch.tensor([10, 10], dtype=torch.long)

    actions = torch.tensor([[0, 1]])
    rtgs = torch.ones(1, 2)
    timesteps = torch.arange(2).unsqueeze(0)
    attn_mask = torch.ones(1, 2, dtype=torch.long)

    candidate_masks = torch.zeros(1, 2, 10, dtype=torch.bool)
    # Only nodes 2, 5, 7 are candidates at step 0
    candidate_masks[0, 0, [2, 5, 7]] = True
    candidate_masks[0, 1, [1, 3, 9]] = True

    logits = model(pyg_batch, graph_sizes, actions, rtgs, timesteps, attn_mask, candidate_masks)

    # Step 0: only nodes 2, 5, 7 should have finite logits
    finite_0 = (logits[0, 0] > float("-inf")).nonzero(as_tuple=True)[0].tolist()
    assert set(finite_0) == {2, 5, 7}, f"Expected {{2, 5, 7}}, got {set(finite_0)}"

    # Step 1: only nodes 1, 3, 9
    finite_1 = (logits[0, 1] > float("-inf")).nonzero(as_tuple=True)[0].tolist()
    assert set(finite_1) == {1, 3, 9}, f"Expected {{1, 3, 9}}, got {set(finite_1)}"

    print("PASS: test_candidate_masking")


def test_gradient_flow():
    """Verify gradients flow through both GNN and Transformer."""
    model = DynamicFusionDT(embed_dim=32, num_heads=2, num_layers=1, d_ff=64, context_len=2)

    graphs = [_make_graph(6, 4, 3) for _ in range(2)]
    pyg_batch = Batch.from_data_list(graphs)
    graph_sizes = torch.tensor([6, 6], dtype=torch.long)

    actions = torch.tensor([[0, 1]])
    rtgs = torch.ones(1, 2)
    timesteps = torch.arange(2).unsqueeze(0)
    attn_mask = torch.ones(1, 2, dtype=torch.long)

    candidate_masks = torch.zeros(1, 2, 6, dtype=torch.bool)
    candidate_masks[0, 0, :3] = True
    candidate_masks[0, 1, :3] = True

    logits = model(pyg_batch, graph_sizes, actions, rtgs, timesteps, attn_mask, candidate_masks)

    # Compute loss on valid candidates only
    target = torch.tensor([[0, 1]])
    loss = torch.nn.functional.cross_entropy(
        logits[0, 0, :3].unsqueeze(0), target[:, 0:1].squeeze(-1),
    )
    loss.backward()

    # Check GNN parameters have gradients
    gnn_has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.graph_encoder.gnn.parameters()
    )
    assert gnn_has_grad, "GNN parameters should have gradients"

    # Check Transformer parameters have gradients
    transformer_has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.blocks.parameters()
    )
    assert transformer_has_grad, "Transformer parameters should have gradients"

    # Check cluster projection has gradients
    cluster_has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.graph_encoder.cluster_proj.parameters()
    )
    assert cluster_has_grad, "Cluster projection should have gradients"

    print("PASS: test_gradient_flow")


def test_cluster_distinct_embeddings():
    """Verify fusion.N nodes get distinct embeddings from their features."""
    model = DynamicFusionDT(embed_dim=32, num_heads=2, num_layers=1, d_ff=64, context_len=1)
    model.eval()

    # Create two graphs: same structure but different cluster features
    g1 = _make_graph(5, 4, 3)
    g2 = _make_graph(5, 4, 3)

    # Make node 4 a cluster in both, but with different features
    g1.is_cluster[4] = True
    g2.is_cluster[4] = True
    g1.cluster_features[4] = torch.ones(34)
    g2.cluster_features[4] = torch.ones(34) * 5.0

    # Same base features and edges
    g2.x = g1.x.clone()
    g2.edge_index = g1.edge_index.clone()
    g2.opcode_ids = g1.opcode_ids.clone()

    batch = Batch.from_data_list([g1, g2])
    node_embeds = model.encode_graphs(batch)

    # Node 4 in g1 (index 4) and node 4 in g2 (index 9) should differ
    embed_g1_cluster = node_embeds[4]
    embed_g2_cluster = node_embeds[9]

    assert not torch.allclose(embed_g1_cluster, embed_g2_cluster, atol=1e-4), \
        "Cluster nodes with different features should have different embeddings"

    # Non-cluster nodes with same features should be identical
    embed_g1_orig = node_embeds[0]
    embed_g2_orig = node_embeds[5]
    assert torch.allclose(embed_g1_orig, embed_g2_orig, atol=1e-5), \
        "Original nodes with same features should have same embeddings"

    print("PASS: test_cluster_distinct_embeddings")


if __name__ == "__main__":
    test_forward_variable_sizes()
    test_no_action_leakage()
    test_candidate_masking()
    test_gradient_flow()
    test_cluster_distinct_embeddings()
    print("\nAll tests passed!")
