"""Dynamic graph Decision Transformer with mutation-aware encoding.

Architecture:
  GraphEncoder: opcode embedding + GraphSAGE → per-node embeddings
  ClusterProjection: 34-dim cluster features → embed_dim
  AttentionPool: variable-size node set → fixed-dim state embedding
  CausalTransformer: [R_t, s_t, a_{t-1}] sequence
  PointerHead: query @ node_embeds.T → logits masked by C_t

Key differences from dt_model.py:
  - No fusion_token (clusters are real nodes with features)
  - No fused_mask_proj (no fixed-width mask)
  - No fixed num_nodes / num_actions
  - Variable graph sizes per step via dynamic contraction
  - [R_t, s_t, a_{t-1}] token order (predict a_t from s_t)
  - Candidate mask restricts action choices to C_t

Reuses GraphSAGE from references/GO_fusion/src/model/graphsage.py.
"""

from __future__ import annotations

import math
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "references", "GO_fusion"))

from src.model.graphsage import GraphSAGE


# ======================================================================
# Graph encoder (reused from dt_model.py, extended for cluster features)
# ======================================================================

class DynamicGraphEncoder(nn.Module):
    """Encode contracted HLO graph with cluster-aware node embeddings.

    Input: PyG Data/Batch with x [N, 19], opcode_ids [N], edge_index [2, E],
           cluster_features [N, 34], is_cluster [N]
    Output: node_embeddings [N, hidden_dim]
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        num_opcodes: int = 132,
        opcode_dim: int = 16,
        num_cont_features: int = 19,
        cluster_feature_dim: int = 34,
        num_gnn_layers: int = 2,
    ):
        super().__init__()
        self.opcode_embed = nn.Embedding(num_opcodes, opcode_dim)

        # Base features: continuous + opcode
        base_input_dim = num_cont_features + opcode_dim
        self.gnn = GraphSAGE(
            num_node_features=base_input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_gnn_layers,
        )

        # Cluster feature projection (added to base embedding)
        self.cluster_proj = nn.Sequential(
            nn.Linear(cluster_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, data) -> torch.Tensor:
        """Encode graph with cluster-aware features.

        Args:
            data: PyG Data/Batch with x, opcode_ids, edge_index,
                  cluster_features, is_cluster

        Returns: node_embeddings [N, hidden_dim]
        """
        opcode_emb = self.opcode_embed(data.opcode_ids)  # [N, opcode_dim]
        h = torch.cat([data.x, opcode_emb], dim=-1)      # [N, base_dim]
        base_embed = self.gnn(h, data.edge_index)         # [N, hidden_dim]

        # Add cluster features for cluster nodes (zero for originals)
        cluster_embed = self.cluster_proj(data.cluster_features)  # [N, hidden_dim]
        # Mask: only add for cluster nodes
        cluster_mask = data.is_cluster.float().unsqueeze(-1)      # [N, 1]
        node_embed = base_embed + cluster_embed * cluster_mask    # [N, hidden_dim]

        return node_embed


# ======================================================================
# Transformer blocks (reused from dt_model.py)
# ======================================================================

class CausalTransformerBlock(nn.Module):
    """Pre-norm transformer block with causal attention."""

    def __init__(self, embed_dim: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True,
        )
        self.ln2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        h = self.ln1(x)
        h, _ = self.attn(h, h, h, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        x = x + h
        h = self.ln2(x)
        x = x + self.ffn(h)
        return x


# ======================================================================
# Dynamic Fusion DT
# ======================================================================

class DynamicFusionDT(nn.Module):
    """Decision Transformer with dynamic graph contraction.

    Token sequence per timestep: [R_t, s_t, a_{t-1}]
    Prediction: a_t from hidden state at s_t position.

    Args:
        embed_dim: Transformer hidden dimension.
        num_heads: Number of attention heads.
        num_layers: Number of transformer blocks.
        d_ff: Feed-forward inner dimension.
        context_len: Max timesteps per window K.
        max_timestep: Max absolute timestep index.
        num_cont_features: Continuous feature dimension in G_0.
        cluster_feature_dim: Cluster feature dimension.
        gnn_hidden_dim: GNN hidden dim (defaults to embed_dim).
        dropout: Dropout rate.
    """

    def __init__(
        self,
        embed_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 3,
        d_ff: int = 128,
        context_len: int = 20,
        max_timestep: int = 1024,
        num_cont_features: int = 19,
        cluster_feature_dim: int = 34,
        gnn_hidden_dim: int = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.context_len = context_len

        # Graph encoder
        gnn_dim = gnn_hidden_dim or embed_dim
        self.graph_encoder = DynamicGraphEncoder(
            hidden_dim=gnn_dim,
            num_cont_features=num_cont_features,
            cluster_feature_dim=cluster_feature_dim,
        )

        # Project GNN output to embed_dim if needed
        if gnn_dim != embed_dim:
            self.node_proj = nn.Linear(gnn_dim, embed_dim)
        else:
            self.node_proj = None

        # Token embeddings
        # Return embedding: scalar → embed_dim
        self.return_embed = nn.Linear(1, embed_dim)

        # State embedding: attention-pooled node embeddings → embed_dim
        self.state_attn_query = nn.Parameter(torch.randn(1, embed_dim) * 0.02)
        self.state_proj = nn.Linear(embed_dim, embed_dim)

        # Action embedding: node embedding of selected action → embed_dim
        # For a_{t-1} at step t. BOS token for first step.
        self.action_proj = nn.Linear(embed_dim, embed_dim)
        self.bos_action = nn.Parameter(torch.randn(embed_dim) * 0.02)

        # Positional
        self.timestep_embed = nn.Embedding(max_timestep, embed_dim)

        # Transformer backbone
        self.blocks = nn.ModuleList([
            CausalTransformerBlock(embed_dim, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.final_ln = nn.LayerNorm(embed_dim)

        # Pointer head: project state hidden → query for dot-product with nodes
        self.query_proj = nn.Linear(embed_dim, embed_dim)

        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if "weight" in name and p.dim() >= 2:
                nn.init.xavier_uniform_(p)
            elif "bias" in name:
                nn.init.zeros_(p)

    def encode_graphs(self, pyg_batch: Batch) -> torch.Tensor:
        """Encode all graphs in a PyG Batch.

        Args:
            pyg_batch: Batch of B*K contracted graphs

        Returns: all_node_embeds [total_nodes, embed_dim]
        """
        node_embeds = self.graph_encoder(pyg_batch)
        if self.node_proj is not None:
            node_embeds = self.node_proj(node_embeds)
        return node_embeds

    def forward(
        self,
        pyg_batch: Batch,
        graph_sizes: torch.Tensor,
        actions: torch.Tensor,
        rtgs: torch.Tensor,
        timesteps: torch.Tensor,
        attn_mask: torch.Tensor,
        candidate_masks: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            pyg_batch: Batch of B*K PyG graphs
            graph_sizes: [B*K] number of nodes per graph
            actions: [B, K] action indices (into contracted graph)
            rtgs: [B, K] return-to-go values
            timesteps: [B, K] timestep indices
            attn_mask: [B, K] valid step mask
            candidate_masks: [B, K, max_Vt] candidate masks

        Returns:
            logits: [B, K, max_Vt] action logits (masked by candidates)
        """
        B, K = actions.shape
        max_vt = candidate_masks.shape[2]
        device = actions.device

        # 1. Encode all graphs
        all_node_embeds = self.encode_graphs(pyg_batch)  # [total_nodes, D]

        # 2. Split node embeddings by graph and pad to max_Vt
        # graph_sizes: [B*K], node embeds split accordingly
        node_embeds_padded = torch.zeros(B * K, max_vt, self.embed_dim, device=device)
        node_mask = torch.zeros(B * K, max_vt, dtype=torch.bool, device=device)

        offset = 0
        for i, sz in enumerate(graph_sizes.tolist()):
            sz = min(sz, max_vt)
            node_embeds_padded[i, :sz] = all_node_embeds[offset:offset + sz]
            node_mask[i, :sz] = True
            offset += sz

        node_embeds_padded = node_embeds_padded.view(B, K, max_vt, self.embed_dim)
        node_mask = node_mask.view(B, K, max_vt)

        # 3. Build token sequence: [R_t, s_t, a_{t-1}] for each timestep
        # R_t: return embedding
        rtg_tokens = self.return_embed(rtgs.unsqueeze(-1))  # [B, K, D]

        # s_t: attention-pooled node embeddings
        query = self.state_attn_query  # [1, D]
        attn_scores = torch.einsum("d,bknd->bkn", query.squeeze(0), node_embeds_padded)
        attn_scores = attn_scores.masked_fill(~node_mask, float("-inf"))
        attn_weights = F.softmax(attn_scores, dim=-1).nan_to_num(0.0)  # [B, K, max_Vt]
        state_pool = torch.einsum("bkn,bknd->bkd", attn_weights, node_embeds_padded)
        state_tokens = self.state_proj(state_pool)  # [B, K, D]

        # a_{t-1}: node embedding of previous action (BOS for first step)
        # Shift actions right: a_{t-1} for step t
        prev_actions = actions[:, :-1]  # [B, K-1]
        prev_action_embeds = torch.zeros(B, K, self.embed_dim, device=device)
        prev_action_embeds[:, 0] = self.bos_action  # BOS for first step

        for b in range(B):
            for k in range(K - 1):
                a_idx = prev_actions[b, k].item()
                if a_idx >= 0 and a_idx < graph_sizes[b * K + k].item():
                    # Get node embedding from step k (not k+1!)
                    prev_action_embeds[b, k + 1] = node_embeds_padded[b, k, a_idx]

        action_tokens = self.action_proj(prev_action_embeds)  # [B, K, D]

        # Add timestep embeddings
        ts_emb = self.timestep_embed(timesteps.clamp(0, 1023))  # [B, K, D]
        rtg_tokens = rtg_tokens + ts_emb
        state_tokens = state_tokens + ts_emb
        action_tokens = action_tokens + ts_emb

        # 4. Interleave: [R_0, s_0, a_{-1}, R_1, s_1, a_0, ...]
        # Total sequence length = 3K
        seq = torch.zeros(B, 3 * K, self.embed_dim, device=device)
        seq[:, 0::3] = rtg_tokens      # positions 0, 3, 6, ...
        seq[:, 1::3] = state_tokens     # positions 1, 4, 7, ...
        seq[:, 2::3] = action_tokens    # positions 2, 5, 8, ...

        # 5. Build causal mask (3K x 3K)
        seq_len = 3 * K
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device), diagonal=1,
        ).bool()

        # Key padding mask: repeat attn_mask 3x for each token type
        # [B, 3K]
        key_pad = attn_mask.repeat_interleave(3, dim=1) == 0  # True = ignore

        # 6. Run transformer
        h = seq
        for block in self.blocks:
            h = block(h, attn_mask=causal_mask, key_padding_mask=key_pad)
        h = self.final_ln(h)

        # 7. Extract hidden states at state positions (predict a_t from s_t)
        state_hidden = h[:, 1::3]  # [B, K, D] — positions 1, 4, 7, ...

        # 8. Pointer head: dot product with node embeddings
        queries = self.query_proj(state_hidden)  # [B, K, D]
        # logits[b, k, v] = queries[b, k] @ node_embeds_padded[b, k, v]
        logits = torch.einsum("bkd,bknd->bkn", queries, node_embeds_padded)

        # Mask non-candidates
        logits = logits.masked_fill(~candidate_masks, float("-inf"))

        return logits

    def predict_action(
        self,
        pyg_batch: Batch,
        graph_sizes: torch.Tensor,
        actions: torch.Tensor,
        rtgs: torch.Tensor,
        timesteps: torch.Tensor,
        attn_mask: torch.Tensor,
        candidate_masks: torch.Tensor,
        step_idx: int = -1,
    ) -> torch.Tensor:
        """Predict action at a specific step (for inference).

        Returns: action_indices [B] — selected node index in contracted graph.
        """
        logits = self.forward(
            pyg_batch, graph_sizes, actions, rtgs, timesteps,
            attn_mask, candidate_masks,
        )
        # Get logits at the target step
        step_logits = logits[:, step_idx]  # [B, max_Vt]
        return step_logits.argmax(dim=-1)  # [B]

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
