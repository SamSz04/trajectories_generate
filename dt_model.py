"""Decision Transformer model for XLA fusion ordering.

Architecture:
  GraphEncoder: opcode embedding + GraphSAGE → node embeddings [N, D]
  DecisionTransformer: causal transformer with pointer-network action head

Reuses GraphSAGE from references/GO_fusion/src/model/graphsage.py.
"""

from __future__ import annotations

import math
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "references", "GO_fusion"))

from src.model.graphsage import GraphSAGE


class GraphEncoder(nn.Module):
    """Encode HLO graph nodes via opcode embedding + GraphSAGE.

    Input: PyG Data with x [N, 19], opcode_ids [N], edge_index [2, E]
    Output: node_embeddings [N, hidden_dim]
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        num_opcodes: int = 132,
        opcode_dim: int = 16,
        num_cont_features: int = 27,
        num_gnn_layers: int = 2,
    ):
        super().__init__()
        self.opcode_embed = nn.Embedding(num_opcodes, opcode_dim)
        self.gnn = GraphSAGE(
            num_node_features=num_cont_features + opcode_dim,
            hidden_dim=hidden_dim,
            num_layers=num_gnn_layers,
        )

    def forward(self, data) -> torch.Tensor:
        opcode_emb = self.opcode_embed(data.opcode_ids)  # [N, 16]
        h = torch.cat([data.x, opcode_emb], dim=-1)  # [N, 35]
        return self.gnn(h, data.edge_index)  # [N, hidden_dim]


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


class DecisionTransformer(nn.Module):
    """Decision Transformer for fusion ordering with pointer-network action head.

    Per timestep, 3 tokens are created: (return, state, action).
    The state-position output predicts the action via dot-product with node embeddings.

    Args:
        num_nodes: Number of nodes in the HLO graph (56).
        embed_dim: Transformer hidden dimension (64).
        num_heads: Number of attention heads (4).
        num_layers: Number of transformer blocks (3).
        d_ff: Feed-forward inner dimension (128).
        context_len: Max number of timesteps per window (20).
        max_timestep: Max absolute timestep index (32).
        dropout: Dropout rate.
    """

    def __init__(
        self,
        num_nodes: int = 56,
        embed_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 3,
        d_ff: int = 128,
        context_len: int = 20,
        max_timestep: int = 256,
        dropout: float = 0.1,
        gnn_hidden_dim: int = None,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.num_actions = num_nodes + 1  # +1 for fusion token
        self.embed_dim = embed_dim
        self.context_len = context_len

        # Graph encoder — may use a different hidden dim than transformer
        gnn_dim = gnn_hidden_dim or embed_dim
        self.gnn_dim = gnn_dim
        self.graph_encoder = GraphEncoder(hidden_dim=gnn_dim)

        # Project GNN output to embed_dim if they differ
        if gnn_dim != embed_dim:
            self.node_proj = nn.Linear(gnn_dim, embed_dim)
        else:
            self.node_proj = None

        # Token embeddings (3 per timestep)
        self.return_embed = nn.Linear(1, embed_dim)
        # State = attention-pooled unfused nodes (D) + projected fused mask (D) → 2D
        self.state_attn_query = nn.Parameter(torch.randn(1, embed_dim) * 0.02)
        self.fused_mask_proj = nn.Linear(num_nodes, embed_dim)
        self.state_embed = nn.Linear(2 * embed_dim, embed_dim)
        self.action_embed = nn.Linear(embed_dim, embed_dim)

        # Positional
        self.timestep_embed = nn.Embedding(max_timestep, embed_dim)

        # Learned embedding for dynamically-created fusion.N nodes
        self.fusion_token = nn.Parameter(torch.randn(embed_dim) * 0.02)

        # Transformer
        self.blocks = nn.ModuleList([
            CausalTransformerBlock(embed_dim, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.final_ln = nn.LayerNorm(embed_dim)

        # Pointer-network action head
        self.query_proj = nn.Linear(embed_dim, embed_dim)

        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if "weight" in name and p.dim() >= 2:
                nn.init.xavier_uniform_(p)
            elif "bias" in name:
                nn.init.zeros_(p)

    def encode_graph(self, data) -> torch.Tensor:
        """Encode the HLO graph. Call once and cache the result.

        Returns: node_embeddings [N, embed_dim]
        """
        ne = self.graph_encoder(data)
        if self.node_proj is not None:
            ne = self.node_proj(ne)
        return ne

    def _get_all_action_embeds(self, node_embeds: torch.Tensor) -> torch.Tensor:
        """Concatenate node embeddings with fusion token.

        Returns: [num_actions, embed_dim] (57 x 64)
        """
        return torch.cat([node_embeds, self.fusion_token.unsqueeze(0)], dim=0)

    def _compute_states(
        self, fused_masks: torch.Tensor, node_embeds: torch.Tensor,
        node_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """Compute state vectors: attention-pooled unfused nodes + projected fused mask.

        Args:
            fused_masks: [B, K, N] binary mask of already-fused nodes
            node_embeds: [N, D] or [B, N, D] node embeddings
            node_mask: [B, N] optional mask for valid nodes (multi-module)

        Returns: [B, K, 2*D] state vectors
        """
        B, K, N = fused_masks.shape
        avail = 1.0 - fused_masks  # [B, K, N]
        if node_mask is not None:
            avail = avail * node_mask.unsqueeze(1)

        # Attention-pooled summary of unfused nodes
        # query: [1, D], keys: node_embeds [N, D] or [B, N, D]
        query = self.state_attn_query  # [1, D]
        if node_embeds.dim() == 2:
            # [1, D] @ [D, N] → [1, N] → expand to [B, K, N]
            attn_scores = torch.matmul(query, node_embeds.T)  # [1, N]
            attn_scores = attn_scores.unsqueeze(0).expand(B, K, -1)  # [B, K, N]
        else:
            # [B, 1, D] @ [B, D, N] → [B, 1, N] → expand to [B, K, N]
            attn_scores = torch.bmm(
                query.expand(B, -1, -1), node_embeds.transpose(1, 2)
            )  # [B, 1, N]
            attn_scores = attn_scores.expand(-1, K, -1)  # [B, K, N]

        # Mask fused/invalid nodes before softmax
        attn_scores = attn_scores.masked_fill(avail == 0, float('-inf'))
        attn_weights = F.softmax(attn_scores, dim=-1)  # [B, K, N]
        attn_weights = attn_weights.nan_to_num(0.0)  # handle all-masked case

        if node_embeds.dim() == 2:
            pool = torch.matmul(attn_weights, node_embeds)  # [B, K, D]
        else:
            pool = torch.einsum('bkn,bnd->bkd', attn_weights, node_embeds)

        # Project fused mask from N-dim binary → D-dim dense
        fused_proj = self.fused_mask_proj(fused_masks)  # [B, K, D]

        return torch.cat([pool, fused_proj], dim=-1)  # [B, K, 2*D]

    def forward(
        self,
        returns_to_go: torch.Tensor,  # [B, K]
        fused_masks: torch.Tensor,     # [B, K, N]
        actions: torch.Tensor,         # [B, K]
        timesteps: torch.Tensor,       # [B, K]
        attn_mask: torch.Tensor,       # [B, K]
        node_embeds: torch.Tensor,     # [N, D] or [B, N, D]
        node_mask: torch.Tensor = None,  # [B, N] valid-node mask (multi-module)
    ) -> torch.Tensor:
        """Forward pass.

        When node_embeds is 2D [N, D], operates in single-graph mode (original).
        When node_embeds is 3D [B, N, D], operates in multi-module mode with
        per-sample graphs. node_mask must be provided in this case.

        Returns: action_logits [B, K, num_actions]
        """
        B, K = returns_to_go.shape
        D = self.embed_dim
        device = returns_to_go.device
        batched = node_embeds.dim() == 3

        # State vectors
        states = self._compute_states(fused_masks, node_embeds, node_mask)

        # Time embeddings
        time_emb = self.timestep_embed(timesteps)  # [B, K, D]

        # Token embeddings
        r_tok = self.return_embed(returns_to_go.unsqueeze(-1)) + time_emb
        s_tok = self.state_embed(states) + time_emb

        # Action embeddings
        if batched:
            fusion = self.fusion_token.unsqueeze(0).unsqueeze(0).expand(B, 1, -1)
            all_action_embeds = torch.cat([node_embeds, fusion], dim=1)  # [B, N+1, D]
            a_emb = all_action_embeds[
                torch.arange(B, device=device).unsqueeze(1), actions
            ]  # [B, K, D]
        else:
            all_action_embeds = self._get_all_action_embeds(node_embeds)  # [N+1, D]
            a_emb = all_action_embeds[actions]  # [B, K, D]

        a_tok = self.action_embed(a_emb) + time_emb

        # Interleave: [r_1, s_1, a_1, r_2, s_2, a_2, ...]
        seq = torch.stack([r_tok, s_tok, a_tok], dim=2)  # [B, K, 3, D]
        seq = seq.reshape(B, 3 * K, D)

        # Causal attention mask (bool: True = masked)
        causal = torch.triu(
            torch.ones(3 * K, 3 * K, device=device, dtype=torch.bool), diagonal=1,
        )

        # Key padding mask (bool: True = masked/padding)
        key_pad = attn_mask.unsqueeze(2).expand(B, K, 3).reshape(B, 3 * K)
        key_pad = key_pad == 0

        # Transformer blocks
        h = seq
        for block in self.blocks:
            h = block(h, attn_mask=causal, key_padding_mask=key_pad)
        h = self.final_ln(h)

        # Extract state-position outputs (indices 1, 4, 7, ...)
        state_out = h[:, 1::3, :]  # [B, K, D]

        # Pointer-network action head
        query = self.query_proj(state_out)  # [B, K, D]
        if batched:
            logits = torch.einsum('bkd,bnd->bkn', query, all_action_embeds)
        else:
            logits = torch.matmul(query, all_action_embeds.T)  # [B, K, N+1]

        # Mask padding positions (multi-module: invalid nodes → -inf)
        if node_mask is not None:
            action_mask = torch.cat(
                [node_mask, torch.ones(B, 1, device=device)], dim=-1,
            )  # [B, N+1], fusion token always valid
            logits = logits.masked_fill(
                ~action_mask.unsqueeze(1).bool(), float('-inf'),
            )

        return logits

    @torch.no_grad()
    def predict_action(
        self,
        returns_to_go: torch.Tensor,  # [1, t]
        fused_masks: torch.Tensor,     # [1, t, N]
        actions: torch.Tensor,         # [1, t-1] (previous actions, empty for t=1)
        timesteps: torch.Tensor,       # [1, t]
        node_embeds: torch.Tensor,     # [N, D] or [1, N, D]
        node_mask: torch.Tensor = None,  # [1, N]
        temperature: float = 0.0,
    ) -> int:
        """Predict next action given history. For autoregressive inference.

        Returns: action index (0..num_actions-1)
        """
        self.eval()
        B, t = returns_to_go.shape
        device = returns_to_go.device

        # Pad actions with a dummy last action (will be ignored)
        if actions.shape[1] < t:
            dummy = torch.zeros(1, 1, dtype=torch.long, device=device)
            actions = torch.cat([actions, dummy], dim=1)

        # Truncate to context_len
        if t > self.context_len:
            start = t - self.context_len
            returns_to_go = returns_to_go[:, start:]
            fused_masks = fused_masks[:, start:]
            actions = actions[:, start:]
            timesteps = timesteps[:, start:]
            t = self.context_len

        attn_mask = torch.ones(1, t, device=device)
        logits = self.forward(
            returns_to_go, fused_masks, actions, timesteps, attn_mask,
            node_embeds, node_mask,
        )

        # Take logits at the last real timestep
        action_logits = logits[0, t - 1]  # [num_actions]

        # Mask already-fused regular nodes
        current_fused = fused_masks[0, t - 1]  # [N]
        action_logits[:self.num_nodes][current_fused[:self.num_nodes].bool()] = float("-inf")

        if temperature <= 0:
            return action_logits.argmax().item()
        else:
            probs = F.softmax(action_logits / temperature, dim=-1)
            return torch.multinomial(probs, 1).item()


if __name__ == "__main__":
    from dt_dataset import create_dataloaders

    traj_path = "output/trajectories_20260420_a100.pt"
    train_loader, val_loader, graph, info = create_dataloaders(traj_path)

    model = DecisionTransformer(
        num_nodes=info["num_nodes"],
        context_len=info["context_len"],
    )

    # Count parameters
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {total:,} total, {trainable:,} trainable")

    # Encode graph once
    node_embeds = model.encode_graph(graph)
    print(f"Node embeddings: {node_embeds.shape}")

    # Forward pass on a batch
    batch = next(iter(train_loader))
    logits = model(
        batch["returns_to_go"],
        batch["fused_masks"],
        batch["actions"],
        batch["timesteps"],
        batch["attn_mask"],
        node_embeds,
    )
    print(f"Logits shape: {logits.shape}")
    print(f"Expected: [B={batch['actions'].shape[0]}, K={info['context_len']}, "
          f"A={info['num_actions']}]")

    # Test autoregressive prediction
    action = model.predict_action(
        batch["returns_to_go"][:1, :1],
        batch["fused_masks"][:1, :1],
        torch.zeros(1, 0, dtype=torch.long),
        batch["timesteps"][:1, :1],
        node_embeds,
    )
    print(f"Predicted action: {action}")
