"""Candidate-ranking priority scorer model for XLA fusion.

Scores each candidate producer at each decision step. Two separate
encoders handle original HLO nodes and dynamically-created fusion
clusters, producing a shared embedding space. A ScoreHead maps
(candidate_embed, context) → scalar priority score.

Architecture:
    CandidateEncoder:
      Original nodes:  MLP(node_features[19] + opcode_embed[16]) -> [64]
      Fusion clusters: MLP(cluster_features[34]) -> [64]
      Two separate MLPs, same output dim.

    ContextEncoder:
      MLP(step_features[5]) -> [32]

    PriorityScorer:
      scores = ScoreHead(concat(candidate_embed[64], context[32])) -> [1]

Total trainable params: ~36K (without GNN), ~40K (with frozen GNN pooling).
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from cluster_features import CLUSTER_FEATURE_DIM


# ======================================================================
# Constants
# ======================================================================

ORIGINAL_FEATURE_DIM = 19    # data.x continuous features (19 in current .pt files)
NUM_OPCODES = 132            # HLO opcode vocabulary size
OPCODE_EMBED_DIM = 16        # opcode embedding dimension
CANDIDATE_EMBED_DIM = 64     # shared output dim for both encoders
CONTEXT_DIM = 32             # context encoder output dim
CONTEXT_INPUT_DIM = 5        # step-level context features
SCORE_HIDDEN_DIM = 128       # ScoreHead hidden dim


# ======================================================================
# Sub-modules
# ======================================================================

class OriginalNodeEncoder(nn.Module):
    """Encode original HLO nodes: node_features[27] + opcode_embed[16] -> [64]."""

    def __init__(
        self,
        feature_dim: int = ORIGINAL_FEATURE_DIM,
        opcode_vocab_size: int = NUM_OPCODES,
        opcode_embed_dim: int = OPCODE_EMBED_DIM,
        output_dim: int = CANDIDATE_EMBED_DIM,
        gnn_embed_dim: int = 0,
    ):
        super().__init__()
        self.opcode_embed = nn.Embedding(opcode_vocab_size, opcode_embed_dim)
        input_dim = feature_dim + opcode_embed_dim + gnn_embed_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
        )

    def forward(
        self,
        features: torch.Tensor,     # [B, N, 27]
        opcode_ids: torch.Tensor,    # [B, N] long
        gnn_embeds: Optional[torch.Tensor] = None,  # [B, N, gnn_dim]
    ) -> torch.Tensor:
        """Returns [B, N, 64]."""
        op_emb = self.opcode_embed(opcode_ids)              # [B, N, 16]
        parts = [features, op_emb]
        if gnn_embeds is not None:
            parts.append(gnn_embeds)
        x = torch.cat(parts, dim=-1)                        # [B, N, input_dim]
        return self.mlp(x)                                   # [B, N, 64]


class ClusterEncoder(nn.Module):
    """Encode fusion clusters: cluster_features[34] -> [64]."""

    def __init__(
        self,
        feature_dim: int = CLUSTER_FEATURE_DIM,
        output_dim: int = CANDIDATE_EMBED_DIM,
        gnn_pool_dim: int = 0,
    ):
        super().__init__()
        input_dim = feature_dim + gnn_pool_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
        )

    def forward(
        self,
        features: torch.Tensor,     # [B, N, 34]
        gnn_pool: Optional[torch.Tensor] = None,  # [B, N, gnn_pool_dim]
    ) -> torch.Tensor:
        """Returns [B, N, 64]."""
        if gnn_pool is not None:
            features = torch.cat([features, gnn_pool], dim=-1)
        return self.mlp(features)


class ContextEncoder(nn.Module):
    """Encode step-level context: [5] -> [32].

    Context features (computed at dataset time):
      [0] step_progress      = step_idx / total_steps
      [1] frac_fused          = n_fused_so_far / n_original_nodes
      [2] frac_alive          = n_alive / n_original_nodes
      [3] frac_fusion_cands   = n_fusion_in_alive / n_alive
      [4] log2(n_fusions + 1) = log2 of fusion clusters created so far
    """

    def __init__(
        self,
        input_dim: int = CONTEXT_INPUT_DIM,
        output_dim: int = CONTEXT_DIM,
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        """context: [B, 5] -> [B, 32]."""
        return self.mlp(context)


class ScoreHead(nn.Module):
    """Map (candidate_embed, context) -> scalar score.

    Input: concat(candidate[64], context[32]) = [96]
    Output: scalar per candidate
    """

    def __init__(
        self,
        candidate_dim: int = CANDIDATE_EMBED_DIM,
        context_dim: int = CONTEXT_DIM,
        hidden_dim: int = SCORE_HIDDEN_DIM,
        dropout: float = 0.1,
    ):
        super().__init__()
        input_dim = candidate_dim + context_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, N, 96] -> [B, N] (scores)."""
        return self.net(x).squeeze(-1)


# ======================================================================
# Main model
# ======================================================================

class PriorityScorer(nn.Module):
    """Candidate-ranking model for XLA fusion priority scoring.

    At each decision step, scores all candidate producers (original nodes
    and fusion clusters) and selects the highest-scored one.

    Forward pass:
      1. Encode original-node candidates via OriginalNodeEncoder
      2. Encode fusion-cluster candidates via ClusterEncoder
      3. Merge into unified candidate embeddings (same dim)
      4. Encode step context via ContextEncoder
      5. Score each candidate via ScoreHead
      6. Mask padded candidates

    Training: softmax cross-entropy over candidate scores.
    Inference: argmax over scores → selected candidate.
    """

    def __init__(
        self,
        original_feature_dim: int = ORIGINAL_FEATURE_DIM,
        cluster_feature_dim: int = CLUSTER_FEATURE_DIM,
        opcode_vocab_size: int = NUM_OPCODES,
        opcode_embed_dim: int = OPCODE_EMBED_DIM,
        candidate_embed_dim: int = CANDIDATE_EMBED_DIM,
        context_input_dim: int = CONTEXT_INPUT_DIM,
        context_dim: int = CONTEXT_DIM,
        score_hidden_dim: int = SCORE_HIDDEN_DIM,
        dropout: float = 0.1,
        gnn_embed_dim: int = 0,
        gnn_pool_dim: int = 0,
    ):
        super().__init__()

        self.original_encoder = OriginalNodeEncoder(
            feature_dim=original_feature_dim,
            opcode_vocab_size=opcode_vocab_size,
            opcode_embed_dim=opcode_embed_dim,
            output_dim=candidate_embed_dim,
            gnn_embed_dim=gnn_embed_dim,
        )
        self.cluster_encoder = ClusterEncoder(
            feature_dim=cluster_feature_dim,
            output_dim=candidate_embed_dim,
            gnn_pool_dim=gnn_pool_dim,
        )
        self.context_encoder = ContextEncoder(
            input_dim=context_input_dim,
            output_dim=context_dim,
        )
        self.score_head = ScoreHead(
            candidate_dim=candidate_embed_dim,
            context_dim=context_dim,
            hidden_dim=score_hidden_dim,
            dropout=dropout,
        )

    def forward(
        self,
        # Original node candidates
        orig_features: torch.Tensor,        # [B, N_orig, 27]
        orig_opcode_ids: torch.Tensor,      # [B, N_orig] long
        orig_mask: torch.Tensor,            # [B, N_orig] bool (True = valid)
        # Fusion cluster candidates
        cluster_features: torch.Tensor,     # [B, N_clust, 34]
        cluster_mask: torch.Tensor,         # [B, N_clust] bool
        # Step context
        context: torch.Tensor,              # [B, 5]
        # Optional GNN embeddings
        orig_gnn_embeds: Optional[torch.Tensor] = None,   # [B, N_orig, gnn_dim]
        cluster_gnn_pool: Optional[torch.Tensor] = None,  # [B, N_clust, gnn_pool_dim]
    ) -> torch.Tensor:
        """Compute scores for all candidates.

        Returns:
            scores: [B, N_orig + N_clust], masked with -inf for padding.
        """
        # Encode candidates
        orig_emb = self.original_encoder(
            orig_features, orig_opcode_ids, orig_gnn_embeds,
        )  # [B, N_orig, 64]
        clust_emb = self.cluster_encoder(
            cluster_features, cluster_gnn_pool,
        )  # [B, N_clust, 64]

        # Concatenate all candidates
        cand_emb = torch.cat([orig_emb, clust_emb], dim=1)   # [B, N_total, 64]
        cand_mask = torch.cat([orig_mask, cluster_mask], dim=1)  # [B, N_total]

        # Encode context and expand to match candidates
        ctx = self.context_encoder(context)                   # [B, 32]
        ctx_exp = ctx.unsqueeze(1).expand(-1, cand_emb.size(1), -1)  # [B, N_total, 32]

        # Score
        score_input = torch.cat([cand_emb, ctx_exp], dim=-1)  # [B, N_total, 96]
        scores = self.score_head(score_input)                  # [B, N_total]

        # Mask invalid candidates
        scores = scores.masked_fill(~cand_mask, float('-inf'))

        return scores

    def compute_loss(
        self,
        scores: torch.Tensor,           # [B, N_total]
        target_idx: torch.Tensor,        # [B] long — index of selected candidate
        sample_weight: Optional[torch.Tensor] = None,  # [B] for advantage weighting
    ) -> torch.Tensor:
        """Softmax cross-entropy loss over candidate set.

        L = -log( exp(score_selected) / sum_j exp(score_j) )
        """
        loss = F.cross_entropy(scores, target_idx, reduction='none')  # [B]
        if sample_weight is not None:
            loss = loss * sample_weight
        return loss.mean()

    @torch.no_grad()
    def predict(
        self, scores: torch.Tensor,
    ) -> torch.Tensor:
        """Return index of highest-scored candidate per batch element."""
        return scores.argmax(dim=-1)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ======================================================================
# Factory
# ======================================================================

def build_scorer(
    gnn_embed_dim: int = 0,
    gnn_pool_dim: int = 0,
    dropout: float = 0.1,
) -> PriorityScorer:
    """Build a PriorityScorer with default hyperparameters."""
    model = PriorityScorer(
        gnn_embed_dim=gnn_embed_dim,
        gnn_pool_dim=gnn_pool_dim,
        dropout=dropout,
    )
    return model


# ======================================================================
# CLI: print model summary
# ======================================================================

if __name__ == "__main__":
    model = build_scorer()
    print(f"PriorityScorer — {model.param_count():,} trainable params")
    print()

    # Print per-module param counts
    for name, mod in [
        ("OriginalNodeEncoder", model.original_encoder),
        ("ClusterEncoder", model.cluster_encoder),
        ("ContextEncoder", model.context_encoder),
        ("ScoreHead", model.score_head),
    ]:
        n = sum(p.numel() for p in mod.parameters() if p.requires_grad)
        print(f"  {name:25s}: {n:,}")

    # Test forward pass
    B, N_orig, N_clust = 4, 20, 10
    scores = model(
        orig_features=torch.randn(B, N_orig, ORIGINAL_FEATURE_DIM),
        orig_opcode_ids=torch.randint(0, NUM_OPCODES, (B, N_orig)),
        orig_mask=torch.ones(B, N_orig, dtype=torch.bool),
        cluster_features=torch.randn(B, N_clust, CLUSTER_FEATURE_DIM),
        cluster_mask=torch.ones(B, N_clust, dtype=torch.bool),
        context=torch.randn(B, CONTEXT_INPUT_DIM),
    )
    print(f"\n  Forward pass: scores shape = {scores.shape}")
    assert scores.shape == (B, N_orig + N_clust)

    # Test loss
    target = torch.randint(0, N_orig + N_clust, (B,))
    loss = model.compute_loss(scores, target)
    print(f"  Loss: {loss.item():.4f}")

    # Test with advantage weights
    weights = torch.tensor([1.0, 2.0, 0.5, 1.5])
    loss_w = model.compute_loss(scores, target, sample_weight=weights)
    print(f"  Weighted loss: {loss_w.item():.4f}")

    # Test prediction
    pred = model.predict(scores)
    print(f"  Predictions: {pred.tolist()}")

    # Test with masking
    mask = torch.ones(B, N_orig, dtype=torch.bool)
    mask[:, 15:] = False  # mask last 5 original candidates
    scores_masked = model(
        orig_features=torch.randn(B, N_orig, ORIGINAL_FEATURE_DIM),
        orig_opcode_ids=torch.randint(0, NUM_OPCODES, (B, N_orig)),
        orig_mask=mask,
        cluster_features=torch.randn(B, N_clust, CLUSTER_FEATURE_DIM),
        cluster_mask=torch.ones(B, N_clust, dtype=torch.bool),
        context=torch.randn(B, CONTEXT_INPUT_DIM),
    )
    assert (scores_masked[:, 15:20] == float('-inf')).all(), "Masked scores should be -inf"
    print(f"  Masking OK: masked positions are -inf")

    print(f"\n  All checks passed!")
