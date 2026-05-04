"""Data types for dynamic graph contraction in FusionDT.

Defines dataclasses for representing fusion clusters, graph step states,
and contracted graph metadata. These are used by graph_contractor.py to
build G_t from G_0 + cluster state at step t.

Key distinction:
  V_t = all nodes in contracted graph (alive originals + active clusters)
  C_t = candidate/action nodes (subset of V_t, from priority queue)
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Set, Tuple


@dataclass
class FusionCluster:
    """A dynamically-created fusion.N cluster in the HLO graph.

    Each cluster represents a group of original HLO nodes that have been
    fused together during XLA PriorityFusion. The cluster becomes a single
    super-node in the contracted graph G_t.
    """
    name: str                           # e.g. "fusion.39"
    members: FrozenSet[str]             # original HLO node names
    root_consumer: str                  # root instruction (traced to original)
    creation_step: int                  # MDP step index where cluster appeared
    fingerprint: str                    # stable identity hash
    feature_status: str = "exact"       # exact / approximate / missing

    @staticmethod
    def make_fingerprint(members: FrozenSet[str], root_consumer: str) -> str:
        """Stable cross-trajectory identifier based on member set + root."""
        key = "|".join(sorted(members)) + "||" + root_consumer
        return hashlib.md5(key.encode()).hexdigest()[:12]


@dataclass
class GraphStepState:
    """Full reconstruction state at a single fusion decision step.

    This separates V_t (graph nodes) from C_t (candidates):
      V_t = alive_original_nodes ∪ set(active_clusters.keys())
      C_t = candidate_names (subset of V_t node names)
    """
    module_key: str
    trajectory_idx: int
    step_idx: int
    step_position: int                          # position in fused-only step list

    # Graph node sets
    alive_original_nodes: FrozenSet[str]        # original nodes still standalone
    active_clusters: Dict[str, FusionCluster]   # fusion.N name -> cluster

    # Action info
    candidate_names: List[str]                  # sorted candidate names (C_t)
    selected_name: Optional[str]                # ground-truth selected producer

    # Fidelity
    graph_fidelity: str = "approximate"         # exact / approximate / unresolved
    notes: List[str] = field(default_factory=list)

    @property
    def node_names(self) -> List[str]:
        """All node names in V_t, sorted for deterministic ordering."""
        return sorted(self.alive_original_nodes | set(self.active_clusters.keys()))

    @property
    def num_nodes(self) -> int:
        return len(self.alive_original_nodes) + len(self.active_clusters)

    @property
    def num_candidates(self) -> int:
        return len(self.candidate_names)


@dataclass
class GraphStepMetadata:
    """Metadata for a contracted graph step, kept OUTSIDE PyG Data.

    This avoids PyG batching issues with dicts/strings/lists.
    """
    module_key: str
    trajectory_idx: int
    step_idx: int
    step_position: int

    # Node mapping (ordered same as contracted graph nodes)
    idx_to_name: List[str]              # contracted node index -> name
    name_to_idx: Dict[str, int]         # name -> contracted node index

    # Cluster info per node (empty for original nodes)
    cluster_members: Dict[str, FrozenSet[str]]  # fusion.N name -> member set

    # Action info
    candidate_names: List[str]
    selected_name: Optional[str]
    selected_idx: Optional[int]         # index into contracted graph nodes
    candidate_indices: List[int]        # candidate indices in contracted graph

    # Quality
    graph_fidelity: str
    notes: List[str]

    # Node type flags
    is_cluster: List[bool]              # per contracted node

    @property
    def num_nodes(self) -> int:
        return len(self.idx_to_name)

    @property
    def num_candidates(self) -> int:
        return len(self.candidate_names)


@dataclass
class FidelityReport:
    """Aggregate fidelity statistics across all reconstructed steps."""
    total_steps: int = 0
    exact_steps: int = 0
    approximate_steps: int = 0
    unresolved_steps: int = 0

    candidates_in_vt: int = 0           # candidates found in V_t
    candidates_missing: int = 0         # candidates NOT in V_t
    selected_in_ct: int = 0             # selected found in C_t
    selected_missing: int = 0           # selected NOT in C_t

    cluster_with_members: int = 0       # clusters with known members
    cluster_without_members: int = 0    # clusters with empty/unknown members

    notes: List[str] = field(default_factory=list)

    @property
    def exact_pct(self) -> float:
        return 100.0 * self.exact_steps / max(self.total_steps, 1)

    @property
    def approximate_pct(self) -> float:
        return 100.0 * self.approximate_steps / max(self.total_steps, 1)

    @property
    def unresolved_pct(self) -> float:
        return 100.0 * self.unresolved_steps / max(self.total_steps, 1)
