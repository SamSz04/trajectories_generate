"""Graph contraction engine for dynamic FusionDT.

Builds contracted graph G_t from G_0 + cluster state at step t.
Each fusion.N cluster becomes a super-node with aggregated features
and rewired edges. The action mask restricts scoring to C_t.

Usage:
    from graph_contractor import reconstruct_trajectory_states, contract_graph

    states = reconstruct_trajectory_states(enriched_traj, g0)
    for state in states:
        data, metadata = contract_graph(g0, state)
"""

from __future__ import annotations

import math
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

import torch
from torch_geometric.data import Data

from dynamic_graph_types import (
    FusionCluster,
    GraphStepState,
    GraphStepMetadata,
    FidelityReport,
)


# ======================================================================
# Feature layout constants (same as cluster_features.py)
# ======================================================================

_FEAT_ETYPE = slice(0, 6)       # one-hot element type (6 dims)
_FEAT_RANK = 6                  # tensor rank
_FEAT_ELEMENTS = 11             # log2(total elements)
_FEAT_BYTES = 12                # log2(total bytes)
_FEAT_IS_GEMM = 17              # is custom-call GEMM
_FEAT_IS_FUSABLE = 18           # is fusable

CLUSTER_FEATURE_DIM = 34


def _safe_log2(x: float) -> float:
    return math.log2(x) if x > 0 else 0.0


# ======================================================================
# Step 1: Reconstruct graph state at each step
# ======================================================================

def reconstruct_trajectory_states(
    enriched_traj: dict,
    g0: Data,
    module_key: str = "",
    trajectory_idx: int = 0,
) -> List[GraphStepState]:
    """Reconstruct GraphStepState at each fused step from enriched data.

    Uses candidate_names as ground truth for C_t, and infers V_t from
    cluster membership and G_0 node set.

    Args:
        enriched_traj: One trajectory dict from enriched .pt file
        g0: Initial PyG Data (G_0)
        module_key: Module identifier
        trajectory_idx: Index within the module's trajectories

    Returns:
        List of GraphStepState, one per fused step
    """
    all_g0_names = set(g0.node_names)
    all_clusters_raw = enriched_traj["clusters"]  # fusion.N -> {members, creation_step, ...}
    steps = enriched_traj["steps"]

    # Pre-build cluster objects from enriched data
    cluster_objects: Dict[str, FusionCluster] = {}
    for cname, cdata in all_clusters_raw.items():
        members = frozenset(cdata["members"])
        cluster_objects[cname] = FusionCluster(
            name=cname,
            members=members,
            root_consumer=cdata["root_consumer"],
            creation_step=cdata["creation_step"],
            fingerprint=cdata["fingerprint"],
            feature_status="exact" if members else "missing",
        )

    states: List[GraphStepState] = []

    for step_pos, step in enumerate(steps):
        step_idx = step["step_idx"]
        candidate_names = step["candidate_names"]
        selected_name = step.get("selected_name")

        # Determine active clusters at this step
        active_clusters, notes = _determine_active_clusters(
            cluster_objects, step_idx, candidate_names,
        )

        # Alive originals: G_0 nodes NOT absorbed by any active cluster
        all_absorbed = set()
        for cl in active_clusters.values():
            all_absorbed.update(cl.members)

        alive_originals = all_g0_names - all_absorbed

        # Validate: all candidates must be in V_t
        vt_names = alive_originals | set(active_clusters.keys())
        missing_cands = set(candidate_names) - vt_names

        fidelity = "exact"
        if missing_cands:
            fidelity = "approximate"
            for m in missing_cands:
                if m.startswith("fusion.") and m in cluster_objects:
                    active_clusters[m] = cluster_objects[m]
                    all_absorbed.update(cluster_objects[m].members)
                    alive_originals -= cluster_objects[m].members
                    notes.append(f"added_missing_cluster:{m}")
                elif m in all_g0_names:
                    alive_originals.add(m)
                    notes.append(f"readded_original:{m}")
                else:
                    # Candidate not in G_0 and not a known cluster
                    if m.startswith("fusion."):
                        active_clusters[m] = FusionCluster(
                            name=m,
                            members=frozenset(),
                            root_consumer="",
                            creation_step=step_idx,
                            fingerprint=FusionCluster.make_fingerprint(frozenset(), ""),
                            feature_status="missing",
                        )
                        notes.append(f"unknown_cluster:{m}")
                        fidelity = "unresolved"
                    else:
                        notes.append(f"unknown_candidate:{m}")
                        fidelity = "unresolved"

        # Validate selected_name is in candidates
        if selected_name and selected_name not in set(candidate_names):
            notes.append(f"selected_not_in_candidates:{selected_name}")
            fidelity = "unresolved"

        states.append(GraphStepState(
            module_key=module_key,
            trajectory_idx=trajectory_idx,
            step_idx=step_idx,
            step_position=step_pos,
            alive_original_nodes=frozenset(alive_originals),
            active_clusters=dict(active_clusters),
            candidate_names=list(candidate_names),
            selected_name=selected_name,
            graph_fidelity=fidelity,
            notes=list(notes),
        ))

    return states


def _determine_active_clusters(
    all_clusters: Dict[str, FusionCluster],
    step_idx: int,
    candidate_names: List[str],
) -> Tuple[Dict[str, FusionCluster], List[str]]:
    """Determine which clusters are active (alive) at a given step.

    A cluster is active if:
    1. It was created at or before step_idx
    2. It has NOT been absorbed by a later cluster (also created by step_idx)

    Absorption: cluster X is absorbed by cluster Y if
      X.members ⊂ Y.members AND Y.creation_step > X.creation_step
      AND Y.creation_step <= step_idx

    Returns (active_clusters, notes).
    """
    notes: List[str] = []

    # Collect clusters created by this step
    created = {
        name: cl for name, cl in all_clusters.items()
        if cl.creation_step <= step_idx
    }

    # Check absorption (smaller cluster absorbed by larger)
    absorbed: Set[str] = set()
    for name, cl in created.items():
        if name in absorbed:
            continue
        for other_name, other_cl in created.items():
            if other_name == name or other_name in absorbed:
                continue
            if (other_cl.creation_step > cl.creation_step
                    and cl.members and other_cl.members
                    and cl.members < other_cl.members):
                absorbed.add(name)
                break

    active = {
        name: cl for name, cl in created.items()
        if name not in absorbed
    }

    # Cross-validate with candidate_names
    cand_fusions = {c for c in candidate_names if c.startswith("fusion.")}
    active_names = set(active.keys())

    # Clusters in candidates but not detected as active
    for cf in cand_fusions - active_names:
        if cf in all_clusters:
            active[cf] = all_clusters[cf]
            notes.append(f"candidate_cluster_not_detected:{cf}")

    return active, notes


# ======================================================================
# Step 2: Contract graph G_0 → G_t
# ======================================================================

def contract_graph(
    g0: Data,
    state: GraphStepState,
    cluster_features_cache: Optional[Dict[str, torch.Tensor]] = None,
) -> Tuple[Data, GraphStepMetadata]:
    """Build contracted graph G_t from G_0 and GraphStepState.

    Creates a new PyG Data where:
    - Original alive nodes keep their features from G_0
    - Fusion clusters become super-nodes with aggregated features
    - Edges are rewired: internal cluster edges removed, external preserved
    - Candidate mask built from state.candidate_names

    Args:
        g0: Initial graph (G_0) with x, edge_index, node_names, opcode_ids
        state: Reconstructed graph state at step t
        cluster_features_cache: Pre-computed 34-dim features per cluster name

    Returns:
        (contracted_data, metadata) where contracted_data is a PyG Data
        and metadata is kept separate for batching safety.
    """
    g0_name_to_idx = {name: i for i, name in enumerate(g0.node_names)}
    original_dim = g0.x.shape[1]  # typically 19

    # Build ordered node list for contracted graph
    alive_sorted = sorted(state.alive_original_nodes)
    cluster_sorted = sorted(state.active_clusters.keys())
    contracted_names = alive_sorted + cluster_sorted

    num_contracted = len(contracted_names)
    name_to_new_idx = {name: i for i, name in enumerate(contracted_names)}

    # --- Build node features ---
    x_base = torch.zeros(num_contracted, original_dim, dtype=torch.float32)
    opcode_ids = torch.zeros(num_contracted, dtype=torch.long)
    cluster_feat = torch.zeros(num_contracted, CLUSTER_FEATURE_DIM, dtype=torch.float32)
    is_cluster_mask = torch.zeros(num_contracted, dtype=torch.bool)

    # Fill original node features
    for i, name in enumerate(alive_sorted):
        if name in g0_name_to_idx:
            orig_idx = g0_name_to_idx[name]
            x_base[i] = g0.x[orig_idx]
            opcode_ids[i] = g0.opcode_ids[orig_idx]

    # Fill cluster super-node features
    for j, cname in enumerate(cluster_sorted):
        new_idx = len(alive_sorted) + j
        is_cluster_mask[new_idx] = True
        cluster = state.active_clusters[cname]

        # Aggregated base features from member nodes
        x_base[new_idx] = _aggregate_member_features(
            g0, cluster.members, g0_name_to_idx, original_dim,
        )

        # Opcode of root consumer (or 0 if unknown)
        if cluster.root_consumer in g0_name_to_idx:
            opcode_ids[new_idx] = g0.opcode_ids[g0_name_to_idx[cluster.root_consumer]]

        # 34-dim cluster features
        if cluster_features_cache and cname in cluster_features_cache:
            cluster_feat[new_idx] = cluster_features_cache[cname]
        else:
            cluster_feat[new_idx] = _build_cluster_features_from_state(
                g0, cluster, g0_name_to_idx,
            )

    # --- Build contracted edges ---
    edge_src, edge_dst = _build_contracted_edges(
        g0, state, g0_name_to_idx, name_to_new_idx,
    )

    if edge_src:
        edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
    else:
        edge_index = torch.zeros(2, 0, dtype=torch.long)

    # --- Build candidate mask ---
    candidate_mask = torch.zeros(num_contracted, dtype=torch.bool)
    candidate_indices = []
    for cand in state.candidate_names:
        if cand in name_to_new_idx:
            idx = name_to_new_idx[cand]
            candidate_mask[idx] = True
            candidate_indices.append(idx)

    # Selected action index
    selected_idx = None
    if state.selected_name and state.selected_name in name_to_new_idx:
        selected_idx = name_to_new_idx[state.selected_name]

    # --- Build PyG Data ---
    data = Data(
        x=x_base,
        edge_index=edge_index,
        opcode_ids=opcode_ids,
        cluster_features=cluster_feat,
        is_cluster=is_cluster_mask,
        candidate_mask=candidate_mask,
        num_nodes=num_contracted,
    )

    # --- Build metadata (kept outside Data for batching) ---
    metadata = GraphStepMetadata(
        module_key=state.module_key,
        trajectory_idx=state.trajectory_idx,
        step_idx=state.step_idx,
        step_position=state.step_position,
        idx_to_name=contracted_names,
        name_to_idx=name_to_new_idx,
        cluster_members={
            cname: cl.members for cname, cl in state.active_clusters.items()
        },
        candidate_names=list(state.candidate_names),
        selected_name=state.selected_name,
        selected_idx=selected_idx,
        candidate_indices=candidate_indices,
        graph_fidelity=state.graph_fidelity,
        notes=list(state.notes),
        is_cluster=[is_cluster_mask[i].item() for i in range(num_contracted)],
    )

    return data, metadata


# ======================================================================
# Feature aggregation for cluster super-nodes
# ======================================================================

def _aggregate_member_features(
    g0: Data,
    members: FrozenSet[str],
    name_to_idx: Dict[str, int],
    original_dim: int,
) -> torch.Tensor:
    """Aggregate G_0 node features for cluster members into a base feature vector.

    Uses per-dimension aggregation rules matching the 19-dim feature layout:
      [0:6]  element type one-hot → mean (approximate distribution)
      [6]    rank → max
      [7:11] shape dims → max per dim
      [11]   log2(total_elements) → log2(sum(2^member_values))
      [12]   log2(total_bytes) → log2(sum(2^member_values))
      [13]   num_users → sum of external users
      [14]   num_operands → sum of external operands
      [15:17] boolean flags → max (any member has it)
      [17]   is_gemm → max
      [18]   is_fusable → 1 (cluster is always fusable)
    """
    feat = torch.zeros(original_dim, dtype=torch.float32)
    member_indices = [name_to_idx[n] for n in members if n in name_to_idx]

    if not member_indices:
        return feat

    member_x = g0.x[member_indices]  # [M, D]

    # [0:6] element type: mean
    feat[0:6] = member_x[:, 0:6].mean(dim=0)

    # [6] rank: max
    feat[6] = member_x[:, 6].max()

    # [7:11] shape dims: max per dim
    if original_dim > 10:
        feat[7:11] = member_x[:, 7:11].max(dim=0).values

    # [11] log2(total_elements): log2(sum(2^v))
    if original_dim > 11:
        vals = member_x[:, 11]
        total = sum(2.0 ** v.item() for v in vals if v.item() > 0)
        feat[11] = _safe_log2(total)

    # [12] log2(total_bytes): log2(sum(2^v))
    if original_dim > 12:
        vals = member_x[:, 12]
        total = sum(2.0 ** v.item() for v in vals if v.item() > 0)
        feat[12] = _safe_log2(total)

    # [13:15] num_users, num_operands: sum
    if original_dim > 14:
        feat[13] = member_x[:, 13].sum()
        feat[14] = member_x[:, 14].sum()

    # [15:17] boolean flags: max
    if original_dim > 16:
        feat[15:17] = member_x[:, 15:17].max(dim=0).values

    # [17] is_gemm: max (any member)
    if original_dim > 17:
        feat[17] = member_x[:, 17].max()

    # [18] is_fusable: always 1 for a cluster
    if original_dim > 18:
        feat[18] = 1.0

    return feat


def _build_cluster_features_from_state(
    g0: Data,
    cluster: FusionCluster,
    name_to_idx: Dict[str, int],
) -> torch.Tensor:
    """Build 34-dim cluster features from G_0 and cluster info.

    Simplified version of cluster_features.compute_cluster_features(),
    using only information available from G_0 and the cluster definition.
    """
    feat = torch.zeros(CLUSTER_FEATURE_DIM, dtype=torch.float32)
    member_indices = [name_to_idx[n] for n in cluster.members if n in name_to_idx]

    feat[0] = _safe_log2(len(cluster.members))

    if not member_indices:
        return feat

    member_x = g0.x[member_indices]
    M = len(member_indices)

    # [11] log2(total_bytes)
    total_bytes = sum(2.0 ** b for b in member_x[:, _FEAT_BYTES].tolist() if b > 0)
    feat[11] = _safe_log2(total_bytes)

    # [12] log2(total_elements)
    total_elems = sum(2.0 ** e for e in member_x[:, _FEAT_ELEMENTS].tolist() if e > 0)
    feat[12] = _safe_log2(total_elems)

    # [13] max_rank
    feat[13] = member_x[:, _FEAT_RANK].max().item()

    # [14-15] fan-in / fan-out from G_0
    member_set = set(member_indices)
    edge_src = g0.edge_index[0].tolist()
    edge_dst = g0.edge_index[1].tolist()
    fan_in = sum(1 for s, d in zip(edge_src, edge_dst)
                 if s not in member_set and d in member_set)
    fan_out = sum(1 for s, d in zip(edge_src, edge_dst)
                  if s in member_set and d not in member_set)
    feat[14] = _safe_log2(fan_in + 1)
    feat[15] = _safe_log2(fan_out + 1)

    # [16] contains_gemm
    feat[16] = float(member_x[:, _FEAT_IS_GEMM].max().item() > 0.5)

    # [17-20] binary flags (approximate without opcode names)
    # Just use is_fusable as a proxy
    feat[20] = 1.0  # cluster is always elementwise-heavy in practice

    # [21] root_opcode_id (normalized)
    root_idx = name_to_idx.get(cluster.root_consumer, -1)
    if root_idx >= 0:
        feat[21] = g0.opcode_ids[root_idx].item() / 132.0

    # [22-23] topo span and mean position (from node indices as proxy)
    indices = torch.tensor(member_indices, dtype=torch.float32)
    indices_norm = indices / max(g0.num_nodes, 1)
    feat[22] = (indices_norm.max() - indices_norm.min()).item()
    feat[23] = indices_norm.mean().item()

    # [24:30] element type histogram
    etype_hist = member_x[:, _FEAT_ETYPE].sum(dim=0)
    total = etype_hist.sum()
    if total > 0:
        etype_hist /= total
    feat[24:30] = etype_hist

    # [30:33] fusion kind (default kLoop)
    feat[30] = 1.0

    return feat


# ======================================================================
# Edge rewiring for contracted graph
# ======================================================================

def _build_contracted_edges(
    g0: Data,
    state: GraphStepState,
    g0_name_to_idx: Dict[str, int],
    contracted_name_to_idx: Dict[str, int],
) -> Tuple[List[int], List[int]]:
    """Build edges for the contracted graph.

    Rules:
    - For each edge (u, v) in G_0:
      - Map u to its contracted node (itself if alive, or its cluster)
      - Map v to its contracted node
      - If both are in the contracted graph and not the same node, add edge
    - Deduplicate edges
    """
    # Build original-node -> contracted-node mapping
    orig_to_contracted: Dict[str, str] = {}

    # Alive originals map to themselves
    for name in state.alive_original_nodes:
        orig_to_contracted[name] = name

    # Members of clusters map to their cluster
    for cname, cluster in state.active_clusters.items():
        for member in cluster.members:
            orig_to_contracted[member] = cname

    # Build edge set (deduplicated)
    edge_set: Set[Tuple[int, int]] = set()

    for i in range(g0.edge_index.shape[1]):
        src_name = g0.node_names[g0.edge_index[0, i].item()]
        dst_name = g0.node_names[g0.edge_index[1, i].item()]

        src_contracted = orig_to_contracted.get(src_name)
        dst_contracted = orig_to_contracted.get(dst_name)

        if src_contracted is None or dst_contracted is None:
            continue
        if src_contracted == dst_contracted:
            continue  # internal edge within same cluster

        src_idx = contracted_name_to_idx.get(src_contracted)
        dst_idx = contracted_name_to_idx.get(dst_contracted)

        if src_idx is not None and dst_idx is not None:
            edge_set.add((src_idx, dst_idx))

    if not edge_set:
        return [], []

    edges = sorted(edge_set)
    return [e[0] for e in edges], [e[1] for e in edges]


# ======================================================================
# Validation
# ======================================================================

def validate_contracted_graph(
    data: Data,
    metadata: GraphStepMetadata,
    report: Optional[FidelityReport] = None,
) -> Dict[str, object]:
    """Run fidelity checks on a contracted graph.

    Returns a dict of check results. Optionally accumulates into report.
    """
    checks = {}

    # 1. All candidates are in the graph
    n_candidates = len(metadata.candidate_names)
    n_cands_in_graph = sum(
        1 for c in metadata.candidate_names if c in metadata.name_to_idx
    )
    checks["candidates_in_graph"] = n_cands_in_graph == n_candidates
    checks["candidates_total"] = n_candidates
    checks["candidates_found"] = n_cands_in_graph

    # 2. Selected action maps to valid index
    if metadata.selected_name:
        checks["selected_in_graph"] = metadata.selected_name in metadata.name_to_idx
        checks["selected_in_candidates"] = metadata.selected_name in set(
            metadata.candidate_names
        )
    else:
        checks["selected_in_graph"] = True
        checks["selected_in_candidates"] = True

    # 3. Cluster nodes have non-empty members
    n_clusters = sum(1 for ic in metadata.is_cluster if ic)
    n_clusters_with_members = sum(
        1 for name, members in metadata.cluster_members.items()
        if members
    )
    checks["clusters_total"] = n_clusters
    checks["clusters_with_members"] = n_clusters_with_members

    # 4. Graph has valid structure
    checks["num_nodes"] = data.num_nodes
    checks["num_edges"] = data.edge_index.shape[1] if data.edge_index.numel() > 0 else 0
    checks["has_valid_structure"] = (
        data.x.shape[0] == data.num_nodes
        and data.opcode_ids.shape[0] == data.num_nodes
    )

    # 5. Candidate mask consistency
    mask_count = data.candidate_mask.sum().item()
    checks["candidate_mask_count"] = mask_count
    checks["candidate_mask_matches"] = mask_count == n_candidates

    # 6. Fidelity status
    checks["graph_fidelity"] = metadata.graph_fidelity

    # Accumulate into report
    if report is not None:
        report.total_steps += 1
        if metadata.graph_fidelity == "exact":
            report.exact_steps += 1
        elif metadata.graph_fidelity == "approximate":
            report.approximate_steps += 1
        else:
            report.unresolved_steps += 1

        report.candidates_in_vt += n_cands_in_graph
        report.candidates_missing += (n_candidates - n_cands_in_graph)

        if metadata.selected_name:
            if checks["selected_in_candidates"]:
                report.selected_in_ct += 1
            else:
                report.selected_missing += 1

        report.cluster_with_members += n_clusters_with_members
        report.cluster_without_members += (n_clusters - n_clusters_with_members)

    return checks


# ======================================================================
# Batch reconstruction: full trajectory
# ======================================================================

def reconstruct_and_contract_trajectory(
    enriched_traj: dict,
    g0: Data,
    module_key: str = "",
    trajectory_idx: int = 0,
    cluster_features_cache: Optional[Dict[str, torch.Tensor]] = None,
    report: Optional[FidelityReport] = None,
) -> List[Tuple[Data, GraphStepMetadata]]:
    """Reconstruct and contract all steps in one trajectory.

    This is the main entry point for building training data.

    Args:
        enriched_traj: One trajectory from enriched .pt
        g0: Initial graph G_0
        module_key: Module identifier
        trajectory_idx: Trajectory index within module
        cluster_features_cache: Pre-computed features per cluster
        report: Optional report to accumulate fidelity stats

    Returns:
        List of (contracted_data, metadata) for each fused step
    """
    # Build cluster features cache from enriched data if not provided
    if cluster_features_cache is None:
        cluster_features_cache = {}
        for cname, cdata in enriched_traj.get("clusters", {}).items():
            if "features" in cdata and isinstance(cdata["features"], torch.Tensor):
                cluster_features_cache[cname] = cdata["features"]

    states = reconstruct_trajectory_states(
        enriched_traj, g0, module_key, trajectory_idx,
    )

    results = []
    for state in states:
        data, metadata = contract_graph(g0, state, cluster_features_cache)

        if report is not None:
            validate_contracted_graph(data, metadata, report)

        results.append((data, metadata))

    return results


# ======================================================================
# Report generation
# ======================================================================

def generate_fidelity_report_for_module(
    enriched_path: str,
    max_trajectories: int = 0,
) -> Tuple[FidelityReport, List[dict]]:
    """Generate fidelity report for one enriched .pt file.

    Args:
        enriched_path: Path to enriched .pt file
        max_trajectories: Limit trajectories (0 = all)

    Returns:
        (FidelityReport, list of per-trajectory stats)
    """
    data = torch.load(enriched_path, map_location="cpu", weights_only=False)
    g0 = data["graph"]
    module_key = data.get("module_key", "")
    trajectories = data["enriched_trajectories"]

    if max_trajectories > 0:
        trajectories = trajectories[:max_trajectories]

    report = FidelityReport()
    per_traj_stats = []

    for traj_idx, traj in enumerate(trajectories):
        traj_report = FidelityReport()

        cluster_cache = {}
        for cname, cdata in traj.get("clusters", {}).items():
            if "features" in cdata and isinstance(cdata["features"], torch.Tensor):
                cluster_cache[cname] = cdata["features"]

        results = reconstruct_and_contract_trajectory(
            traj, g0, module_key, traj_idx, cluster_cache, traj_report,
        )

        # Aggregate into module report
        report.total_steps += traj_report.total_steps
        report.exact_steps += traj_report.exact_steps
        report.approximate_steps += traj_report.approximate_steps
        report.unresolved_steps += traj_report.unresolved_steps
        report.candidates_in_vt += traj_report.candidates_in_vt
        report.candidates_missing += traj_report.candidates_missing
        report.selected_in_ct += traj_report.selected_in_ct
        report.selected_missing += traj_report.selected_missing
        report.cluster_with_members += traj_report.cluster_with_members
        report.cluster_without_members += traj_report.cluster_without_members

        per_traj_stats.append({
            "trajectory_idx": traj_idx,
            "strategy": traj.get("strategy", ""),
            "total_steps": traj_report.total_steps,
            "exact_steps": traj_report.exact_steps,
            "approximate_steps": traj_report.approximate_steps,
            "unresolved_steps": traj_report.unresolved_steps,
            "num_contracted_graphs": len(results),
            "mean_graph_size": (
                sum(d.num_nodes for d, _ in results) / len(results)
                if results else 0
            ),
        })

    return report, per_traj_stats
