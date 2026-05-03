"""Extract dynamic fusion-cluster features from XLA PriorityFusion dumps.

Reconstructs fusion cluster membership, tracks candidate sets through the
fusion loop, and computes per-candidate feature vectors for training a
dynamic priority scorer.

Key concepts:
- "Fusion cluster": a fusion.N instruction created during PriorityFusion,
  represented as the set of original HLO nodes it absorbed.
- "Alive set": approximate set of producers available for dequeuing at each
  decision step (see documented limitations in track_alive_sets).
- "Cluster features": 33-dim static feature vector per cluster, derived from
  aggregate properties of member nodes. A 34th dim (steps_since_creation) is
  time-varying and should be set at training time.

Usage:
    # Inspect a single dump (debugging)
    python cluster_features.py inspect \\
        --dump output/multi_dumps/Llama-3-8B/gqa_layer/xla_default/priority_fusion_dump.txt \\
        --traj-pt output/multi_trajectories_merged/Llama-3-8B__gqa_layer.pt

    # Process one module (all its strategies)
    python cluster_features.py single \\
        --traj-pt output/multi_trajectories_merged/Llama-3-8B__gqa_layer.pt \\
        --dump-base output/multi_dumps/Llama-3-8B/gqa_layer

    # Batch process all modules
    python cluster_features.py batch \\
        --traj-dir output/multi_trajectories_merged \\
        --dump-root output/multi_dumps \\
        --output-dir output/dynamic_features
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import torch

# Add current directory for dump_parser
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dump_parser import (
    FusionEvent, UpdatePriorityEvent, IneligibleEvent, MdpStep,
    ParsedDump, parse_dump,
)


# ======================================================================
# Constants — feature layout
# ======================================================================

# Indices into data.x (19 or 27-dim continuous features from feature_encoder.py)
# The first 19 dims are always present; dims 19-26 may be absent in older .pt files
_FEAT_ETYPE = slice(0, 6)       # one-hot element type (6 dims)
_FEAT_RANK = 6                  # tensor rank
_FEAT_ELEMENTS = 11             # log2(total elements)
_FEAT_BYTES = 12                # log2(total bytes)
_FEAT_IS_GEMM = 17              # is custom-call GEMM
_FEAT_IS_FUSABLE = 18           # is fusable
_FEAT_TOPO_POS = 19             # normalized topo position (0-1), may not exist

# Cluster feature vector layout (34 dims total):
#   [0]      log2(cluster_size)
#   [1:11]   opcode histogram (10 bins, normalized)
#   [11]     log2(total_bytes)
#   [12]     log2(total_elements)
#   [13]     max_rank
#   [14]     log2(cluster_fan_in + 1)
#   [15]     log2(cluster_fan_out + 1)
#   [16]     contains_gemm
#   [17]     contains_reduce
#   [18]     contains_broadcast
#   [19]     contains_bitcast
#   [20]     contains_elementwise (majority)
#   [21]     root_opcode_id (normalized)
#   [22]     topo_span
#   [23]     mean_topo_pos
#   [24:30]  element_type_hist (6 bins, normalized)
#   [30:33]  fusion_kind one-hot (default kLoop)
#   [33]     steps_since_creation (set to 0; recomputed at training time)
CLUSTER_FEATURE_DIM = 34

# Opcode histogram bins for cluster features
_OPCODE_HIST_NAMES = [
    'add', 'multiply', 'convert', 'broadcast', 'reduce',
    'bitcast', 'compare', 'select', 'fusion',
]
_N_OPCODE_HIST_BINS = len(_OPCODE_HIST_NAMES) + 1  # +1 for "other"

# Elementwise opcodes for contains_elementwise
_ELEMENTWISE_OPCODES = frozenset({
    'add', 'subtract', 'multiply', 'divide', 'maximum', 'minimum',
    'negate', 'abs', 'sign', 'convert', 'bitcast-convert',
    'compare', 'select', 'and', 'or', 'xor', 'not',
    'exponential', 'log', 'tanh', 'logistic', 'sqrt', 'rsqrt',
    'sine', 'cosine', 'power', 'clamp', 'floor', 'ceil',
})


# ======================================================================
# Data classes
# ======================================================================

@dataclass
class FusionCluster:
    """A dynamically-created fusion.N cluster and its original members."""
    fusion_name: str
    members: Set[str]       # original HLO node names (recursively resolved)
    creation_step: int      # MDP step index where cluster first appeared
    root_consumer: str      # first consumer (traced to original node name)


# ======================================================================
# Helpers
# ======================================================================

def _safe_log2(x: float) -> float:
    return math.log2(x) if x > 0 else 0.0


def _get_opcode_names() -> List[str]:
    """Load OPCODE_VOCAB from feature_encoder (or return minimal fallback)."""
    try:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        sys.path.insert(0, os.path.join(project_root, "references", "GO_fusion"))
        from src.hlo_parser.feature_encoder import OPCODE_VOCAB
        return OPCODE_VOCAB
    except ImportError:
        pass
    # Fallback: try PYTHONPATH
    try:
        from src.hlo_parser.feature_encoder import OPCODE_VOCAB
        return OPCODE_VOCAB
    except ImportError:
        return ['unknown'] * 132


# Module-level cache
_OPCODE_VOCAB: Optional[List[str]] = None


def _opcode_vocab() -> List[str]:
    global _OPCODE_VOCAB
    if _OPCODE_VOCAB is None:
        _OPCODE_VOCAB = _get_opcode_names()
    return _OPCODE_VOCAB


def _find_dump_file(dump_dir: Path) -> Optional[Path]:
    """Find priority_fusion_dump.txt in a dump directory."""
    direct = dump_dir / "priority_fusion_dump.txt"
    if direct.exists():
        return direct
    candidates = list(dump_dir.glob("*priority_fusion_dump.txt"))
    return candidates[0] if candidates else None


def _get_fusible_nodes(data) -> Set[str]:
    """Get fusible original node names from graph data."""
    fusible = set()
    for idx in range(data.num_nodes):
        if data.x[idx, _FEAT_IS_FUSABLE].item() > 0.5:
            fusible.add(data.node_names[idx])
    return fusible


def _infer_strategy(dirname: str) -> str:
    """Infer strategy name from dump directory name."""
    if dirname == "xla_default":
        return "xla_default"
    if dirname.startswith("random_"):
        return "random"
    if dirname.startswith("perturbed_"):
        return "perturbed"
    if dirname.startswith("greedy_"):
        return dirname.replace("greedy_", "")
    if dirname.startswith("sa_"):
        return "simulated_annealing"
    return dirname


# ======================================================================
# Step 1A: Reconstruct fusion cluster membership
# ======================================================================

def reconstruct_clusters(parsed: ParsedDump) -> Dict[str, FusionCluster]:
    """Walk fused MDP steps to build fusion cluster membership.

    Each cluster maps a fusion.N name to its set of original HLO member
    node names, resolved recursively when a producer or consumer is itself
    a fusion.
    """
    clusters: Dict[str, FusionCluster] = {}

    def resolve(name: str) -> Set[str]:
        if name in clusters:
            return set(clusters[name].members)
        return {name}

    def trace_root(name: str) -> str:
        """Follow root_consumer chain to an original node."""
        visited: Set[str] = set()
        while name in clusters and name not in visited:
            visited.add(name)
            name = clusters[name].root_consumer
        return name

    for step in parsed.steps:
        if not step.was_fused or step.fusion_name is None:
            continue

        F = step.fusion_name
        P = step.producer_name
        consumers = step.consumer_names

        p_members = resolve(P)
        c_members: Set[str] = set()
        for c in consumers:
            c_members |= resolve(c)

        if F in clusters:
            clusters[F].members |= p_members | c_members
        else:
            root = consumers[0] if consumers else ''
            root = trace_root(root) if root in clusters else root
            clusters[F] = FusionCluster(
                fusion_name=F,
                members=p_members | c_members,
                creation_step=step.step_idx,
                root_consumer=root,
            )

    return clusters


# ======================================================================
# Step 1B: Track alive (candidate) set per step
# ======================================================================

def track_alive_sets(
    parsed: ParsedDump,
    fusible_originals: Set[str],
) -> List[Set[str]]:
    """Track alive producer set before each MDP step.

    Returns a list where entry i is the candidate set BEFORE step i.

    Limitations:
    - Queue removals for zero-user nodes are not logged in the dump.
    - Priority values are not tracked; only membership.
    - Some consumers may remain in the alive set incorrectly if they
      were not fusible producers to begin with.
    """
    alive = set(fusible_originals)
    alive_sets: List[Set[str]] = []

    for step in parsed.steps:
        alive_sets.append(frozenset(alive))  # snapshot before action

        P = step.producer_name
        alive.discard(P)

        if step.was_fused and step.fusion_name is not None:
            F = step.fusion_name
            for c in step.consumer_names:
                if c != F:
                    alive.discard(c)
            alive.add(F)

    return alive_sets


# ======================================================================
# Step 1C: Compute cluster features
# ======================================================================

def compute_cluster_features(
    cluster: FusionCluster,
    data,
    name_to_idx: Dict[str, int],
    opcode_names: List[str],
) -> torch.Tensor:
    """Compute 34-dim feature vector for a fusion cluster.

    Index 33 (steps_since_creation) is set to 0 here; it should be
    overwritten at training time with (current_step - creation_step).
    """
    feat = torch.zeros(CLUSTER_FEATURE_DIM, dtype=torch.float32)

    member_indices = [name_to_idx[n] for n in cluster.members if n in name_to_idx]
    if not member_indices:
        feat[0] = _safe_log2(len(cluster.members))
        return feat

    member_x = data.x[member_indices]           # [M, 27]
    member_opcodes = data.opcode_ids[member_indices]  # [M]
    M = len(member_indices)

    # --- [0] log2(cluster_size) ---
    feat[0] = _safe_log2(len(cluster.members))

    # --- [1:11] opcode histogram (normalized) ---
    opcode_counts = torch.zeros(_N_OPCODE_HIST_BINS)
    for op_idx in member_opcodes.tolist():
        name = opcode_names[op_idx] if op_idx < len(opcode_names) else ''
        try:
            bin_idx = _OPCODE_HIST_NAMES.index(name)
        except ValueError:
            bin_idx = _N_OPCODE_HIST_BINS - 1  # other
        opcode_counts[bin_idx] += 1
    opcode_counts /= max(M, 1)
    feat[1:11] = opcode_counts

    # --- [11] log2(total_bytes) ---
    total_bytes = sum(2.0 ** b for b in member_x[:, _FEAT_BYTES].tolist() if b > 0)
    feat[11] = _safe_log2(total_bytes)

    # --- [12] log2(total_elements) ---
    total_elems = sum(2.0 ** e for e in member_x[:, _FEAT_ELEMENTS].tolist() if e > 0)
    feat[12] = _safe_log2(total_elems)

    # --- [13] max_rank ---
    feat[13] = member_x[:, _FEAT_RANK].max().item()

    # --- [14-15] cluster fan-in / fan-out (log scale) ---
    member_set = set(member_indices)
    edge_src = data.edge_index[0].tolist()
    edge_dst = data.edge_index[1].tolist()
    fan_in = sum(1 for s, d in zip(edge_src, edge_dst)
                 if s not in member_set and d in member_set)
    fan_out = sum(1 for s, d in zip(edge_src, edge_dst)
                  if s in member_set and d not in member_set)
    feat[14] = _safe_log2(fan_in + 1)
    feat[15] = _safe_log2(fan_out + 1)

    # --- [16-20] contains_* binary ---
    feat[16] = float(member_x[:, _FEAT_IS_GEMM].max().item() > 0.5)

    op_name_list = [
        opcode_names[i] if i < len(opcode_names) else ''
        for i in member_opcodes.tolist()
    ]
    op_name_set = set(op_name_list)
    feat[17] = float('reduce' in op_name_set)
    feat[18] = float('broadcast' in op_name_set)
    feat[19] = float('bitcast' in op_name_set)

    ew_count = sum(1 for n in op_name_list if n in _ELEMENTWISE_OPCODES)
    feat[20] = float(ew_count > M / 2)

    # --- [21] root_opcode_id (normalized) ---
    root_idx = name_to_idx.get(cluster.root_consumer, -1)
    root_opcode = data.opcode_ids[root_idx].item() if root_idx >= 0 else 0
    feat[21] = root_opcode / 132.0

    # --- [22] topo_span ---
    feat_dim = data.x.shape[1]
    if feat_dim > _FEAT_TOPO_POS:
        topo = member_x[:, _FEAT_TOPO_POS]
        feat[22] = (topo.max() - topo.min()).item()
        feat[23] = topo.mean().item()
    else:
        # topo_pos not in features — use normalized index as proxy
        indices = torch.tensor(member_indices, dtype=torch.float32)
        indices /= max(data.num_nodes, 1)
        feat[22] = (indices.max() - indices.min()).item()
        feat[23] = indices.mean().item()

    # --- [23] mean_topo_pos (handled above) ---

    # --- [24:30] element_type histogram (normalized) ---
    etype_hist = member_x[:, _FEAT_ETYPE].sum(dim=0)  # [6]
    total = etype_hist.sum()
    if total > 0:
        etype_hist /= total
    feat[24:30] = etype_hist

    # --- [30:33] fusion_kind one-hot (default kLoop) ---
    feat[30] = 1.0  # kLoop

    # --- [33] steps_since_creation (placeholder — set at training time) ---
    feat[33] = 0.0

    return feat


# ======================================================================
# Step 1D: Cluster fingerprint
# ======================================================================

def cluster_fingerprint(cluster: FusionCluster) -> str:
    """Stable cross-trajectory identifier based on member set + root."""
    key = '|'.join(sorted(cluster.members)) + '||' + cluster.root_consumer
    return hashlib.md5(key.encode()).hexdigest()[:12]


# ======================================================================
# Step 1E-F: Extract enriched trajectory from one dump
# ======================================================================

def extract_enriched_trajectory(
    dump_path: Path,
    data,
    name_to_idx: Dict[str, int],
    fusible_originals: Set[str],
    opcode_names: List[str],
    strategy_detail: str = '',
    reward_gpu: Optional[float] = None,
) -> dict:
    """Process one dump into an enriched trajectory with candidate sets.

    Returns a dict with cluster info, per-step candidate sets, and stats.
    This is the unit of output stored per strategy.
    """
    parsed = parse_dump(str(dump_path))

    clusters = reconstruct_clusters(parsed)
    alive_sets = track_alive_sets(parsed, fusible_originals)

    # Compute static cluster features
    cluster_data: Dict[str, dict] = {}
    for fname, cl in clusters.items():
        cluster_data[fname] = {
            'members': sorted(cl.members),
            'creation_step': cl.creation_step,
            'root_consumer': cl.root_consumer,
            'fingerprint': cluster_fingerprint(cl),
            'features': compute_cluster_features(
                cl, data, name_to_idx, opcode_names,
            ),
        }

    # Build enriched steps (fused only)
    enriched_steps: List[dict] = []
    candidate_set_sizes: List[int] = []
    n_original_sel = 0
    n_fusion_sel = 0
    n_not_in_alive = 0

    for step in parsed.steps:
        if not step.was_fused:
            continue

        candidates = set(alive_sets[step.step_idx]) if step.step_idx < len(alive_sets) else set()
        selected = step.producer_name

        if selected not in candidates:
            n_not_in_alive += 1
            candidates = candidates | {selected}

        cand_list = sorted(candidates)
        sel_idx = cand_list.index(selected)

        if selected.startswith('fusion.'):
            n_fusion_sel += 1
        else:
            n_original_sel += 1

        enriched_steps.append({
            'step_idx': step.step_idx,
            'selected_name': selected,
            'selected_idx': sel_idx,
            'candidate_names': cand_list,
            'us_fused': step.us_fused,
            'us_unfused': step.us_unfused,
        })
        candidate_set_sizes.append(len(cand_list))

    return {
        'strategy': _infer_strategy(strategy_detail),
        'strategy_detail': strategy_detail,
        'reward_gpu': reward_gpu,
        'clusters': cluster_data,
        'steps': enriched_steps,
        'candidate_set_sizes': candidate_set_sizes,
        'stats': {
            'n_fused_steps': len(enriched_steps),
            'n_total_steps': len(parsed.steps),
            'n_clusters': len(clusters),
            'n_original_selected': n_original_sel,
            'n_fusion_selected': n_fusion_sel,
            'n_selected_not_in_alive': n_not_in_alive,
            'mean_candidate_set_size': (
                sum(candidate_set_sizes) / len(candidate_set_sizes)
                if candidate_set_sizes else 0
            ),
        },
    }


# ======================================================================
# Module processing
# ======================================================================

def process_module(
    traj_pt_path: str,
    dump_base: str,
    output_path: str,
    gpu_csv_path: Optional[str] = None,
    verbose: bool = True,
) -> dict:
    """Process all strategy dumps for one module.

    Loads the initial graph from traj_pt_path, scans dump_base for
    per-strategy dumps, extracts enriched trajectories, and saves.

    Returns stats dict.
    """
    traj_pt_path = Path(traj_pt_path)
    dump_base_p = Path(dump_base)
    output_path_p = Path(output_path)
    output_path_p.parent.mkdir(parents=True, exist_ok=True)

    module_key = traj_pt_path.stem
    if verbose:
        print(f"\n{'='*60}")
        print(f"Processing: {module_key}")
        print(f"{'='*60}")

    # Load graph from .pt
    pt_data = torch.load(str(traj_pt_path), weights_only=False)
    graph = pt_data['initial_graph']
    name_to_idx = {name: idx for idx, name in enumerate(graph.node_names)}
    fusible_originals = _get_fusible_nodes(graph)
    opcode_names = _opcode_vocab()

    if verbose:
        print(f"  Graph: {graph.num_nodes} nodes, "
              f"{graph.edge_index.shape[1]} edges, "
              f"{len(fusible_originals)} fusible")

    # Reward lookup from .pt trajectories
    reward_map: Dict[str, float] = {}
    for traj in pt_data.get('trajectories', []):
        detail = traj.get('strategy_detail', '')
        r = traj.get('reward_real_gpu') or traj.get('reward_gpu')
        if r is not None:
            reward_map[detail] = r

    # Optional GPU CSV rewards
    if gpu_csv_path:
        _load_gpu_csv_rewards(gpu_csv_path, module_key, reward_map)

    # Find dump directories
    if not dump_base_p.exists():
        msg = f"dump_base not found: {dump_base_p}"
        if verbose:
            print(f"  ERROR: {msg}")
        return {'module_key': module_key, 'n_trajectories': 0, 'error': msg}

    dump_dirs = sorted([
        d for d in dump_base_p.iterdir()
        if d.is_dir() and not d.name.startswith('sa_work')
    ])
    if verbose:
        print(f"  Dump directories: {len(dump_dirs)}")

    # Process each strategy
    enriched_trajectories: List[dict] = []
    errors: List[str] = []

    for dump_dir in dump_dirs:
        dump_file = _find_dump_file(dump_dir)
        if not dump_file:
            continue

        try:
            result = extract_enriched_trajectory(
                dump_file, graph, name_to_idx, fusible_originals,
                opcode_names,
                strategy_detail=dump_dir.name,
                reward_gpu=reward_map.get(dump_dir.name),
            )
            enriched_trajectories.append(result)

            if verbose:
                s = result['stats']
                print(f"  {dump_dir.name}: {s['n_fused_steps']} steps, "
                      f"{s['n_clusters']} clusters, "
                      f"fusion_sel={s['n_fusion_selected']}")
        except Exception as e:
            errors.append(f"{dump_dir.name}: {e}")
            if verbose:
                print(f"  ERROR {dump_dir.name}: {e}")

    if verbose and errors:
        print(f"\n  Errors ({len(errors)}):")
        for err in errors[:5]:
            print(f"    {err}")
        if len(errors) > 5:
            print(f"    ... and {len(errors) - 5} more")

    stats = _compute_module_stats(enriched_trajectories, module_key)

    save_data = {
        'module_key': module_key,
        'graph': graph,
        'original_feature_dim': graph.x.shape[1],
        'cluster_feature_dim': CLUSTER_FEATURE_DIM,
        'enriched_trajectories': enriched_trajectories,
        'stats': stats,
    }
    torch.save(save_data, str(output_path_p))

    if verbose:
        print(f"\n  Saved {len(enriched_trajectories)} trajectories "
              f"to {output_path_p}")
        _print_stats(stats)

    return stats


def _load_gpu_csv_rewards(
    csv_path: str,
    module_key: str,
    reward_map: Dict[str, float],
):
    """Load GPU kernel times from CSV, compute rewards, merge into map."""
    try:
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            xla_default_us = None
            strategy_times: Dict[str, float] = {}

            for row in reader:
                row_module = row.get('module', '')
                if row_module not in module_key and module_key not in row_module:
                    continue

                detail = row.get('strategy_detail', row.get('strategy', ''))
                kernel_us = float(row.get('kernel_time_us', 0))

                if detail == 'xla_default':
                    xla_default_us = kernel_us
                strategy_times[detail] = kernel_us

            if xla_default_us and xla_default_us > 0:
                for detail, us in strategy_times.items():
                    if detail not in reward_map and us > 0:
                        reward_map[detail] = xla_default_us / us
    except Exception:
        pass


def _compute_module_stats(
    trajectories: List[dict], module_key: str = '',
) -> dict:
    """Compute aggregate statistics for a module."""
    if not trajectories:
        return {'module_key': module_key, 'n_trajectories': 0}

    total_steps = sum(t['stats']['n_fused_steps'] for t in trajectories)
    total_orig = sum(t['stats']['n_original_selected'] for t in trajectories)
    total_fus = sum(t['stats']['n_fusion_selected'] for t in trajectories)
    total_miss = sum(t['stats']['n_selected_not_in_alive'] for t in trajectories)

    all_cluster_sizes = [
        len(c['members'])
        for t in trajectories for c in t['clusters'].values()
    ]
    all_cand_sizes = [
        sz for t in trajectories for sz in t['candidate_set_sizes']
    ]

    return {
        'module_key': module_key,
        'n_trajectories': len(trajectories),
        'n_total_steps': total_steps,
        'n_original_selected': total_orig,
        'n_fusion_selected': total_fus,
        'fusion_selection_pct': (
            100.0 * total_fus / total_steps if total_steps > 0 else 0
        ),
        'n_selected_not_in_alive': total_miss,
        'alive_miss_pct': (
            100.0 * total_miss / total_steps if total_steps > 0 else 0
        ),
        'mean_cluster_size': (
            sum(all_cluster_sizes) / len(all_cluster_sizes)
            if all_cluster_sizes else 0
        ),
        'max_cluster_size': max(all_cluster_sizes) if all_cluster_sizes else 0,
        'mean_candidate_set_size': (
            sum(all_cand_sizes) / len(all_cand_sizes)
            if all_cand_sizes else 0
        ),
        'median_candidate_set_size': (
            sorted(all_cand_sizes)[len(all_cand_sizes) // 2]
            if all_cand_sizes else 0
        ),
    }


def _print_stats(stats: dict):
    """Print formatted module statistics."""
    print(f"\n  --- Module Stats ---")
    print(f"  Trajectories: {stats['n_trajectories']}")
    if stats['n_trajectories'] == 0:
        return
    print(f"  Total fused steps: {stats['n_total_steps']}")
    n_o, n_f = stats['n_original_selected'], stats['n_fusion_selected']
    print(f"  Selections: {n_o} original, {n_f} fusion.N "
          f"({stats['fusion_selection_pct']:.1f}%)")
    miss = stats['n_selected_not_in_alive']
    if miss > 0:
        print(f"  Alive-set misses: {miss} ({stats['alive_miss_pct']:.1f}%)")
    print(f"  Clusters: mean size {stats['mean_cluster_size']:.1f}, "
          f"max {stats['max_cluster_size']}")
    print(f"  Candidate sets: mean {stats['mean_candidate_set_size']:.1f}, "
          f"median {stats['median_candidate_set_size']}")


# ======================================================================
# Batch processing + reporting
# ======================================================================

def batch_process(
    traj_dir: str,
    dump_root: str,
    output_dir: str,
    gpu_csv: Optional[str] = None,
):
    """Process all modules in batch."""
    traj_dir_p = Path(traj_dir)
    dump_root_p = Path(dump_root)
    output_dir_p = Path(output_dir)
    output_dir_p.mkdir(parents=True, exist_ok=True)

    pt_files = sorted(traj_dir_p.glob("*.pt"))
    print(f"Found {len(pt_files)} trajectory files in {traj_dir_p}")

    all_stats: List[dict] = []

    for pt_file in pt_files:
        module_key = pt_file.stem
        parts = module_key.split('__', 1)
        if len(parts) != 2:
            print(f"SKIP {module_key}: can't parse Model__module")
            continue

        model_name, module_name = parts
        dump_base = dump_root_p / model_name / module_name
        output_path = output_dir_p / f"{module_key}.pt"

        if not dump_base.exists():
            print(f"SKIP {module_key}: no dumps at {dump_base}")
            continue

        stats = process_module(
            traj_pt_path=str(pt_file),
            dump_base=str(dump_base),
            output_path=str(output_path),
            gpu_csv_path=gpu_csv,
        )
        all_stats.append(stats)

    _write_summary_csv(all_stats, output_dir_p / "summary.csv")
    _write_report(all_stats, output_dir_p / "report.md")

    print(f"\n{'='*60}")
    print(f"Batch complete: {len(all_stats)} modules processed")
    print(f"Output: {output_dir_p}")


def _write_summary_csv(stats_list: List[dict], path: Path):
    """Write per-module summary CSV."""
    if not stats_list:
        return

    fields = [
        'module_key', 'n_trajectories', 'n_total_steps',
        'n_original_selected', 'n_fusion_selected', 'fusion_selection_pct',
        'n_selected_not_in_alive', 'alive_miss_pct',
        'mean_cluster_size', 'max_cluster_size',
        'mean_candidate_set_size', 'median_candidate_set_size',
    ]
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        for s in stats_list:
            writer.writerow(s)

    print(f"Wrote summary: {path}")


def _write_report(stats_list: List[dict], path: Path):
    """Write aggregate report in markdown."""
    if not stats_list:
        return

    total_trajs = sum(s.get('n_trajectories', 0) for s in stats_list)
    total_steps = sum(s.get('n_total_steps', 0) for s in stats_list)
    total_orig = sum(s.get('n_original_selected', 0) for s in stats_list)
    total_fus = sum(s.get('n_fusion_selected', 0) for s in stats_list)
    total_miss = sum(s.get('n_selected_not_in_alive', 0) for s in stats_list)

    lines = [
        "# Dynamic Feature Extraction Report",
        "",
        f"**Modules processed**: {len(stats_list)}",
        f"**Total trajectories**: {total_trajs}",
        f"**Total fused decision steps**: {total_steps}",
        "",
        "## Selection Breakdown",
        "",
        "| Category | Count | % |",
        "|---|---|---|",
    ]
    if total_steps > 0:
        lines.append(f"| Original node | {total_orig} | "
                     f"{100*total_orig/total_steps:.1f}% |")
        lines.append(f"| Fusion.N cluster | {total_fus} | "
                     f"{100*total_fus/total_steps:.1f}% |")
    lines.append("")
    if total_steps > 0 and total_miss > 0:
        lines.append(
            f"**Alive-set reconstruction misses**: {total_miss} "
            f"({100*total_miss/total_steps:.2f}% of steps)")
        lines.append("")

    lines.extend([
        "## Per-Module Summary",
        "",
        "| Module | Trajs | Steps | Fusion% | MeanCluster | MeanCandSet |",
        "|---|---|---|---|---|---|",
    ])
    for s in sorted(stats_list,
                    key=lambda x: x.get('fusion_selection_pct', 0),
                    reverse=True):
        lines.append(
            f"| {s.get('module_key','')} "
            f"| {s.get('n_trajectories',0)} "
            f"| {s.get('n_total_steps',0)} "
            f"| {s.get('fusion_selection_pct',0):.1f}% "
            f"| {s.get('mean_cluster_size',0):.1f} "
            f"| {s.get('mean_candidate_set_size',0):.1f} |"
        )

    lines.extend([
        "",
        "## Limitations",
        "",
        "- Candidate set is approximate: dump does not log queue state directly",
        "- Queue removals for zero-user nodes are not tracked",
        "- Cluster fan-in/fan-out computed from initial graph, not mutated graph",
        "- Fusion kind defaults to kLoop for all dynamic fusions",
        "- `steps_since_creation` is set to 0 in stored features; "
        "recomputed at training time",
    ])

    path.write_text('\n'.join(lines))
    print(f"Wrote report: {path}")


# ======================================================================
# Inspect subcommand (debugging)
# ======================================================================

def _inspect_dump(dump_path: str, traj_pt_path: str):
    """Inspect one dump: clusters, alive sets, example features."""
    print(f"Loading graph from {traj_pt_path}...")
    pt_data = torch.load(traj_pt_path, weights_only=False)
    graph = pt_data['initial_graph']
    name_to_idx = {name: idx for idx, name in enumerate(graph.node_names)}
    fusible = _get_fusible_nodes(graph)
    opcode_names = _opcode_vocab()

    print(f"  {graph.num_nodes} nodes, {len(fusible)} fusible")

    print(f"\nParsing dump: {dump_path}...")
    parsed = parse_dump(dump_path)
    print(f"  {len(parsed.steps)} steps ({parsed.num_fusions} fused, "
          f"{parsed.num_ineligible} ineligible)")

    # Clusters
    clusters = reconstruct_clusters(parsed)
    print(f"\nFusion clusters: {len(clusters)}")
    for name, cl in sorted(clusters.items(), key=lambda x: x[1].creation_step):
        fp = cluster_fingerprint(cl)
        print(f"  {name} (step {cl.creation_step}, {len(cl.members)} members, "
              f"root={cl.root_consumer}, fp={fp})")
        if len(cl.members) <= 6:
            print(f"    members: {sorted(cl.members)}")

    # Alive sets
    alive_sets = track_alive_sets(parsed, fusible)
    print(f"\nAlive set evolution ({len(alive_sets)} steps):")
    step_count = max(1, len(alive_sets) // 8)
    for i in range(0, len(alive_sets), step_count):
        n_orig = sum(1 for x in alive_sets[i] if not x.startswith('fusion.'))
        n_fus = sum(1 for x in alive_sets[i] if x.startswith('fusion.'))
        print(f"  Step {i:4d}: {len(alive_sets[i]):3d} candidates "
              f"({n_orig} original, {n_fus} fusion.N)")

    # Verify selections are in alive sets
    n_miss = 0
    for step in parsed.steps:
        if not step.was_fused or step.step_idx >= len(alive_sets):
            continue
        if step.producer_name not in alive_sets[step.step_idx]:
            n_miss += 1
    fused = parsed.num_fusions
    print(f"\n  Alive-set coverage: {fused - n_miss}/{fused} "
          f"({100*(fused-n_miss)/fused:.1f}%)")

    # Example features
    if clusters:
        cl = next(iter(clusters.values()))
        feats = compute_cluster_features(cl, graph, name_to_idx, opcode_names)
        print(f"\nExample features for {cl.fusion_name} "
              f"({len(cl.members)} members):")
        print(f"  [0]  log2(size)       = {feats[0]:.2f}")
        print(f"  [1:11] opcode_hist    = "
              f"{[round(v, 3) for v in feats[1:11].tolist()]}")
        print(f"  [11] log2(bytes)      = {feats[11]:.2f}")
        print(f"  [12] log2(elems)      = {feats[12]:.2f}")
        print(f"  [13] max_rank         = {feats[13]:.0f}")
        print(f"  [14] log2(fan_in+1)   = {feats[14]:.2f}")
        print(f"  [15] log2(fan_out+1)  = {feats[15]:.2f}")
        print(f"  [16] contains_gemm    = {feats[16]:.0f}")
        print(f"  [17] contains_reduce  = {feats[17]:.0f}")
        print(f"  [18] contains_bcast   = {feats[18]:.0f}")
        print(f"  [19] contains_bitcast = {feats[19]:.0f}")
        print(f"  [20] contains_ewise   = {feats[20]:.0f}")
        print(f"  [21] root_opcode_id   = {feats[21]:.3f}")
        print(f"  [22] topo_span        = {feats[22]:.3f}")
        print(f"  [23] mean_topo_pos    = {feats[23]:.3f}")
        print(f"  [24:30] etype_hist    = "
              f"{[round(v, 3) for v in feats[24:30].tolist()]}")
        print(f"  [30:33] fusion_kind   = {feats[30:33].tolist()}")
        print(f"  [33] steps_since_crt  = {feats[33]:.0f} (placeholder)")


# ======================================================================
# CLI entry point
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Extract dynamic fusion-cluster features from XLA dumps"
    )
    sub = parser.add_subparsers(dest="command")

    # single
    p1 = sub.add_parser("single", help="Process one module")
    p1.add_argument("--traj-pt", required=True, help="Merged .pt trajectory file")
    p1.add_argument("--dump-base", required=True,
                    help="Directory with per-strategy dump subdirs")
    p1.add_argument("--output", default=None, help="Output .pt path")
    p1.add_argument("--gpu-csv", default=None, help="GPU profiles CSV")

    # batch
    p2 = sub.add_parser("batch", help="Process all modules")
    p2.add_argument("--traj-dir", required=True,
                    help="Directory of merged .pt files")
    p2.add_argument("--dump-root", required=True,
                    help="Root of multi_dumps tree")
    p2.add_argument("--output-dir", default="output/dynamic_features")
    p2.add_argument("--gpu-csv", default=None, help="GPU profiles CSV")

    # inspect
    p3 = sub.add_parser("inspect", help="Inspect one dump (debugging)")
    p3.add_argument("--dump", required=True, help="Path to dump file")
    p3.add_argument("--traj-pt", required=True, help=".pt file for graph data")

    args = parser.parse_args()

    if args.command == "single":
        output = args.output or (
            f"output/dynamic_features/{Path(args.traj_pt).stem}.pt"
        )
        process_module(
            traj_pt_path=args.traj_pt,
            dump_base=args.dump_base,
            output_path=output,
            gpu_csv_path=args.gpu_csv,
        )
    elif args.command == "batch":
        batch_process(
            traj_dir=args.traj_dir,
            dump_root=args.dump_root,
            output_dir=args.output_dir,
            gpu_csv=args.gpu_csv,
        )
    elif args.command == "inspect":
        _inspect_dump(args.dump, args.traj_pt)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
