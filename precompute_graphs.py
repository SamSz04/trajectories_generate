"""Pre-compute contracted graphs for all enriched trajectories.

Eliminates the ~10min graph contraction bottleneck during training by
computing all (Data, GraphStepMetadata) pairs up front and saving them
to disk as per-module .pt files.

Usage:
    python precompute_graphs.py \
        --enriched-dir output/dynamic_features \
        --output-dir output/precomputed_graphs \
        [--skip-computation] [--max-trajs-per-module 0] [--num-workers 8]

Output structure:
    output/precomputed_graphs/
        <module_key>.pt   # dict with keys: graphs, metadatas, traj_map
        manifest.json     # summary of all pre-computed modules
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool, cpu_count
from typing import Dict, List, Optional, Tuple

import torch
from torch_geometric.data import Data

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dynamic_graph_types import GraphStepMetadata, FidelityReport
from graph_contractor import reconstruct_and_contract_trajectory


def _metadata_to_dict(m: GraphStepMetadata) -> dict:
    """Convert GraphStepMetadata to a serializable dict."""
    return {
        "module_key": m.module_key,
        "trajectory_idx": m.trajectory_idx,
        "step_idx": m.step_idx,
        "step_position": m.step_position,
        "idx_to_name": m.idx_to_name,
        "name_to_idx": m.name_to_idx,
        "cluster_members": {k: list(v) for k, v in m.cluster_members.items()},
        "candidate_names": m.candidate_names,
        "selected_name": m.selected_name,
        "selected_idx": m.selected_idx,
        "candidate_indices": m.candidate_indices,
        "graph_fidelity": m.graph_fidelity,
        "notes": m.notes,
        "is_cluster": m.is_cluster,
    }


def _dict_to_metadata(d: dict) -> GraphStepMetadata:
    """Reconstruct GraphStepMetadata from a dict."""
    return GraphStepMetadata(
        module_key=d["module_key"],
        trajectory_idx=d["trajectory_idx"],
        step_idx=d["step_idx"],
        step_position=d["step_position"],
        idx_to_name=d["idx_to_name"],
        name_to_idx=d["name_to_idx"],
        cluster_members={k: frozenset(v) for k, v in d["cluster_members"].items()},
        candidate_names=d["candidate_names"],
        selected_name=d["selected_name"],
        selected_idx=d["selected_idx"],
        candidate_indices=d["candidate_indices"],
        graph_fidelity=d["graph_fidelity"],
        notes=d["notes"],
        is_cluster=d["is_cluster"],
    )


def precompute_module(
    pt_path: str,
    output_dir: str,
    max_trajectories: int = 0,
) -> dict:
    """Pre-compute all contracted graphs for one module.

    Returns summary dict with stats.
    """
    filename = os.path.basename(pt_path).replace(".pt", "")
    data = torch.load(pt_path, map_location="cpu", weights_only=False)
    module_key = data.get("module_key", filename)
    g0 = data["graph"]
    enriched_trajs = data["enriched_trajectories"]

    if max_trajectories > 0:
        enriched_trajs = enriched_trajs[:max_trajectories]

    # (traj_idx, step_position) -> index into flat lists
    traj_map = {}  # traj_idx -> {step_pos -> flat_index}
    all_graphs: List[Data] = []
    all_metadatas: List[dict] = []

    report = FidelityReport()
    t0 = time.time()

    for ti, traj in enumerate(enriched_trajs):
        # Build cluster features cache
        cluster_cache = {}
        for cname, cdata in traj.get("clusters", {}).items():
            if "features" in cdata and isinstance(cdata["features"], torch.Tensor):
                cluster_cache[cname] = cdata["features"]

        results = reconstruct_and_contract_trajectory(
            enriched_traj=traj,
            g0=g0,
            module_key=module_key,
            trajectory_idx=ti,
            cluster_features_cache=cluster_cache,
            report=report,
        )

        step_map = {}
        for graph_data, metadata in results:
            flat_idx = len(all_graphs)
            step_map[metadata.step_position] = flat_idx
            all_graphs.append(graph_data)
            all_metadatas.append(_metadata_to_dict(metadata))

        traj_map[ti] = step_map

    elapsed = time.time() - t0
    n_steps = len(all_graphs)

    # Save to disk
    out_path = os.path.join(output_dir, f"{filename}.pt")
    torch.save({
        "graphs": all_graphs,
        "metadatas": all_metadatas,
        "traj_map": traj_map,
        "module_key": module_key,
        "num_trajectories": len(enriched_trajs),
        "num_steps": n_steps,
    }, out_path)

    size_mb = os.path.getsize(out_path) / (1024 * 1024)

    summary = {
        "module_key": module_key,
        "filename": filename,
        "num_trajectories": len(enriched_trajs),
        "num_steps": n_steps,
        "num_nodes_g0": g0.num_nodes,
        "elapsed_sec": round(elapsed, 1),
        "file_size_mb": round(size_mb, 1),
        "fidelity": {
            "exact": report.exact_steps,
            "approximate": report.approximate_steps,
            "unresolved": report.unresolved_steps,
            "selected_in_ct": report.selected_in_ct,
            "selected_missing": report.selected_missing,
        },
    }

    print(f"  {module_key}: {n_steps} steps, "
          f"{len(enriched_trajs)} trajs, "
          f"{elapsed:.0f}s, {size_mb:.1f} MB")

    return summary


def _worker(args):
    """Multiprocessing worker wrapper."""
    return precompute_module(*args)


def main():
    parser = argparse.ArgumentParser(description="Pre-compute contracted graphs")
    parser.add_argument("--enriched-dir", default="output/dynamic_features")
    parser.add_argument("--output-dir", default="output/precomputed_graphs")
    parser.add_argument("--skip-computation", action="store_true", default=True)
    parser.add_argument("--max-trajs-per-module", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0,
                        help="Parallel workers (0 = sequential)")
    args = parser.parse_args()

    import glob
    pt_files = sorted(glob.glob(os.path.join(args.enriched_dir, "*.pt")))
    print(f"Found {len(pt_files)} enriched .pt files")

    if args.skip_computation:
        pt_files = [f for f in pt_files
                    if "computation" not in os.path.basename(f).lower()]
        print(f"After skipping computation modules: {len(pt_files)}")

    os.makedirs(args.output_dir, exist_ok=True)

    t0 = time.time()
    summaries = []

    if args.num_workers > 0:
        tasks = [(f, args.output_dir, args.max_trajs_per_module) for f in pt_files]
        with Pool(min(args.num_workers, len(pt_files))) as pool:
            summaries = pool.map(_worker, tasks)
    else:
        for pt_file in pt_files:
            summary = precompute_module(pt_file, args.output_dir,
                                        args.max_trajs_per_module)
            summaries.append(summary)

    total_elapsed = time.time() - t0
    total_steps = sum(s["num_steps"] for s in summaries)
    total_size = sum(s["file_size_mb"] for s in summaries)

    print(f"\nDone: {len(summaries)} modules, {total_steps} steps, "
          f"{total_size:.0f} MB, {total_elapsed:.0f}s")

    # Write manifest
    manifest = {
        "num_modules": len(summaries),
        "total_steps": total_steps,
        "total_size_mb": round(total_size, 1),
        "total_elapsed_sec": round(total_elapsed, 1),
        "modules": summaries,
    }
    manifest_path = os.path.join(args.output_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
