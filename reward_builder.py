"""Dual-channel return conditioning for Dynamic Fusion DT.

Computes per-step return vectors combining dense cost-model proxy
rewards with trajectory-level real GPU returns.

Key design: normalize per-step r_cm BEFORE backward cumsum to keep
RTG_cm in a bounded range across modules of vastly different scale.

Return modes:
  flat_real:  [[R_real_norm]] * T                    → return_dim=1
  cm_dense:   [[RTG_cm_t]] per step                  → return_dim=1
  hybrid:     [[RTG_cm_t, R_real_norm]] per step     → return_dim=2

Usage:
    from reward_builder import RewardConfig, compute_normalization_stats, build_returns

    config = RewardConfig(return_mode="hybrid", norm_mode="module_p95")
    norm_stats = compute_normalization_stats(train_modules, config.norm_mode)
    returns, fidelities = build_returns(traj, "Llama__gqa_layer", norm_stats, config)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple


# ======================================================================
# Configuration
# ======================================================================

@dataclass
class RewardConfig:
    """Configuration for return computation."""
    return_mode: str = "flat_real"       # flat_real | cm_dense | hybrid
    norm_mode: str = "module_p95"        # module_p95 | module_zscore | trajectory_l1
    real_return_norm: str = "legacy_max" # legacy_max | raw | clipped
    max_gpu_reward: float = 1.0          # For legacy_max normalization
    rtg_clip_lo: float = -10.0
    rtg_clip_hi: float = 10.0

    @property
    def return_dim(self) -> int:
        if self.return_mode == "hybrid":
            return 2
        return 1


# ======================================================================
# Reward fidelity classification
# ======================================================================

class RewardFidelity(str, Enum):
    """Classification of per-step cost-model reward quality."""
    EXACT = "exact"                # us_fused > 0 and us_unfused > 0
    PRIORITY_ONLY = "priority_only"  # Has data from initial_priorities only
    ZERO_EXACT = "zero_exact"      # r_cm = 0 exactly (us_fused == us_unfused)
    IMPUTED = "imputed"            # r_cm was imputed/interpolated
    MISSING = "missing"            # No cost-model data at all


def classify_step_fidelity(step: dict) -> RewardFidelity:
    """Classify a step's cost-model reward fidelity."""
    us_fused = step.get("us_fused")
    us_unfused = step.get("us_unfused")

    if us_fused is None or us_unfused is None:
        return RewardFidelity.MISSING

    us_fused = float(us_fused)
    us_unfused = float(us_unfused)

    if us_fused == 0 and us_unfused == 0:
        return RewardFidelity.MISSING

    if us_fused == us_unfused:
        return RewardFidelity.ZERO_EXACT

    if us_fused > 0 and us_unfused > 0:
        return RewardFidelity.EXACT

    # One is zero, other is positive — likely from initial_priorities
    return RewardFidelity.PRIORITY_ONLY


# ======================================================================
# Normalization statistics
# ======================================================================

def compute_normalization_stats(
    module_data_list: List[dict],
    norm_mode: str = "module_p95",
) -> Dict[str, dict]:
    """Compute per-module normalization statistics from training data.

    MUST be called on training split only. Save result to checkpoint
    for reuse during validation and evaluation.

    Args:
        module_data_list: List of module dicts with "module_key", "trajectories"
        norm_mode: Normalization strategy

    Returns:
        {module_key: {"p95": float, "mean": float, "std": float, "n_steps": int}}
    """
    stats = {}

    for mdata in module_data_list:
        module_key = mdata["module_key"]
        all_r_cm = []

        for traj in mdata["trajectories"]:
            steps = traj.get("steps", [])
            for s in steps:
                fidelity = classify_step_fidelity(s)
                if fidelity == RewardFidelity.MISSING:
                    continue
                r_cm = float(s.get("us_unfused", 0)) - float(s.get("us_fused", 0))
                all_r_cm.append(r_cm)

        if not all_r_cm:
            stats[module_key] = {
                "p95": 1.0, "mean": 0.0, "std": 1.0, "n_steps": 0,
            }
            continue

        abs_vals = [abs(r) for r in all_r_cm]
        abs_vals_sorted = sorted(abs_vals)

        n = len(all_r_cm)
        mean_val = sum(all_r_cm) / n
        var_val = sum((r - mean_val) ** 2 for r in all_r_cm) / max(n - 1, 1)
        std_val = math.sqrt(var_val)

        p95_idx = min(int(n * 0.95), n - 1)
        p95_val = abs_vals_sorted[p95_idx] if abs_vals_sorted else 1.0

        stats[module_key] = {
            "p95": max(p95_val, 1e-8),
            "mean": mean_val,
            "std": max(std_val, 1e-8),
            "n_steps": n,
        }

    return stats


# ======================================================================
# Per-step normalization
# ======================================================================

def normalize_per_step(
    r_cm_raw: List[float],
    module_stats: dict,
    norm_mode: str,
) -> List[float]:
    """Normalize raw per-step cost-model rewards.

    Applied BEFORE backward cumsum to keep RTG_cm bounded.
    """
    if norm_mode == "module_p95":
        scale = module_stats.get("p95", 1.0)
        return [r / max(scale, 1e-8) for r in r_cm_raw]

    elif norm_mode == "module_zscore":
        mu = module_stats.get("mean", 0.0)
        sigma = module_stats.get("std", 1.0)
        return [(r - mu) / max(sigma, 1e-8) for r in r_cm_raw]

    elif norm_mode == "trajectory_l1":
        l1 = sum(abs(r) for r in r_cm_raw)
        if l1 < 1e-8:
            return [0.0] * len(r_cm_raw)
        return [r / l1 for r in r_cm_raw]

    else:
        raise ValueError(f"Unknown norm_mode: {norm_mode}")


# ======================================================================
# Advantage computation
# ======================================================================

def compute_advantage(gpu_reward: Optional[float]) -> float:
    """Compute advantage: A_real = 1 - 1/R_real.

    R_real = t_XLA / t_strategy (gpu_reward). R_real > 1 means faster.
    A_real > 0 when trajectory beats XLA default.
    """
    if gpu_reward is None or gpu_reward <= 0:
        return 0.0
    return 1.0 - 1.0 / gpu_reward


def advantage_weight(
    A_real: float,
    beta: float = 1.0,
    w_min: float = 0.1,
    w_max: float = 5.0,
) -> float:
    """Compute trajectory weight: w = clip(exp(beta * A_real), w_min, w_max)."""
    return min(max(math.exp(beta * A_real), w_min), w_max)


# ======================================================================
# Return building
# ======================================================================

def _normalize_real_return(
    gpu_reward: Optional[float],
    real_return_norm: str,
    max_gpu_reward: float,
) -> float:
    """Normalize the trajectory-level real GPU return."""
    if gpu_reward is None:
        return 1.0  # neutral: assume XLA-default performance

    r = float(gpu_reward)

    if real_return_norm == "legacy_max":
        return r / max(max_gpu_reward, 1e-8)
    elif real_return_norm == "raw":
        return r
    elif real_return_norm == "clipped":
        return min(max(r, 0.0), 5.0)
    else:
        raise ValueError(f"Unknown real_return_norm: {real_return_norm}")


def build_returns(
    traj: dict,
    module_key: str,
    norm_stats: Dict[str, dict],
    config: RewardConfig,
) -> Tuple[List[List[float]], List[str]]:
    """Build per-step return vectors and fidelity labels for one trajectory.

    Returns:
        (returns_list, fidelity_list)
        returns_list: List[List[float]] of length T, inner lists have return_dim elements
        fidelity_list: List[str] of RewardFidelity values
    """
    steps = traj.get("steps", [])
    T = len(steps)

    if T == 0:
        return [], []

    # Classify fidelity
    fidelities = [classify_step_fidelity(s).value for s in steps]

    # R_real (trajectory-level, constant across steps)
    gpu_reward = traj.get("reward_gpu")
    R_real = _normalize_real_return(
        gpu_reward, config.real_return_norm, config.max_gpu_reward,
    )

    if config.return_mode == "flat_real":
        returns = [[R_real]] * T
        return returns, fidelities

    # Cost-model path: compute normalized per-step r_cm then backward cumsum
    module_stats = norm_stats.get(module_key, {"p95": 1.0, "mean": 0.0, "std": 1.0})

    # 1. Raw per-step cost-model rewards
    r_cm_raw = []
    for s in steps:
        us_unfused = float(s.get("us_unfused", 0))
        us_fused = float(s.get("us_fused", 0))
        r_cm_raw.append(us_unfused - us_fused)

    # 2. Normalize per-step BEFORE backward cumsum
    r_cm_norm = normalize_per_step(r_cm_raw, module_stats, config.norm_mode)

    # 3. Backward cumsum
    rtg_cm = [0.0] * T
    rtg_cm[T - 1] = r_cm_norm[T - 1]
    for t in range(T - 2, -1, -1):
        rtg_cm[t] = r_cm_norm[t] + rtg_cm[t + 1]

    # 4. Clip
    rtg_cm = [max(config.rtg_clip_lo, min(config.rtg_clip_hi, v)) for v in rtg_cm]

    if config.return_mode == "cm_dense":
        returns = [[rtg_cm[t]] for t in range(T)]
        return returns, fidelities

    elif config.return_mode == "hybrid":
        returns = [[rtg_cm[t], R_real] for t in range(T)]
        return returns, fidelities

    else:
        raise ValueError(f"Unknown return_mode: {config.return_mode}")


# ======================================================================
# Diagnostic report
# ======================================================================

def compute_reward_diagnostics(
    module_data_list: List[dict],
    norm_stats: Dict[str, dict],
) -> Dict[str, object]:
    """Compute diagnostic statistics for reward_build_report.md.

    Returns a dict with:
      - per_module: {module_key: {raw_stats, fidelity_counts, ...}}
      - global: {total_steps, zero_frac, correlation, ...}
    """
    import numpy as np

    per_module = {}
    all_sum_r_cm = []
    all_R_real = []

    for mdata in module_data_list:
        module_key = mdata["module_key"]
        all_r_cm = []
        fidelity_counts = {f.value: 0 for f in RewardFidelity}

        for traj in mdata["trajectories"]:
            steps = traj.get("steps", [])
            traj_sum = 0.0
            for s in steps:
                fid = classify_step_fidelity(s)
                fidelity_counts[fid.value] += 1
                r_cm = float(s.get("us_unfused", 0)) - float(s.get("us_fused", 0))
                all_r_cm.append(r_cm)
                traj_sum += r_cm

            gpu_r = traj.get("reward_gpu")
            if gpu_r is not None:
                all_sum_r_cm.append(traj_sum)
                all_R_real.append(float(gpu_r))

        if all_r_cm:
            arr = np.array(all_r_cm)
            raw_stats = {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
                "p5": float(np.percentile(arr, 5)),
                "p25": float(np.percentile(arr, 25)),
                "p50": float(np.percentile(arr, 50)),
                "p75": float(np.percentile(arr, 75)),
                "p95": float(np.percentile(arr, 95)),
                "zero_frac": float(np.mean(arr == 0)),
            }
        else:
            raw_stats = {}

        per_module[module_key] = {
            "raw_stats": raw_stats,
            "fidelity_counts": fidelity_counts,
            "norm_stats": norm_stats.get(module_key, {}),
            "n_steps": len(all_r_cm),
        }

    # Global correlation
    corr = None
    if len(all_sum_r_cm) > 2:
        arr_cm = np.array(all_sum_r_cm)
        arr_real = np.array(all_R_real)
        if arr_cm.std() > 1e-8 and arr_real.std() > 1e-8:
            corr = float(np.corrcoef(arr_cm, arr_real)[0, 1])

    global_stats = {
        "total_steps": sum(m["n_steps"] for m in per_module.values()),
        "total_modules": len(per_module),
        "sum_r_cm_vs_R_real_correlation": corr,
    }

    return {"per_module": per_module, "global": global_stats}
