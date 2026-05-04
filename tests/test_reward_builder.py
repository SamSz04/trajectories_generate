"""Tests for reward_builder.py — dual-channel return conditioning."""

import math
import sys
import os
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reward_builder import (
    RewardConfig,
    RewardFidelity,
    classify_step_fidelity,
    compute_normalization_stats,
    normalize_per_step,
    compute_advantage,
    advantage_weight,
    build_returns,
)


# ======================================================================
# Fixtures
# ======================================================================

def _make_step(us_unfused=100.0, us_fused=80.0):
    return {"us_unfused": us_unfused, "us_fused": us_fused}


def _make_traj(steps, reward_gpu=1.5, strategy="test"):
    return {
        "steps": steps,
        "reward_gpu": reward_gpu,
        "strategy": strategy,
        "n_steps": len(steps),
    }


def _make_module(module_key, trajs):
    return {
        "module_key": module_key,
        "architecture": "TestArch",
        "trajectories": trajs,
    }


# ======================================================================
# RewardFidelity classification
# ======================================================================

class TestFidelityClassification:
    def test_exact(self):
        assert classify_step_fidelity({"us_unfused": 100, "us_fused": 80}) == RewardFidelity.EXACT

    def test_zero_exact(self):
        assert classify_step_fidelity({"us_unfused": 50, "us_fused": 50}) == RewardFidelity.ZERO_EXACT

    def test_missing_none(self):
        assert classify_step_fidelity({}) == RewardFidelity.MISSING
        assert classify_step_fidelity({"us_unfused": None, "us_fused": None}) == RewardFidelity.MISSING

    def test_missing_both_zero(self):
        assert classify_step_fidelity({"us_unfused": 0, "us_fused": 0}) == RewardFidelity.MISSING

    def test_priority_only(self):
        # One zero, one positive — likely from initial_priorities
        assert classify_step_fidelity({"us_unfused": 100, "us_fused": 0}) == RewardFidelity.PRIORITY_ONLY


# ======================================================================
# Normalization stats
# ======================================================================

class TestNormalizationStats:
    def test_module_p95_basic(self):
        steps = [_make_step(100, 80), _make_step(200, 100), _make_step(50, 50)]
        traj = _make_traj(steps)
        modules = [_make_module("mod_a", [traj])]

        stats = compute_normalization_stats(modules, "module_p95")
        assert "mod_a" in stats
        assert stats["mod_a"]["n_steps"] == 3  # zero_exact still counted
        assert stats["mod_a"]["p95"] > 0

    def test_empty_module(self):
        modules = [_make_module("empty", [_make_traj([])])]
        stats = compute_normalization_stats(modules, "module_p95")
        assert stats["empty"]["n_steps"] == 0
        assert stats["empty"]["p95"] == 1.0  # fallback

    def test_all_missing(self):
        steps = [{"us_unfused": 0, "us_fused": 0}]
        modules = [_make_module("miss", [_make_traj(steps)])]
        stats = compute_normalization_stats(modules, "module_p95")
        assert stats["miss"]["n_steps"] == 0

    def test_multiple_trajectories(self):
        t1 = _make_traj([_make_step(100, 80), _make_step(200, 150)])
        t2 = _make_traj([_make_step(300, 100), _make_step(50, 50)])
        modules = [_make_module("multi", [t1, t2])]

        stats = compute_normalization_stats(modules, "module_p95")
        assert stats["multi"]["n_steps"] == 4


# ======================================================================
# Per-step normalization
# ======================================================================

class TestNormalizePerStep:
    def test_module_p95(self):
        raw = [10.0, -20.0, 5.0]
        stats = {"p95": 20.0}
        result = normalize_per_step(raw, stats, "module_p95")
        assert result == [0.5, -1.0, 0.25]

    def test_module_zscore(self):
        raw = [10.0, 20.0, 30.0]
        stats = {"mean": 20.0, "std": 10.0}
        result = normalize_per_step(raw, stats, "module_zscore")
        assert result == [-1.0, 0.0, 1.0]

    def test_trajectory_l1(self):
        raw = [10.0, -20.0, 30.0]
        result = normalize_per_step(raw, {}, "trajectory_l1")
        l1 = 60.0
        assert abs(result[0] - 10.0 / l1) < 1e-8
        assert abs(result[1] - (-20.0 / l1)) < 1e-8

    def test_trajectory_l1_all_zero(self):
        raw = [0.0, 0.0, 0.0]
        result = normalize_per_step(raw, {}, "trajectory_l1")
        assert result == [0.0, 0.0, 0.0]

    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError):
            normalize_per_step([1.0], {}, "unknown")


# ======================================================================
# Advantage computation
# ======================================================================

class TestAdvantage:
    def test_beats_default(self):
        # R_real=2.0 → A_real = 1 - 1/2 = 0.5
        assert abs(compute_advantage(2.0) - 0.5) < 1e-8

    def test_matches_default(self):
        # R_real=1.0 → A_real = 0
        assert compute_advantage(1.0) == 0.0

    def test_worse_than_default(self):
        # R_real=0.5 → A_real = 1 - 2 = -1.0
        assert abs(compute_advantage(0.5) - (-1.0)) < 1e-8

    def test_none_reward(self):
        assert compute_advantage(None) == 0.0

    def test_zero_reward(self):
        assert compute_advantage(0.0) == 0.0

    def test_weight_positive(self):
        A = 0.5
        w = advantage_weight(A, beta=1.0, w_min=0.1, w_max=5.0)
        assert w == min(max(math.exp(0.5), 0.1), 5.0)

    def test_weight_clipped_low(self):
        A = -10.0
        w = advantage_weight(A, beta=1.0, w_min=0.1, w_max=5.0)
        assert w == 0.1

    def test_weight_clipped_high(self):
        A = 10.0
        w = advantage_weight(A, beta=1.0, w_min=0.1, w_max=5.0)
        assert w == 5.0


# ======================================================================
# build_returns
# ======================================================================

class TestBuildReturns:
    def test_flat_real_legacy_max(self):
        """flat_real with legacy_max should reproduce current behavior."""
        steps = [_make_step(100, 80)] * 5
        traj = _make_traj(steps, reward_gpu=1.5)
        config = RewardConfig(
            return_mode="flat_real",
            real_return_norm="legacy_max",
            max_gpu_reward=2.0,
        )
        returns, fids = build_returns(traj, "mod", {}, config)

        assert len(returns) == 5
        assert len(fids) == 5
        # R_real = 1.5 / 2.0 = 0.75
        for r in returns:
            assert len(r) == 1
            assert abs(r[0] - 0.75) < 1e-8

    def test_flat_real_raw(self):
        steps = [_make_step()] * 3
        traj = _make_traj(steps, reward_gpu=1.5)
        config = RewardConfig(return_mode="flat_real", real_return_norm="raw")
        returns, _ = build_returns(traj, "mod", {}, config)
        assert all(r[0] == 1.5 for r in returns)

    def test_flat_real_clipped(self):
        steps = [_make_step()] * 2
        traj = _make_traj(steps, reward_gpu=8.0)
        config = RewardConfig(return_mode="flat_real", real_return_norm="clipped")
        returns, _ = build_returns(traj, "mod", {}, config)
        assert all(r[0] == 5.0 for r in returns)  # clipped

    def test_cm_dense_basic(self):
        # r_cm = [20, 100, 0] (steps: 100-80, 200-100, 50-50)
        steps = [_make_step(100, 80), _make_step(200, 100), _make_step(50, 50)]
        traj = _make_traj(steps)
        norm_stats = {"mod": {"p95": 100.0, "mean": 0.0, "std": 1.0}}
        config = RewardConfig(return_mode="cm_dense", norm_mode="module_p95")

        returns, fids = build_returns(traj, "mod", norm_stats, config)

        assert len(returns) == 3
        assert all(len(r) == 1 for r in returns)

        # r_cm_norm = [20/100, 100/100, 0/100] = [0.2, 1.0, 0.0]
        # RTG[2] = 0.0, RTG[1] = 1.0, RTG[0] = 0.2 + 1.0 = 1.2  (wait, backward cumsum)
        # Actually: RTG[2] = r_norm[2] = 0.0
        #          RTG[1] = r_norm[1] + RTG[2] = 1.0
        #          RTG[0] = r_norm[0] + RTG[1] = 0.2 + 1.0 = 1.2
        assert abs(returns[0][0] - 1.2) < 1e-6
        assert abs(returns[1][0] - 1.0) < 1e-6
        assert abs(returns[2][0] - 0.0) < 1e-6

    def test_cm_dense_clipping(self):
        """RTG should be clipped to [-10, 10]."""
        # 50 steps with r_cm = 100 each, p95 = 1.0 → r_norm = 100
        # RTG[0] = 50*100 = 5000 → clipped to 10
        steps = [_make_step(200, 100)] * 50
        traj = _make_traj(steps)
        norm_stats = {"mod": {"p95": 1.0}}
        config = RewardConfig(return_mode="cm_dense", norm_mode="module_p95")

        returns, _ = build_returns(traj, "mod", norm_stats, config)
        assert returns[0][0] == 10.0  # clipped

    def test_hybrid_shape(self):
        steps = [_make_step(100, 80), _make_step(200, 100)]
        traj = _make_traj(steps, reward_gpu=1.5)
        norm_stats = {"mod": {"p95": 100.0}}
        config = RewardConfig(
            return_mode="hybrid",
            norm_mode="module_p95",
            real_return_norm="raw",
        )

        returns, fids = build_returns(traj, "mod", norm_stats, config)

        assert len(returns) == 2
        assert all(len(r) == 2 for r in returns)  # return_dim = 2
        # Channel 1: RTG_cm, Channel 2: R_real
        assert returns[0][1] == 1.5  # R_real constant
        assert returns[1][1] == 1.5

    def test_empty_trajectory(self):
        traj = _make_traj([], reward_gpu=1.5)
        config = RewardConfig(return_mode="flat_real")
        returns, fids = build_returns(traj, "mod", {}, config)
        assert returns == []
        assert fids == []

    def test_missing_gpu_reward(self):
        steps = [_make_step()] * 3
        traj = _make_traj(steps, reward_gpu=None)
        config = RewardConfig(return_mode="flat_real", real_return_norm="raw")
        returns, _ = build_returns(traj, "mod", {}, config)
        # Fallback R_real = 1.0
        assert all(r[0] == 1.0 for r in returns)

    def test_all_zero_r_cm(self):
        steps = [_make_step(50, 50)] * 5  # all zero r_cm
        traj = _make_traj(steps, reward_gpu=1.0)
        norm_stats = {"mod": {"p95": 1.0}}
        config = RewardConfig(return_mode="cm_dense", norm_mode="module_p95")
        returns, _ = build_returns(traj, "mod", norm_stats, config)
        assert all(r[0] == 0.0 for r in returns)

    def test_fidelity_labels(self):
        steps = [
            _make_step(100, 80),         # EXACT
            {"us_unfused": 50, "us_fused": 50},  # ZERO_EXACT
            {},                           # MISSING
        ]
        traj = _make_traj(steps)
        config = RewardConfig(return_mode="flat_real")
        _, fids = build_returns(traj, "mod", {}, config)
        assert fids[0] == "exact"
        assert fids[1] == "zero_exact"
        assert fids[2] == "missing"

    def test_return_dim_property(self):
        assert RewardConfig(return_mode="flat_real").return_dim == 1
        assert RewardConfig(return_mode="cm_dense").return_dim == 1
        assert RewardConfig(return_mode="hybrid").return_dim == 2

    def test_unknown_return_mode_raises(self):
        steps = [_make_step()]
        traj = _make_traj(steps)
        config = RewardConfig(return_mode="unknown")
        with pytest.raises(ValueError):
            build_returns(traj, "mod", {}, config)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
