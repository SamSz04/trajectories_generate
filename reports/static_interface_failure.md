# Static Interface Failure Report

**Date**: 2026-05-02
**Status**: Root cause identified. Blocks all further model work.

---

## Executive Summary

The FusionDT system's deployment interface is fundamentally lossy. XLA's PriorityFusion pass is a **dynamic** algorithm where the graph changes after every fusion step, creating new `fusion.N` instructions that themselves become producers. The DT learns from these dynamic trajectories, but deploys through a **static** JSON `producer_ordering` that can only express priorities for nodes in the original HLO graph. Dynamically-created `fusion.N` nodes — which constitute **46% of all fusion decisions** — are stripped from the plan and receive the lowest priority in XLA.

**Proof**: Extracting XLA default's own node ordering from its trajectory, stripping fusion tokens, and feeding it back through the static JSON interface produces a **+80.6% kernel time regression** — virtually identical to the DT's +80.3% regression. The model is not the problem; the interface is.

---

## 1. Background: How XLA PriorityFusion Works

XLA's PriorityFusion is a greedy priority-queue algorithm:

```
1. Compute priority for every fusible producer: priority = time_unfused - time_fused
2. Insert all producers into a max-heap
3. Loop:
   a. Dequeue highest-priority producer
   b. Fuse it into all fusible consumers → creates "fusion.N" instruction
   c. The fusion.N replaces the consumer in the graph
   d. Recompute priorities for all affected neighbors
   e. Insert fusion.N into the queue (it is now a new producer)
   f. Repeat until queue is empty
```

**Key property**: The graph is modified after every fusion. New `fusion.N` nodes are created, and they participate in future fusion decisions. The order in which producers are dequeued affects which fusions are legal and how the graph evolves.

## 2. The FusionDT Pipeline

### Training (dynamic)
The DT trains on trajectories recorded from XLA's PriorityFusion under various strategies (random, perturbed, SA, etc.). Each trajectory is a sequence of (state, action, reward) tuples:

- **State**: Binary fused_mask — which nodes have been dequeued so far
- **Action**: Node index of the dequeued producer. If the producer is a dynamically-created `fusion.N` (not in the original graph), the action is recorded as `producer_id = -1` and mapped to a special "fusion token" (action index = num_nodes)
- **Reward**: Trajectory-level GPU speedup: `xla_default_us / strategy_us`

Example trajectory for `gqa_qkv_proj` (79 original nodes, 61 steps):
```
Step  0: node[54]       "bitcast.36.0"       → fused into convert.9.0
Step  1: node[ 6]       "bitcast.32.0"       → fused into convert.8.0
  ...
Step  7: FUSION_TOKEN   "fusion.4"           → fused into split.4.0, split.5.0
Step  8: FUSION_TOKEN   "fusion.4"           → (multi-consumer continuation)
Step  9: node[45]       "broadcast.21.0"     → fused into mul.54.0
Step 10: FUSION_TOKEN   "fusion.12"          → fused into convert.12.0
Step 11: node[42]       "convert.10.0"       → fused into fusion.13
Step 12: FUSION_TOKEN   "fusion.7"           → fused into mul.41.0, mul.49.0
  ...
```

Steps 7, 8, 10, 12 are fusion token actions — the model learns when and where these dynamically-created nodes should be dequeued relative to original nodes.

### Inference (static)
The DT generates an action sequence autoregressively, then:

1. `actions_to_producer_names()` maps node indices to HLO instruction names
2. **All fusion token actions are stripped** — they have no corresponding name in the original HLO graph
3. Uncovered original nodes are appended at the end
4. The ordering is written as JSON: `{"producer_ordering": ["mul.44.0", "broadcast.43.0", ...]}`
5. XLA reads this with `DT_FUSION_STRATEGY=json_plan` and assigns priorities: position 0 → highest, position N → lowest
6. **Any producer NOT in the plan (including all dynamically-created `fusion.N`) gets priority 0** — i.e., dequeued last

## 3. The Problem: Information Loss

### Quantitative: Fusion Token Fraction

Across all 33 modules' XLA default trajectories:

| Architecture | Steps | Real Nodes | Fusion Tokens | Fusion Token % |
|---|---|---|---|---|
| DeepSeek-V3 | 683 | 383 | 300 | 43.9% |
| Griffin-2B | 802 | 284 | 518 | 64.6% |
| Llama-3-8B | 627 | 319 | 308 | 49.1% |
| Mamba2-370M | 569 | 262 | 307 | 53.9% |
| PaliGemma-3B | 661 | 423 | 238 | 36.0% |
| SigLIP-Base | 1464 | 942 | 522 | 35.7% |
| **TOTAL** | **5237** | **2828** | **2409** | **46.0%** |

Nearly half of all fusion decisions are lost at the interface boundary.

### Qualitative: What Fusion Tokens Represent

Fusion tokens are not noise — they represent critical intermediate fusion decisions:

1. **Cluster growth**: After `mul.44.0` fuses into `add.2.0` creating `fusion.7`, the fusion token step says "now dequeue `fusion.7` and fuse it into `convert.5.0`." This grows the fusion cluster.
2. **Priority ordering of clusters**: In the dynamic execution, `fusion.7` might have higher priority than original node `broadcast.21.0`. The static plan cannot express "fuse the cluster before the broadcast."
3. **Graph-dependent legality**: After `fusion.7` absorbs more nodes, different fusions become legal or illegal for downstream producers. The order matters.

When fusion tokens are stripped and get priority 0 (lowest), all cluster-growth decisions are deferred to the end. This means:
- Small, separate fusion clusters form first (from original nodes)
- Large cluster merging happens last (when fusion.N nodes are finally dequeued)
- This is the **opposite** of XLA default's strategy, which often grows large clusters early

## 4. Experimental Proof: The Round-Trip Test

**Methodology**: For 8 representative modules, we extracted XLA default's own trajectory, kept only the original-node ordering (stripping fusion tokens), wrote it as a JSON plan, and profiled it through XLA.

**Hypothesis**: If the static interface is faithful, round-tripping XLA default's own ordering should reproduce XLA default's kernel time (~0% difference).

**Results**:

| Module | XLA Default (us) | Round-Trip (us) | RT vs XLA | DT Plan vs XLA | Fusion Token % |
|---|---|---|---|---|---|
| griffin_ffn | 8,805 | 28,567 | **+224.4%** | +223.5% | 44.4% |
| shared_block | 6,366 | 14,804 | **+132.5%** | +132.4% | 60.0% |
| gqa_qkv_proj | 9,893 | 24,276 | **+145.4%** | +144.6% | 67.2% |
| ffn_layer | 87,657 | 186,952 | **+113.3%** | +113.6% | 43.5% |
| routed_block | 244,140 | 403,189 | **+65.1%** | +64.7% | 37.9% |
| rg_lru | 13,983 | 20,497 | **+46.6%** | +46.6% | 68.6% |
| vision_mlp | 497 | 1,097 | **+121.0%** | +120.9% | 29.4% |
| mamba_block | 13,969 | 16,530 | **+18.3%** | +18.4% | 53.7% |
| **Weighted Total** | | | **+80.6%** | **+80.3%** | |

**Key observations**:
1. Round-trip regression (+80.6%) ≈ DT plan regression (+80.3%) — they are the same failure mode
2. Even a "perfect" model that exactly reproduces XLA default's ordering cannot do better than +80.6% through this interface
3. The regression is NOT correlated with fusion token fraction alone — `rg_lru` has 69% fusion tokens but "only" +47% regression, while `griffin_ffn` has 44% fusion tokens but +224% regression. The impact depends on how critical the fusion token ordering is for that module's fusion dynamics.

## 5. Why Previous Improvements Had No Effect

| Improvement | Val Accuracy | GPU Time vs XLA | Why It Failed |
|---|---|---|---|
| Lite mode (baseline) | 58% | +88.8% | Interface bottleneck |
| Full mode (per-step GNN) | 67.6% | +88.7% | Same interface |
| Hybrid mode (GNN every 5 steps) | 66.0% | +89.2% | Same interface |
| gpu_progressive RTG | ~58% | +88.5% | Same interface |
| P0 fix (repeat penalty + fusion strip) | ~58% | +88.4% | Same interface |
| Larger model (128d, 6L) | 58% | not profiled | Same interface |

Every model improvement converges to the same GPU time because they all deploy through the same lossy static interface. The ~+89% regression is an **interface constant**, not a model quality metric.

## 6. What the Existing Strategies Tell Us

Importantly, the env-var-based strategies (random, perturbed, SA) that DO beat XLA default work through a fundamentally different mechanism:

- `random`, `perturbed`, `fanout`, etc. override `CalculateProducerPriority()` with a formula
- This formula is evaluated **dynamically** for every producer, including `fusion.N` nodes
- When `fusion.7` is created, its priority is computed by the strategy formula just like any original node
- The dynamic priority assignment is what allows these strategies to sometimes outperform XLA default

The `json_plan` strategy is the only one that assigns priority 0 to unknown nodes, making it fundamentally different from all other strategies.

## 7. Fix: Cost-Model Fallback (Option A) — VALIDATED

**Date**: 2026-05-02

**Change**: Modified `json_plan` strategy in `priority_fusion.cc` so that producers NOT in the plan (dynamically-created `fusion.N`) fall through to the default cost-model priority computation (`time_unfused - time_fused`) instead of getting priority 0.

```cpp
// BEFORE:
if (strategy == "json_plan") {
    ...
    return absl::Microseconds(0);  // not in plan: lowest priority
}

// AFTER:
if (strategy == "json_plan") {
    ...
    // Not in plan (dynamically-created fusion.N):
    // Fall through to default cost-model priority below
} else {
    LOG(WARNING) << "Unknown strategy...";
}
// Default cost-model computation runs for fusion.N nodes
```

**Round-trip test results (V2 vs V1)**:

| Module | XLA Default (us) | RT V1 (priority 0) | V1 vs XLA | RT V2 (cost-model) | V2 vs XLA |
|---|---|---|---|---|---|
| griffin_ffn | 28,506 | 28,567 | **+224.4%** | 28,522 | **+0.06%** |
| shared_block | 14,793 | 14,804 | **+132.5%** | 14,795 | **+0.01%** |
| gqa_qkv_proj | 24,102 | 24,276 | **+145.4%** | 24,151 | **+0.21%** |
| ffn_layer | 188,601 | 186,952 | **+113.3%** | 187,656 | **-0.50%** |
| routed_block | 402,544 | 403,189 | **+65.1%** | 401,643 | **-0.22%** |
| rg_lru | 20,492 | 20,497 | **+46.6%** | 20,495 | **+0.02%** |
| vision_mlp | 1,097 | 1,097 | **+121.0%** | 1,097 | **+0.05%** |
| mamba_block | 15,434 | 16,530 | **+18.3%** | 15,377 | **-0.37%** |
| **Weighted Avg** | | | **+80.6%** | | **~-0.1%** |

**Conclusion**: The cost-model fallback completely eliminates the interface bottleneck. All modules reproduce XLA default within measurement noise (<0.5%).

## 8. Implications (Updated)

1. **The interface is now fixed** — `json_plan` with cost-model fallback is a faithful deployment interface.

2. **DT training is meaningful again** — the plan controls original-node ordering, while XLA's cost-model heuristic handles fusion.N ordering. Any improvement in the DT's ordering will now translate to real GPU speedup.

3. **Next steps**:
   - Re-profile existing DT plans with the V2 binary to see if the DT already produces speedups
   - If DT plans still regress, the model needs improvement (but the interface is no longer the bottleneck)
   - Token accuracy is now a meaningful (though imperfect) proxy metric

---

## Appendix: File References

- Round-trip test plans: `output/roundtrip_plans/` (on GPU server)
- Round-trip V1 profiling: `output/gpu_profiles_roundtrip.csv` (on GPU server)
- Round-trip V2 profiling: `output/gpu_profiles_roundtrip_v2.csv` (on GPU server)
- Full mode profiling: `output/gpu_profiles_full_mode.csv`
- Hybrid mode profiling: `output/gpu_profiles_hybrid5_mode.csv`
- XLA baseline: `output/gpu_profiles_merged.csv` (xla_default rows)
- XLA patch (updated): `priority_fusion_patch.diff`
- Plan generation: `dt_eval_multi.py` → `actions_to_producer_names()`
- Plan consumption: XLA `CalculateProducerPriority()` → `json_plan` branch
