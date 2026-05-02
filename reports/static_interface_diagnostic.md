# Static Interface Diagnostic Report

**Date**: 2026-05-02
**Status**: Diagnostic complete. Findings inform next-stage architecture design.

---

## 1. Summary

XLA's PriorityFusion pass is a **dynamic greedy algorithm**: after each fusion step the graph mutates, producing new `fusion.N` instructions that re-enter the priority queue and participate in future decisions. Our initial Decision Transformer (DT) pipeline learned from these dynamic trajectories but deployed through a **static JSON interface** (`producer_ordering`) that could only name nodes present in the original, pre-fusion HLO graph. Every dynamically-created `fusion.N` node was silently dropped from the plan and defaulted to the lowest priority inside XLA.

This report documents the failure, quantifies its severity, and describes a diagnostic patch (cost-model fallback) that isolates the root cause. The fallback is an analytical tool, not a proposed method; the intended next step is a learned dynamic priority model.

---

## 2. Background: XLA PriorityFusion Loop

```
1.  Compute priority for every fusible producer
2.  Insert all producers into a max-heap
3.  Loop:
      a. Pop highest-priority producer P
      b. Fuse P into its consumers -> creates a new "fusion.N" instruction
      c. fusion.N replaces the consumer(s) in the live graph
      d. Recompute priorities for all affected neighbours
      e. Push fusion.N into the heap (it is now a candidate producer)
      f. Repeat until heap is empty
```

Two properties matter:

- **Graph mutation**: each fusion changes the graph, making some future fusions legal and others illegal.
- **Dynamic producers**: `fusion.N` nodes created mid-loop participate in later decisions on equal footing with original HLO instructions.

---

## 3. The Static Interface

### Training side (dynamic)

The DT trains on full trajectories recorded from XLA under various priority strategies. Each step is a (state, action, reward) tuple:

- **State**: binary fused-mask over original nodes.
- **Action**: index of the dequeued producer. If the producer is a dynamically-created `fusion.N` (absent from the original graph), the action is mapped to a special fusion token (index = `num_nodes`).
- **Reward**: trajectory-level GPU speedup `xla_default_us / strategy_us`.

### Deployment side (static)

At inference the DT emits an action sequence autoregressively, then:

1. `actions_to_producer_names()` maps node indices to original HLO instruction names.
2. All fusion-token actions are **discarded** -- they have no corresponding name.
3. Uncovered original nodes are appended at the end.
4. The result is serialized as `{"producer_ordering": ["mul.44.0", "broadcast.43.0", ...]}`.
5. XLA reads this via `DT_FUSION_STRATEGY=json_plan` and assigns priority by position.
6. Any producer **not** in the plan (including every dynamically-created `fusion.N`) originally received priority 0 -- i.e., dequeued last.

---

## 4. Quantifying Information Loss

### 4.1 Fusion-token fraction across architectures

Across all 33 modules' XLA-default trajectories:

| Architecture | Trajectory Steps | Original Nodes | Fusion Tokens | Fusion Token % |
|---|---|---|---|---|
| DeepSeek-V3 | 683 | 383 | 300 | 43.9% |
| Griffin-2B | 802 | 284 | 518 | 64.6% |
| Llama-3-8B | 627 | 319 | 308 | 49.1% |
| Mamba2-370M | 569 | 262 | 307 | 53.9% |
| PaliGemma-3B | 661 | 423 | 238 | 36.0% |
| SigLIP-Base | 1464 | 942 | 522 | 35.7% |
| **Total** | **5237** | **2828** | **2409** | **46.0%** |

Nearly half of all fusion decisions are invisible to the static interface.

### 4.2 What fusion tokens represent

Fusion tokens are not noise. They encode:

1. **Cluster growth**: after `mul.44.0` fuses into `add.2.0` creating `fusion.7`, the next fusion-token step says "now dequeue `fusion.7` and fuse it into `convert.5.0`." This grows the cluster.
2. **Inter-cluster priority**: in the dynamic execution `fusion.7` may have higher cost-model priority than a later original node. The static plan cannot express "fuse the cluster before the broadcast."
3. **Legality dependencies**: after `fusion.7` absorbs more nodes, different fusions become legal or illegal for downstream producers.

When fusion tokens are stripped and their corresponding `fusion.N` nodes receive priority 0 (lowest), all cluster-growth decisions are deferred to the very end. Small, isolated clusters form first (from original nodes), and large merges happen last. This reverses XLA default's strategy, which often grows large clusters early.

---

## 5. Round-Trip Proof

**Method**: extract XLA default's own trajectory, keep only the original-node sub-ordering, write it as a JSON plan, and profile it through XLA.

**Hypothesis**: if the interface is faithful, round-tripping XLA default's own ordering should reproduce its kernel time (0% difference).

### 5.1 Results (V1 -- priority 0 for unknown producers)

| Module | XLA Default (us) | Round-Trip V1 (us) | V1 vs XLA | DT Plan vs XLA | Fusion Token % |
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

Key observations:

- Round-trip regression (+80.6%) and DT plan regression (+80.3%) are effectively identical -- the model is not the problem; the interface is.
- Even a hypothetical perfect model cannot do better than +80.6% through this interface.
- The per-module regression does not correlate with fusion-token fraction alone: `rg_lru` (69% fusion tokens) regresses +47%, while `griffin_ffn` (44% fusion tokens) regresses +224%. The impact depends on how critical the fusion-token ordering is for that module's graph dynamics.

### 5.2 Why every model improvement converged to the same GPU time

| Model Variant | Val Accuracy | GPU Time vs XLA |
|---|---|---|
| Lite mode (baseline) | 58.0% | +88.8% |
| Full mode (per-step GNN) | 67.6% | +88.7% |
| Hybrid mode (GNN every 5 steps) | 66.0% | +89.2% |
| gpu_progressive RTG | ~58% | +88.5% |
| P0 fix (repeat penalty + fusion strip) | ~58% | +88.4% |
| Larger model (128d, 6L) | 58% | not profiled |

The ~+89% regression is an interface constant, not a model-quality metric.

---

## 6. Diagnostic Patch: Cost-Model Fallback

To isolate the root cause we modified the `json_plan` strategy in `priority_fusion.cc` so that producers absent from the plan fall through to XLA's default cost-model priority computation (`time_unfused - time_fused`) instead of receiving priority 0.

```cpp
// BEFORE (V1): unknown producers get lowest priority
if (strategy == "json_plan") {
    auto it = plan.find(producer->name());
    if (it != plan.end()) {
        return absl::Microseconds(plan.size() - it->second);
    }
    return absl::Microseconds(0);  // <- the bug
}

// AFTER (V2 fallback): unknown producers use cost-model priority
if (strategy == "json_plan") {
    auto it = plan.find(producer->name());
    if (it != plan.end()) {
        return absl::Microseconds(plan.size() - it->second);
    }
    // Fall through to default cost-model computation below
} else {
    LOG(WARNING) << "Unknown strategy...";
}
// ... default cost-model code runs for fusion.N nodes
```

### 6.1 Round-trip V2 results

| Module | XLA Default (us) | V1 vs XLA | V2 (fallback) vs XLA |
|---|---|---|---|
| griffin_ffn | 28,506 | +224.4% | **+0.06%** |
| shared_block | 14,793 | +132.5% | **+0.01%** |
| gqa_qkv_proj | 24,102 | +145.4% | **+0.21%** |
| ffn_layer | 188,601 | +113.3% | **-0.50%** |
| routed_block | 402,544 | +65.1% | **-0.22%** |
| rg_lru | 20,492 | +46.6% | **+0.02%** |
| vision_mlp | 1,097 | +121.0% | **+0.05%** |
| mamba_block | 15,434 | +18.3% | **-0.37%** |
| **Weighted Avg** | | **+80.6%** | **~-0.1%** |

All modules reproduce XLA default within measurement noise (<0.5%). The interface bottleneck is eliminated.

### 6.2 DT plan profiling with V2 binary

| Module | XLA Default (us) | DT Plan V2 (us) | DT vs XLA |
|---|---|---|---|
| routed_block | 401,857 | 401,137 | -0.18% |
| shared_block | 14,791 | 14,793 | +0.01% |
| griffin_ffn | 28,448 | 28,540 | +0.33% |
| rg_lru | 20,493 | 20,493 | +0.00% |
| ffn_layer | 188,134 | 187,697 | -0.23% |
| gqa_qkv_proj | 24,119 | 23,942 | -0.73% |
| mamba_block | 15,432 | 15,460 | +0.18% |
| vision_mlp | 1,096 | 1,096 | -0.02% |
| **Weighted Total** | **694,370** | **693,159** | **-0.17%** |

DT plans now match XLA default within noise. The +80-224% regressions are gone.

---

## 7. Why the Fallback Is Not a Solution

The cost-model fallback is a useful diagnostic tool but **not a viable research contribution**, for two reasons:

### 7.1 Priority-scale mismatch makes the plan irrelevant

Plan priorities range from 1 to `plan.size()` microseconds (typically 1-80 us). XLA's cost-model priorities (`time_unfused - time_fused`) range from roughly 100 to 5,000+ us. Because the cost-model values are 10-100x larger, they dominate the priority queue for **all** producers -- both `fusion.N` nodes (which get cost-model priority by fallback) and original nodes (whose tiny plan priorities are negligible compared to their own cost-model values, which XLA would have computed anyway). The DT's ordering is effectively ignored; XLA runs its default cost-model schedule regardless of the plan.

This explains why V2 DT plans match XLA default exactly: the plan has no effect.

A subsequent experiment (V3) scaled plan priorities to a 1M base (`1,000,000 - position`) so that plan nodes always dequeue before any `fusion.N` node. Partial round-trip results on 4 modules showed <0.3% regression, suggesting the scaling preserves interface fidelity. However, this approach raises a deeper issue:

### 7.2 The method still depends on XLA's analytical cost model

Even with priority scaling, the fallback scheme delegates all `fusion.N` ordering to XLA's `time_unfused - time_fused` cost model. This cost model is an **analytical estimate** (GPU performance model based on operation shapes, memory bandwidth, and compute throughput). The learned model controls only the original-node ordering; roughly half of the fusion decisions are still handled by the same hand-crafted heuristic we aim to improve upon.

A publishable method should not rely on the analytical cost model as a crutch for dynamic nodes. The final system must be able to score both original producers and dynamically-created `fusion.N` producers through a single learned mechanism.

---

## 8. Implications for Next-Stage Design

### 8.1 What this diagnostic establishes

1. **The static plan interface is fundamentally lossy.** Any approach that maps a learned sequence to a static JSON `producer_ordering` will lose 46% of fusion decisions at the interface boundary.
2. **Model quality was never the bottleneck.** The +89% GPU regression was an interface artifact; every model variant (Lite, Full, Hybrid, larger architectures) converged to the same number because they all deployed through the same broken interface.
3. **Cost-model fallback restores fidelity but not controllability.** The DT's plan becomes invisible when cost-model priorities dominate.
4. **Dynamic participation is required.** A learned scheduler must operate inside XLA's fusion loop, scoring every producer (including `fusion.N`) at each step, rather than emitting a one-shot static plan before compilation begins.

### 8.2 Design requirements for the next stage

From the failure analysis, the replacement system must satisfy:

| Requirement | Rationale |
|---|---|
| Score `fusion.N` nodes | 46% of decisions involve dynamically-created producers |
| Run inside the fusion loop | The graph changes after every step; a static plan cannot anticipate these changes |
| Not depend on XLA's analytical cost model | The cost model is the baseline to beat, not a component to reuse |
| Generalize across modules | LOO experiments show 33-63% accuracy on unseen architectures; the new system should maintain or improve this |

### 8.3 What carries forward

Despite the interface failure, the following infrastructure and findings remain valuable:

- **Trajectory data**: 4,280 trajectories across 33 modules, 6 architectures, 176 strategies. These trajectories faithfully record dynamic fusion decisions including `fusion.N` steps.
- **Graph encoder (GraphSAGE)**: the GNN that encodes HLO graphs into per-node embeddings is independent of the deployment interface and can be reused.
- **Cost-model reward signal**: the dual reward (`us_unfused / us_fused` per step + GPU kernel time per trajectory) recorded in trajectories provides training signal for any learned priority function.
- **Patched XLA binary**: the env-var-controlled strategy dispatch in `priority_fusion.cc` can be extended to call an external learned model at each priority evaluation.

---

## Appendix: File References

| Artifact | Location |
|---|---|
| XLA C++ patch | `priority_fusion_patch.diff` |
| XLA source (GPU server) | `/root/rivermind-data/xla/xla/backends/gpu/transforms/priority_fusion.cc` |
| Round-trip V1 plans | `output/roundtrip_plans/` (GPU server) |
| Round-trip V2 profiles | `output/gpu_profiles_roundtrip_v2.csv` (GPU server) |
| DT V2 profiles | `output/gpu_profiles_dt_v2.csv` (GPU server) |
| Partial round-trip V3 profiles | `output/gpu_profiles_roundtrip_v3.csv` (GPU server) |
| Full/Hybrid profiling | `output/gpu_profiles_full_mode.csv`, `output/gpu_profiles_hybrid5_mode.csv` |
| DT plan generation | `dt_eval_multi.py` -> `actions_to_producer_names()` |
| Plan consumption | `CalculateProducerPriority()` -> `json_plan` branch |
| XLA build command | `bazel build --config=cuda --repo_env=HERMETIC_CUDA_COMPUTE_CAPABILITIES="sm_80" //xla/tools:run_hlo_module` |
