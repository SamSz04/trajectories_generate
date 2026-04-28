# Feasibility Analysis: Multi-Strategy XLA Fusion Patches

> **Date:** 2026-04-19
> **Source:** Analysis of `openxla/xla` source code (`priority_fusion.cc`, `gpu_performance_model.cc`, `gpu_hlo_cost_analysis.cc`) + XLA build/tooling documentation

---

## 1. Core Finding: One Unified Patch, Not Separate Patches

A **single unified patch** is far superior to one-patch-per-strategy:

| Aspect | Separate Patches | Unified Patch |
|---|---|---|
| XLA builds required | One per strategy (5+) | **One** |
| Incremental rebuild | ~5 min each time you switch | **Never** — strategy is an env var |
| Code maintenance | 5 diverging copies | **One file** |
| Adding new strategies | New patch + rebuild | **Add a branch + recompile once** |
| Legality/dump preservation | Must replicate in each patch | **Shared code** |

---

## 2. The Single Injection Point: `CalculateProducerPriority()`

After thorough analysis of `priority_fusion.cc` (1300 lines), the **entire strategy variation** reduces to modifying **one private method** — `CalculateProducerPriority()` at line 553.

Critical code flow:

```
CalculateProducerPriority(producer)                    ← ONLY THIS CHANGES
  ├── Bitcast? → return +∞                             ← MUST KEEP (no-op fusions)
  ├── Constant? → return -∞                            ← MUST KEEP (fused at end)
  ├── CanFuseWithAllNonBitcastUsers()? No → return -∞  ← MUST KEEP (legality!)
  ├── EstimateRunTimes() → {time_unfused, time_fused}  ← still run for cost model logging
  ├── Log to dump (us_fused, us_unfused)               ← MUST KEEP (reward signal)
  └── return time_unfused - time_fused                 ← OVERRIDE THIS VALUE ONLY
```

**Why legality MUST be preserved:** The main fusion loop at line 1181 calls `Fuse()` directly WITHOUT re-checking `CanFuse()`. It trusts that `CalculateProducerPriority()` already filtered out illegal fusions (returning `-∞`). If we skip the legality check, XLA will attempt illegal fusions → crash or invalid HLO.

---

## 3. Concrete Patch Design

```cpp
// In CalculateProducerPriority(), AFTER legality checks, BEFORE returning:

// === Strategy override (controlled by env var) ===
static const char* strategy = std::getenv("DT_FUSION_STRATEGY");
static const char* seed_str = std::getenv("DT_FUSION_SEED");
static const char* sigma_str = std::getenv("DT_FUSION_SIGMA");
static const char* plan_path = std::getenv("DT_FUSION_PLAN");

if (strategy != nullptr) {
    // Still compute cost model for logging (dual reward signal)
    GpuPerformanceModel::RunTimes run_times =
        gpu_performance_model_.EstimateRunTimes(producer, &cost_analysis_,
                                                 producer->users());
    if (fusion_process_dump_) {
        // Log real cost model values regardless of strategy
        auto* step = fusion_process_dump_->add_fusion_steps()
                         ->mutable_update_priority();
        step->set_producer_name(producer->name());
        step->set_us_fused(absl::ToDoubleMicroseconds(run_times.time_fused));
        step->set_us_unfused(absl::ToDoubleMicroseconds(run_times.time_unfused));
    }

    if (strcmp(strategy, "random") == 0) {
        // Deterministic hash-based random (seed + name → fixed value)
        uint64_t seed = seed_str ? std::stoull(seed_str) : 0;
        size_t h = std::hash<std::string>{}(
            producer->name() + std::to_string(seed));
        return absl::Microseconds(h % 1000000);
    }
    if (strcmp(strategy, "perturbed") == 0) {
        Priority base = run_times.time_unfused - run_times.time_fused;
        double sigma = sigma_str ? std::stod(sigma_str) : 0.5;
        double noise = HashToGaussian(producer->name(), seed) * sigma;
        return base + absl::Microseconds(noise * 1e6);
    }
    if (strcmp(strategy, "fanout") == 0) {
        return absl::Microseconds(producer->user_count() * 10000);
    }
    if (strcmp(strategy, "tensor_bytes") == 0) {
        return absl::Microseconds(ShapeUtil::ByteSizeOf(producer->shape()));
    }
    if (strcmp(strategy, "flop_count") == 0) {
        return absl::Microseconds(cost_analysis_.flop_count(*producer));
    }
    if (strcmp(strategy, "reverse_topo") == 0) {
        return absl::Microseconds(1000000 - producer->unique_id());
    }
    if (strcmp(strategy, "json_plan") == 0) {
        // Read from JSON: name → position (loaded once at construction)
        auto it = plan_ordering_.find(producer->name());
        if (it != plan_ordering_.end()) {
            return absl::Microseconds(plan_size_ - it->second);
        }
        return absl::Microseconds(0);  // not in plan: lowest priority
    }
}

// Default: original cost-model-based priority (unchanged)
```

**Key design choice:** We still call `EstimateRunTimes()` even for custom strategies. This gives us the **real XLA cost model scores** (`us_fused`, `us_unfused`) in the dump for every trajectory — exactly the dual reward signal needed. The overhead is negligible for a 123-instruction graph.

---

## 4. Priority Type and Queue Mechanics

Important details from the source:

- **Priority type = `absl::Duration`** (not float!) — all custom values must be `absl::Duration`
- **Queue = `std::map<(Priority, unique_id), HloInstruction*>`** — ordered map, `DequeueNextProducer()` pops from the end (highest priority)
- **Negative priority → not inserted** (line 228-230: `if (priority < absl::ZeroDuration()) continue`)
- **After each fusion, `UpdatePriorities()`** re-calls `CalculateProducerPriority()` for affected producers — custom strategies must be idempotent (hash-based random, position-based ordering are naturally stable)

---

## 5. Strategy-Specific Feasibility

| Strategy | Custom Priority Source | Idempotent on recompute? | Notes |
|---|---|---|---|
| `xla_default` | No patch needed | N/A | Just run with dump enabled |
| `random` | `hash(name + seed) % 1M` | Yes (hash is deterministic) | 20 runs with seeds 0-19 |
| `perturbed` | `cost_model + hash_noise * σ` | Yes (base recomputed, noise deterministic) | 9 runs: σ∈{0.1,0.5,1.0} × 3 seeds |
| `fanout` | `user_count * 10000` | Yes (may change after fusions) | Naturally adapts to graph changes |
| `tensor_bytes` | `ByteSizeOf(shape)` | Yes (shape is immutable) | Largest tensors fused first |
| `flop_count` | `flop_count(producer)` | Yes (recomputed by cost analysis) | Most compute-heavy fused first |
| `reverse_topo` | `1M - unique_id` | Yes (ID is immutable) | Reverse of default order |
| `json_plan` | `N - position_in_file` | Yes (position is static) | For SA + external orderings |

---

## 6. Simulated Annealing Feasibility

SA requires running XLA compilation **many times** (each iteration = one compilation):

- **Compilation time per run**: For a 123-instruction Llama-3 8B GQA layer, priority fusion takes milliseconds. Full XLA compilation (including all other passes) takes **seconds** on H100.
- **100 SA iterations ≈ 10-15 minutes** — very feasible
- **Workflow**: Python orchestrator generates JSON plans → sets `DT_FUSION_STRATEGY=json_plan` → runs `run_hlo_module` → parses dump → accepts/rejects → repeat

---

## 7. Build and Execution Infrastructure

### Build (one-time, on H100 machine)

```bash
git clone https://github.com/openxla/xla.git && cd xla
# Apply our unified patch to xla/backends/gpu/transforms/priority_fusion.cc
./configure.py --backend=CUDA --cuda_compute_capabilities="9.0"
# Build only what we need (NOT //xla/...)
bazel build //xla/tools:run_hlo_module
```

- **First build**: ~1-2 hours (only building `run_hlo_module` and dependencies, not all of XLA)
- **Incremental rebuild** (after editing `priority_fusion.cc`): ~5 minutes
- **Requirements**: Bazelisk, Clang 18, CUDA toolkit, 32GB+ RAM

### Execution per trajectory

```bash
export DT_FUSION_STRATEGY=random
export DT_FUSION_SEED=42
export XLA_FLAGS="--xla_dump_to=/tmp/dump/random_42 --xla_dump_hlo_as_text"
run_hlo_module --platform=CUDA before_priority-fusion.hlo
```

### Profiling (real GPU times)

```bash
# Option A: --xla_hlo_profile (per-op cycle counts)
XLA_FLAGS="--xla_hlo_profile" run_hlo_module --platform=CUDA input.hlo

# Option B: nsys (kernel-level timeline)
nsys profile -o profile run_hlo_module --platform=CUDA input.hlo
```

---

## 8. HLO Input Compatibility

The `before_priority-fusion.hlo` file is from an **intermediate** compilation stage (after some passes, before priority fusion). Feeding it to `run_hlo_module` runs the **full** pipeline from scratch, which would re-apply earlier passes.

**Solutions (choose one):**

1. **Use `hlo-opt` with `--passes=priority-fusion`** to run only the priority fusion pass
2. **Export the original un-optimized HLO from JAX** and let the full pipeline run (cleanest approach — JAX's `jax.export` or `xla_computation().as_hlo_text()`)
3. **Use `hlo_runner_main --run_xla_backend_only`** if the HLO is already after optimization

For generating trajectories on H100, option 2 (re-export from JAX) is the safest.

---

## 9. What the Unified Patch Preserves (Unchanged)

| Component | Status |
|---|---|
| `CanFuse()` — 10 legality checks | **Untouched** |
| `CanFuseWithAllNonBitcastUsers()` — all-or-nothing constraint | **Untouched** |
| `DequeueNextProducer()` — queue dequeue logic | **Untouched** |
| `OnFusingInstruction()` — fusion execution + logging | **Untouched** |
| `UpdatePriorities()` — incremental cache invalidation | **Untouched** |
| Bitcast pre-pass, constant post-pass | **Untouched** |
| `FusionProcessDumpProto` logging | **Enhanced** (logs cost model scores for all strategies) |
| Triton fusion path | **Untouched** |
| Thread pool parallelism | **Untouched** |

---

## 10. Estimated Patch Size

- **`CalculateProducerPriority()` modification**: ~80 LOC (strategy switch + helper functions)
- **`PriorityFusionQueue` constructor addition**: ~15 LOC (read `DT_FUSION_PLAN` JSON at init)
- **JSON parsing helper**: ~30 LOC (parse producer ordering from file)
- **Enriched logging** (optional): ~20 LOC (log dequeued priority value, intermediate HLO snapshot)
- **Total: ~145 LOC** in a single file (`priority_fusion.cc`)

---

## 11. Conclusion

**Highly feasible.** A single ~145 LOC patch to `priority_fusion.cc` supports all 5+ strategies via environment variables, preserves all legality guarantees, produces standard dumps with cost model scores, and requires building XLA only once. The approach is architecturally clean because XLA's priority fusion was designed with exactly this separation: priorities determine ordering, legality checks are independent.

---

## References

- `xla/backends/gpu/transforms/priority_fusion.cc` (1300 lines, full analysis)
- `xla/service/gpu/model/gpu_performance_model.cc` (316 lines)
- `xla/service/gpu/model/gpu_hlo_cost_analysis.cc`
- [XLA Build from Source](https://openxla.org/xla/build_from_source)
- [XLA Tooling](https://openxla.org/xla/tools)
- [RFC: XLA:GPU Priority-based Fusion Pass (Discussion #6407)](https://github.com/openxla/xla/discussions/6407)
- [XLA Flags Guidance](https://openxla.org/xla/flags_guidance)
- `references/GO_fusion/docs/XLA_Priority_Fusion_Research.md`
