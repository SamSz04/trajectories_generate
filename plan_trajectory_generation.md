# Plan: Trajectory Generation Test Pipeline (v2)

## Context

Generate 50 diverse fusion trajectories from a single Llama-3 8B GQA layer HLO graph. **All trajectories are generated through real XLA** — no Python fusion simulator. A single unified C++ patch to `priority_fusion.cc` (~145 LOC) supports all strategies via environment variables.

Test data at `trajectories_generate/hlo_dumps/llama3_8B_gqa_layer/`:
- `before_priority-fusion.hlo` (262 lines, 123 ENTRY instructions, rank-5 tensors, 6 GEMM fusions)
- `priority_fusion_dump.txt` (3,486 lines, 101 fusions → ~70 MDP steps)

**Reward signals**: XLA cost model scores (always logged in dump) + real H100 GPU profiling times.

---

## Architecture: Everything Through Real XLA

```
                            ┌─────────────────────────────────┐
                            │   Patched XLA (built once)      │
                            │   priority_fusion.cc + ~145 LOC │
                            └──────────┬──────────────────────┘
                                       │
           ┌───────────────────────────┼───────────────────────────┐
           │                           │                           │
    DT_FUSION_STRATEGY=          DT_FUSION_STRATEGY=         DT_FUSION_STRATEGY=
    random (seed=0..19)          perturbed (σ,seed)          json_plan (SA iter)
           │                           │                           │
           ▼                           ▼                           ▼
    ┌──────────────┐           ┌──────────────┐           ┌──────────────┐
    │ XLA compile  │           │ XLA compile  │           │ XLA compile  │
    │ + dump + run │  × 20     │ + dump + run │  × 9      │ + dump + run │  × 10
    └──────┬───────┘           └──────┬───────┘           └──────┬───────┘
           │                           │                           │
           ▼                           ▼                           ▼
    priority_fusion_dump.txt × 50  (each contains fusion steps + us_fused/us_unfused)
           │
           ▼
    ┌──────────────────────────────────────────┐
    │  dump_parser.py  (parse ALL 50 dumps)    │
    │  → trajectory steps, action masks,       │
    │    cost model scores, real GPU times      │
    └──────────────────┬───────────────────────┘
                       ▼
              output/trajectories.pt
```

**No Python simulation. No GO_fusion fusion logic. Every fusion decision comes from real XLA.**

---

## 50 Trajectories: Strategy Breakdown

All controlled by env vars on the patched XLA binary:

| # | Strategy | Env Vars | Count | Description |
|---|---|---|---|---|
| 1 | `xla_default` | (none — unpatched behavior) | 1 | XLA's original cost-model priorities |
| 2-21 | `random` | `DT_FUSION_STRATEGY=random DT_FUSION_SEED=0..19` | 20 | Hash-based deterministic random priorities |
| 22-30 | `perturbed` | `DT_FUSION_STRATEGY=perturbed DT_FUSION_SIGMA=σ DT_FUSION_SEED=s` | 9 | Cost-model + Gaussian noise (σ∈{0.1,0.5,1.0} × 3 seeds) |
| 31-40 | `greedy` | `DT_FUSION_STRATEGY={fanout,tensor_bytes,flop_count,reverse_topo,...}` | 10 | Heuristic orderings (5 strategies × 2 variants) |
| 41-50 | `json_plan` | `DT_FUSION_STRATEGY=json_plan DT_FUSION_PLAN=<path>` | 10 | SA-optimized orderings from `sa_orchestrator.py` |

---

## Files to Create

All files in `/Users/shizhan/Desktop/offlineRL_fusion/trajectories_generate/`:

### 1. `priority_fusion_patch.diff` — Unified XLA C++ Patch

Single patch to `xla/backends/gpu/transforms/priority_fusion.cc` (~145 LOC):

**Injection point:** `CalculateProducerPriority()` (line 553)

```
PRESERVED (unchanged):
  ├── Bitcast → +∞ priority
  ├── Constant → -∞ priority
  ├── CanFuseWithAllNonBitcastUsers() legality check
  ├── All 10 CanFuse() legality checks
  ├── DequeueNextProducer() queue logic
  ├── OnFusingInstruction() fusion execution
  ├── FusionProcessDumpProto logging
  └── Triton fusion path

ADDED:
  ├── Read DT_FUSION_STRATEGY env var
  ├── Always compute & log cost model (us_fused, us_unfused) for dual reward
  ├── Strategy switch: random, perturbed, fanout, tensor_bytes, flop_count,
  │   reverse_topo, json_plan
  ├── JSON plan reader (loaded once at PriorityFusionQueue construction)
  └── HashToGaussian helper for deterministic noise
```

Env vars:
- `DT_FUSION_STRATEGY` — strategy name (omit for XLA default)
- `DT_FUSION_SEED` — random seed (integer)
- `DT_FUSION_SIGMA` — noise sigma for perturbed strategy (float)
- `DT_FUSION_PLAN` — path to JSON ordering file (for json_plan strategy)

### 2. `dump_parser.py` — XLA Dump Parser (Universal)

Parses **any** `priority_fusion_dump.txt` (from any strategy) into trajectory data.

- Parse `fusion {}` events → extract `(producer_name, consumer_names, fusion_name)` per step
- Parse `update_priority {}` events → extract `us_fused`, `us_unfused` per node per step
- Parse `producer_ineligible {}` events → track skipped producers
- Group multi-consumer fusions of same producer into single MDP steps
- Reconstruct action masks from the fusion sequence (which producers were available at each step)
- Output per trajectory:
  ```python
  {
      "steps": [
          {
              "step_idx": int,
              "producer_name": str,
              "consumer_names": [str],
              "us_fused": float,
              "us_unfused": float,
              "was_fused": bool,       # True for fusion events, False for ineligible
          },
          ...
      ],
      "total_us_fused": float,        # sum of us_fused across all fusions
      "total_us_unfused": float,      # sum of us_unfused across all fusions
      "num_fusions": int,
      "num_ineligible": int,
  }
  ```

### 3. `sa_orchestrator.py` — Simulated Annealing Search

Python script that orchestrates SA using the `json_plan` strategy:

1. Parse XLA default dump to get initial producer ordering
2. For each SA iteration (100 iterations):
   a. Perturb the ordering (swap 2 random producers)
   b. Write JSON plan file: `{"producer_ordering": ["op1", "op2", ...]}`
   c. Run: `DT_FUSION_STRATEGY=json_plan DT_FUSION_PLAN=plan.json run_hlo_module ...`
   d. Parse resulting dump → extract `sum(us_unfused - us_fused)` as objective
   e. Metropolis acceptance: accept if better, or with probability `exp(-delta/T)`
   f. Cool temperature: `T *= 0.95`
3. Save top-10 orderings as the 10 SA trajectories

### 4. `xla_runner.sh` — Batch Execution Script (H100)

Runs all 50 strategy configurations sequentially:

```bash
#!/bin/bash
XLA_TOOL=/path/to/run_hlo_module
HLO_INPUT=hlo_dumps/llama3_8B_gqa_layer/before_priority-fusion.hlo
DUMP_BASE=output/dumps

# Strategy 1: XLA default (no env var)
XLA_FLAGS="--xla_dump_to=$DUMP_BASE/xla_default" \
  $XLA_TOOL --platform=CUDA $HLO_INPUT

# Strategies 2-21: Random (seeds 0-19)
for seed in $(seq 0 19); do
  DT_FUSION_STRATEGY=random DT_FUSION_SEED=$seed \
  XLA_FLAGS="--xla_dump_to=$DUMP_BASE/random_$seed" \
    $XLA_TOOL --platform=CUDA $HLO_INPUT
done

# Strategies 22-30: Perturbed (σ × seed combinations)
for sigma in 0.1 0.5 1.0; do
  for seed in 0 1 2; do
    DT_FUSION_STRATEGY=perturbed DT_FUSION_SIGMA=$sigma DT_FUSION_SEED=$seed \
    XLA_FLAGS="--xla_dump_to=$DUMP_BASE/perturbed_${sigma}_${seed}" \
      $XLA_TOOL --platform=CUDA $HLO_INPUT
  done
done

# Strategies 31-40: Greedy variants
for strategy in fanout tensor_bytes flop_count reverse_topo forward_topo \
                min_fanout small_tensors low_flops random_walk degree_weighted; do
  DT_FUSION_STRATEGY=$strategy \
  XLA_FLAGS="--xla_dump_to=$DUMP_BASE/greedy_$strategy" \
    $XLA_TOOL --platform=CUDA $HLO_INPUT
done

# Strategies 41-50: SA (run sa_orchestrator.py first, then use json_plan)
python sa_orchestrator.py --xla_tool=$XLA_TOOL --hlo_input=$HLO_INPUT \
  --dump_base=$DUMP_BASE --num_trajectories=10
```

### 5. `collect_trajectories.py` — Assemble Final Trajectories

After `xla_runner.sh` completes:

1. For each dump directory in `output/dumps/`:
   a. Parse the `priority_fusion_dump.txt` via `dump_parser.py`
   b. Extract trajectory steps + cost model scores
   c. Parse profiling results (if `--xla_hlo_profile` was used)
2. Build initial HLO graph representation:
   a. Parse HLO via GO_fusion's `parse_hlo_file()` → `HloModule`
   b. Build PyG graph via `build_graph()` + `encode_features()` (for DT state input)
3. Package into final trajectory format:
   ```python
   {
       "hlo_file": str,
       "hlo_module_name": "jit_forward_pure",
       "initial_graph": PyG.Data,      # GNN-encoded initial HLO graph
       "strategy": str,                 # e.g., "random_seed5"
       "steps": [
           {
               "step_idx": int,
               "producer_name": str,
               "producer_id": int,      # node index in PyG graph
               "consumer_names": [str],
               "us_fused": float,
               "us_unfused": float,
           },
           ...
       ],
       "reward_cost_model": float,      # sum(us_unfused) / sum(us_fused)
       "reward_real_gpu": float,        # t_unfused / t_fused (from profiling)
       "num_fusion_steps": int,
   }
   ```
4. Save all 50 trajectories to `output/trajectories.pt`
5. Print summary statistics (reward distribution, step counts, strategy comparison)

### 6. `build_xla.sh` — XLA Build Script

```bash
#!/bin/bash
# One-time setup on GPU machine (A100 or H100)
git clone https://github.com/openxla/xla.git && cd xla
git apply /path/to/priority_fusion_patch.diff

# For A100 (compute capability 8.0):
./configure.py --backend=CUDA --cuda_compute_capabilities="8.0"

# For H100 (compute capability 9.0):
# ./configure.py --backend=CUDA --cuda_compute_capabilities="9.0"

# For both (builds a fat binary):
# ./configure.py --backend=CUDA --cuda_compute_capabilities="8.0,9.0"

bazel build //xla/tools:run_hlo_module
```

---

## Reused Components (from GO_fusion — graph encoding only)

| Component | File | Function/Class | Used For |
|---|---|---|---|
| HLO Parser | `references/GO_fusion/src/hlo_parser/parser.py` | `parse_hlo_file()` | Parse HLO text → HloModule |
| IR Data Classes | `references/GO_fusion/src/hlo_parser/hlo_ir.py` | `HloModule`, `HloInstruction` | Data structures |
| Graph Builder | `references/GO_fusion/src/hlo_parser/graph_builder.py` | `build_graph()` | HLO → PyG Data |
| Feature Encoder | `references/GO_fusion/src/hlo_parser/feature_encoder.py` | `encode_features()` | Node feature vectors |

**NOT used:** `fusion_simulator.py`, `fusion_rules.py`, `performance_model.py`, `fusion_env.py` — all fusion logic comes from real XLA.

---

## Implementation Order

| Step | File | Where | Depends On |
|---|---|---|---|
| 1 | `dump_parser.py` | Local | — |
| 2 | `priority_fusion_patch.diff` | Local (write), H100 (apply) | — |
| 3 | `build_xla.sh` | H100 | Step 2 |
| 4 | `xla_runner.sh` | H100 | Step 3 |
| 5 | `sa_orchestrator.py` | H100 | Steps 1, 3 |
| 6 | `collect_trajectories.py` | H100 | Steps 1, 4, 5 |

**Steps 1-2 can be developed locally.** Steps 3-6 require H100 with CUDA + Bazel.

---

## Key Design Decisions

- **No Python simulation**: Every fusion decision comes from real XLA's `PriorityFusionQueue`
- **Single unified patch**: One ~145 LOC patch, all strategies via env vars, build once
- **Dual reward always logged**: Cost model runs for ALL strategies (even random) → `us_fused`/`us_unfused` always in dump
- **Legality untouched**: All 10 `CanFuse()` checks, bitcast/constant handling, Triton path preserved
- **SA uses real XLA objective**: SA objective = `sum(us_unfused - us_fused)` from real XLA cost model
- **GO_fusion used ONLY for graph encoding**: `parse_hlo_file()` + `build_graph()` + `encode_features()` — never for fusion simulation

---

## Verification

1. **Dump parser**: Correctly extracts ~70 MDP steps from the existing default dump (3,486 lines)
2. **Patch correctness**: `DT_FUSION_STRATEGY` unset → identical behavior to unpatched XLA (dump should match)
3. **Strategy diversity**: 50 dumps show meaningfully different fusion orderings and cluster counts
4. **Cost model logging**: Every dump contains `us_fused`/`us_unfused` for all `update_priority` events
5. **Reward sanity**: All `reward_cost_model > 1.0`; XLA default should rank among the best
6. **Trajectory format**: `torch.load("output/trajectories.pt")` returns list of 50 dicts with expected fields
7. **SA convergence**: SA objective improves over iterations (plotted in summary)

---

## Estimated Timeline

| Phase | Task | Time |
|---|---|---|
| Local dev | `dump_parser.py` | 1 day |
| Local dev | `priority_fusion_patch.diff` | 1 day |
| H100 setup | Build XLA from source | 1-2 hours |
| H100 run | 50 strategy runs (`xla_runner.sh`) | ~30 min |
| H100 run | SA search (100 iterations × 10 restarts) | ~2 hours |
| H100 run | `collect_trajectories.py` | ~10 min |
| **Total** | | **~2 days local + half day H100** |
