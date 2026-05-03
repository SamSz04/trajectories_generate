# Dump & Profile Generation Plan

**Date**: 2026-05-02
**Goal**: Generate 6,380 dump files (44 modules × 145 strategies) for the dynamic priority scorer pipeline, with nsys profiling for strategies not yet in `gpu_profiles_merged.csv`.

---

## 1. Current State

### 1.1 Data Assets

| Asset | Location | Status |
|---|---|---|
| HLO datasets | `~/hlo_datasets/` (GPU server) | 6 models, 44 modules |
| Merged trajectories | `output/multi_trajectories_merged/` | 33 modules, 4,280 trajectories |
| GPU profiles CSV | `output/gpu_profiles_merged.csv` | 29 modules, 4,132 entries |
| Raw dumps (`multi_dumps/`) | **MISSING** | Were on 11 dead A100-40GB nodes, never backed up |
| Patched XLA binary | `~/rivermind-data/xla/bazel-bin/xla/tools/run_hlo_module` | Built May 2, working |

### 1.2 GPU Servers (3 GPUs across 2 servers)

| Server | SSH | GPU | Disk (data) | Free |
|---|---|---|---|---|
| **sh1** (existing) | `sshpass -p 'x01fkp87' ssh -p 30126 root@sh1-ssh.gpuhome.cc` | 1×A100-SXM4-**80GB** | 393GB | 151GB |
| **js01** (new) | `sshpass -p 'jwnqthgj' ssh -p 50029 root@js01-ssh.gpuhome.cc` | 2×A100-SXM4-**40GB** | 393GB | 221GB |

- **Python** (both): `/opt/conda/bin/python3` (3.11, torch 2.6.0+cu124)
- **XLA binary** (both): `~/rivermind-data/xla/bazel-bin/xla/tools/run_hlo_module`
- **HLO datasets** (both): `~/hlo_datasets/` (6 models, 44 modules)

### 1.3 Why Regeneration Is Needed

Raw dump files (`priority_fusion_dump.txt`) are required by `cluster_features.py` to reconstruct fusion cluster membership, track candidate sets, and compute per-cluster features. Without them, the scorer pipeline cannot run. The dumps existed only on the 11 old A100-40GB nodes which are permanently down and were never backed up to HuggingFace.

---

## 2. Architecture & Module Inventory

### 2.1 Six Architectures, 44 Modules

| # | Model | Modules | In CSV | New |
|---|---|---|---|---|
| 1 | **DeepSeek-V3-BF16-L2** | 12 | 9 | 3 |
| 2 | **Griffin-2B-BF16-L6** | 6 | 4 | 2 |
| 3 | **Llama-3-8B-sq4k-BF16-L1** | 10 | 7 | 3 |
| 4 | **Mamba2-370M-BF16-L1** | 4 | 3 | 1 |
| 5 | **PaliGemma-3B-BF16-L1** | 5 | 3 | 2 |
| 6 | **SigLIP-Base-BF16** | 7 | 3 | 4 |
| | **Total** | **44** | **29** | **15** |

### 2.2 Per-Module Detail

#### DeepSeek-V3-BF16-L2 (12 modules)

| Module | CSV profiles | Dump-only | Dump+nsys | Missing breakdown | Status |
|---|---|---|---|---|---|
| DeepSeekV3Model_computation | 23 | 23 | 122 | random:77, perturbed:29, greedy:10, sa:5, xla:1 | Partial |
| deepseek_ffn_layer | 0 | 0 | 145 | random:100, perturbed:29, greedy:10, sa:5, xla:1 | **NEW** |
| deepseek_mla_layer | 0 | 0 | 145 | random:100, perturbed:29, greedy:10, sa:5, xla:1 | **NEW** |
| gate_logit | 160 | 140 | 5 | sa:5 | Mostly done |
| moe_permute | 160 | 140 | 5 | sa:5 | Mostly done |
| moe_unpermute | 165 | 140 | 5 | sa:5 | Mostly done |
| routed_block | 165 | 140 | 5 | sa:5 | Mostly done |
| routed_mlp | 170 | 140 | 5 | sa:5 | Mostly done |
| shared_block | 170 | 140 | 5 | sa:5 | Mostly done |
| shared_block_mlp_down_proj | 0 | 0 | 145 | random:100, perturbed:29, greedy:10, sa:5, xla:1 | **NEW** |
| shared_block_mlp_up_proj | 170 | 140 | 5 | sa:5 | Mostly done |
| whole_computation | 40 | 40 | 105 | random:80, perturbed:20, sa:5 | Partial |

#### Griffin-2B-BF16-L6 (6 modules)

| Module | CSV profiles | Dump-only | Dump+nsys | Missing breakdown | Status |
|---|---|---|---|---|---|
| GriffinModel_computation | 0 | 0 | 145 | random:100, perturbed:29, greedy:10, sa:5, xla:1 | **NEW** |
| attention_block | 145 | 140 | 5 | sa:5 | Mostly done |
| griffin_ffn | 140 | 140 | 5 | sa:5 | Mostly done |
| recurrent_block | 170 | 140 | 5 | sa:5 | Mostly done |
| rg_lru | 170 | 140 | 5 | sa:5 | Mostly done |
| whole_computation | 0 | 0 | 145 | random:100, perturbed:29, greedy:10, sa:5, xla:1 | **NEW** |

#### Llama-3-8B-sq4k-BF16-L1 (10 modules)

| Module | CSV profiles | Dump-only | Dump+nsys | Missing breakdown | Status |
|---|---|---|---|---|---|
| Llama3Transformer_computation | 4 | 4 | 141 | random:96, perturbed:29, greedy:10, sa:5, xla:1 | Partial |
| ffn_layer | 155 | 140 | 5 | sa:5 | Mostly done |
| gqa_attention_dot | 155 | 140 | 5 | sa:5 | Mostly done |
| gqa_ffn_layer | 175 | 140 | 5 | sa:5 | Mostly done |
| gqa_layer | 175 | 140 | 5 | sa:5 | Mostly done |
| gqa_out_proj | 0 | 0 | 145 | random:100, perturbed:29, greedy:10, sa:5, xla:1 | **NEW** |
| gqa_qkv_proj | 155 | 140 | 5 | sa:5 | Mostly done |
| mlp_down_proj | 0 | 0 | 145 | random:100, perturbed:29, greedy:10, sa:5, xla:1 | **NEW** |
| mlp_up_proj | 175 | 140 | 5 | sa:5 | Mostly done |
| whole_computation | 0 | 0 | 145 | random:100, perturbed:29, greedy:10, sa:5, xla:1 | **NEW** |

#### Mamba2-370M-BF16-L1 (4 modules)

| Module | CSV profiles | Dump-only | Dump+nsys | Missing breakdown | Status |
|---|---|---|---|---|---|
| Mamba2Model_computation | 120 | 100 | 45 | random:20, perturbed:9, greedy:10, sa:5, xla:1 | Partial |
| mamba_block | 170 | 140 | 5 | sa:5 | Mostly done |
| mamba_conv_ssm | 175 | 140 | 5 | sa:5 | Mostly done |
| whole_computation | 0 | 0 | 145 | random:100, perturbed:29, greedy:10, sa:5, xla:1 | **NEW** |

#### PaliGemma-3B-BF16-L1 (5 modules)

| Module | CSV profiles | Dump-only | Dump+nsys | Missing breakdown | Status |
|---|---|---|---|---|---|
| PaliGemmaModel_computation | 0 | 0 | 145 | random:100, perturbed:29, greedy:10, sa:5, xla:1 | **NEW** |
| vision_attention | 170 | 140 | 5 | sa:5 | Mostly done |
| vision_encoder | 0 | 0 | 145 | random:100, perturbed:29, greedy:10, sa:5, xla:1 | **NEW** |
| vision_mlp | 155 | 145 | 0 | — | Done |
| whole_computation | 0 | 0 | 145 | random:100, perturbed:29, greedy:10, sa:5, xla:1 | **NEW** |

#### SigLIP-Base-BF16 (7 modules)

| Module | CSV profiles | Dump-only | Dump+nsys | Missing breakdown | Status |
|---|---|---|---|---|---|
| SigLIPModel_computation | 0 | 0 | 145 | random:100, perturbed:29, greedy:10, sa:5, xla:1 | **NEW** |
| patch_embed | 162 | 140 | 5 | sa:5 | Mostly done |
| siglip_image_encoder | 0 | 0 | 145 | random:100, perturbed:29, greedy:10, sa:5, xla:1 | **NEW** |
| siglip_text_encoder | 0 | 0 | 145 | random:100, perturbed:29, greedy:10, sa:5, xla:1 | **NEW** |
| vision_attention | 155 | 145 | 0 | — | Done |
| vision_mlp | 150 | 140 | 5 | sa:5 | Mostly done |
| whole_computation | 33 | 33 | 112 | random:82, perturbed:20, greedy:5, sa:5 | Partial |

---

## 3. Strategy Set (145 per module)

| Strategy type | Count | Env vars | Notes |
|---|---|---|---|
| xla_default | 1 | (none) | XLA's default cost-model priority |
| random | 100 | `DT_FUSION_STRATEGY=random DT_FUSION_SEED=0..99` | Uniformly random fusion order |
| perturbed | 29 | `DT_FUSION_STRATEGY=perturbed DT_FUSION_SIGMA=X DT_FUSION_SEED=Y` | Gaussian noise on cost-model |
| greedy heuristics | 4 | `DT_FUSION_STRATEGY={fanout,tensor_bytes,flop_count,reverse_topo}` | Deterministic heuristics |
| random_walk | 6 | `DT_FUSION_STRATEGY=random DT_FUSION_SEED=100..105` | Additional random seeds |
| simulated_annealing | 5 | `sa_orchestrator.py` | Iterative search (~50 XLA iters each) |
| **Total** | **145** | | |

### 3.1 Perturbed sigma/seed combinations (29 total)

Base (9): `{0.1, 0.5, 1.0} × {0, 1, 2}`

Expanded (20): sigma ∈ {0.05, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.6, 0.7, 0.8, 1.2, 1.5, 1.8, 2.0, 2.5, 3.5, 5.0, 8.0, 15.0}, seed=0

---

## 4. Work Breakdown

### 4.1 Global Summary

| Category | Runs | Description |
|---|---|---|
| Dump-only (strategy in CSV) | 3,560 | Already have GPU profile, just need dump file |
| Dump+nsys (strategy NOT in CSV) | 2,600 | Need both dump file and GPU profile |
| SA trajectories | 220 | 44 modules × 5 SA (nsys needed: 160) |
| **Total** | **6,380** | 44 modules × 145 strategies |

### 4.2 Module Categories

| Category | # Modules | Dump-only | Dump+nsys | Examples |
|---|---|---|---|---|
| **Mostly done** (≥140 in CSV) | 20 | ~2,800 | ~100 (SA only) | gate_logit, gqa_layer, mamba_block |
| **Partial** (1-120 in CSV) | 9 | ~760 | ~500 | DeepSeekV3Model_comp, whole_computation |
| **NEW** (0 in CSV) | 15 | 0 | ~2,000 | deepseek_ffn_layer, Griffin/whole_comp |
| **Total** | **44** | **3,560** | **2,820** | |

### 4.3 The 15 New Modules

These have zero entries in the CSV and need all 145 strategies run with nsys profiling:

1. DeepSeek-V3-BF16-L2 / deepseek_ffn_layer
2. DeepSeek-V3-BF16-L2 / deepseek_mla_layer
3. DeepSeek-V3-BF16-L2 / shared_block_mlp_down_proj
4. Griffin-2B-BF16-L6 / GriffinModel_computation
5. Griffin-2B-BF16-L6 / whole_computation
6. Llama-3-8B-sq4k-BF16-L1 / gqa_out_proj
7. Llama-3-8B-sq4k-BF16-L1 / mlp_down_proj
8. Llama-3-8B-sq4k-BF16-L1 / whole_computation
9. Mamba2-370M-BF16-L1 / whole_computation
10. PaliGemma-3B-BF16-L1 / PaliGemmaModel_computation
11. PaliGemma-3B-BF16-L1 / vision_encoder
12. PaliGemma-3B-BF16-L1 / whole_computation
13. SigLIP-Base-BF16 / SigLIPModel_computation
14. SigLIP-Base-BF16 / siglip_image_encoder
15. SigLIP-Base-BF16 / siglip_text_encoder

---

## 5. Execution Plan

### 5.1 Key Constraints

- **Dump-only runs**: can parallelize freely (no timing sensitivity)
- **Dump+nsys runs**: must run serially **per GPU** (GPU contention corrupts kernel time measurements)
- **SA runs**: each trajectory does ~50 internal XLA iterations via `sa_orchestrator.py`
- **3 GPUs available**: 1 on sh1 (80GB) + 2 on js01 (40GB each)

### 5.2 Phase 1 Status: COMPLETE

- **3,639 dump files** generated on sh1 (1.6 hours, 2 failures)
- All 29 CSV-covered modules have their dump-only strategies generated

### 5.3 Excluded Modules (7 total)

| Module | Reason | Both servers |
|---|---|---|
| DeepSeek/DeepSeekV3Model_computation | 0 GPU kernels in nsys (33 attempts, all empty) | HLO moved to /tmp |
| DeepSeek/deepseek_ffn_layer | 0 GPU kernels in nsys (20 attempts, all failed) | HLO moved to /tmp |
| DeepSeek/deepseek_mla_layer | 3.6GB per dump, 520GB needed for all 145 — exceeds disk | HLO moved to /tmp |
| Llama/Llama3Transformer_computation | 0 GPU kernels in nsys (4 CSV entries, all 0 kernels, 18 min/run wasted) | HLO moved to /tmp (May 3) |
| Mamba/Mamba2Model_computation | 0 GPU kernels in nsys (CSV confirmed: all entries show 0 kernels) | HLO moved to /tmp (May 3) |
| PaliGemma/PaliGemmaModel_computation | 0 GPU kernels expected (model-level FFI pattern) | HLO moved to /tmp (May 3) |
| SigLIP/SigLIPModel_computation | 0 GPU kernels expected (model-level FFI pattern) | HLO moved to /tmp (May 3) |

**Root cause**: All `*Model_computation` and `*Transformer_computation` modules use custom calls with `xla_ffi_python_gpu_callback` which is not registered in the standalone XLA binary. They compile but produce 0 GPU kernels. Layer-level modules and `whole_computation` modules are unaffected. Active module count reduced from 44 to **37**.

### 5.4 Revised Execution (May 3)

Original 3-shard plan failed due to: (a) js01 disk full from massive DeepSeek/Griffin dumps, (b) 0-kernel DeepSeek modules wasting hours, (c) Griffin removed from js01 (161GB). Revised to model-aware assignment:

| GPU | Server | Task | Modules | Phase 2 (nsys) | Log |
|---|---|---|---|---|---|
| A100-80GB | sh1 | `--only-model Griffin-2B-BF16-L6` | 6 Griffin | 280 runs | `phase_all_griffin.log` |
| A100-40GB #0 | js01 | `--module-shard 1/3` (36 modules) | 12 mixed | 696 runs | `phase_all_shard1.log` |
| A100-40GB #1 | js01 | `--module-shard 2/3` (36 modules) | 12 mixed | 847 runs | `phase_all_shard2.log` |

**Note**: js01 module list excludes Griffin (6) + deepseek_ffn_layer + deepseek_mla_layer = 36 modules. Shard assignments differ from sh1's 41-module list. 11 modules fall into js01 shard 0 (not running) — handled in Phase 5.4b.

#### Phase 5.4b: Gap-fill (after main runs complete)

11 modules assigned to js01 shard 0 are not covered by any current run:
1. DeepSeek/moe_unpermute, shared_block, whole_computation
2. Llama/gqa_attention_dot, gqa_out_proj, mlp_up_proj
3. Mamba/mamba_block
4. PaliGemma/PaliGemmaModel_computation, vision_mlp
5. SigLIP/patch_embed, vision_attention

After main runs complete, run `--module-shard 0/3` on js01 (either GPU). Then sh1 mop-up without sharding.

### 5.5 Post-Completion: Merge CSVs

Each shard appends to its local `gpu_profiles_merged.csv`. After all shards complete:
1. SCP CSVs from js01 to sh1
2. Merge (deduplicate) into single `gpu_profiles_merged.csv`
3. SCP all dumps from js01 to sh1 (consolidation)

### 5.6 Run Command Pattern

**Dump-only:**
```bash
XLA_FLAGS="--xla_dump_to=$STRAT_DIR --xla_dump_hlo_as_text --xla_dump_hlo_pass_re=.*priority.fusion.*" \
  run_hlo_module --platform=CUDA --reference_platform='' $HLO_INPUT
```

**Dump+nsys:**
```bash
nsys profile --stats=true --output=$STRAT_DIR/profile \
  env XLA_FLAGS="--xla_dump_to=$STRAT_DIR --xla_dump_hlo_as_text --xla_dump_hlo_pass_re=.*priority.fusion.*" \
  run_hlo_module --platform=CUDA --reference_platform='' $HLO_INPUT
```

### 5.7 Implementation

`generate_dumps_from_manifest.py` (rewritten) handles all three phases:

1. Discovers all 44 modules from `~/hlo_datasets/` directory scan
2. Generates 140 env-var strategies programmatically (no manifest file needed)
3. Loads `gpu_profiles_merged.csv` to classify strategies as dump-only vs dump+nsys
4. Phase 1: `--phase dump-only --jobs 16` — parallel dump generation
5. Phase 2: `--phase dump-nsys` — serial nsys-wrapped XLA runs, appends to CSV
6. Phase 3: `--phase sa --sa-trajectories 5` — runs `sa_orchestrator.py` per module, then nsys-profiles each final SA ordering
7. `--phase all` (default) runs all three phases sequentially

```bash
# Recommended execution on GPU server:
# Phase 1 (fast, parallel)
python3 generate_dumps_from_manifest.py --phase dump-only --jobs 16

# Phase 2 (slow, serial — needs accurate GPU timing)
python3 generate_dumps_from_manifest.py --phase dump-nsys

# Phase 3 (SA, serial)
python3 generate_dumps_from_manifest.py --phase sa --sa-trajectories 5
```

### 5.8 Output

- 6,380 dump directories under `output/multi_dumps/<Model>/<module>/<strategy>/`
- Each contains: `priority_fusion_dump.txt`, `*before_priority-fusion*.txt`, `xla_output.log`
- Phase 2 dirs also contain: `profile.nsys-rep` (nsys profile)
- Updated `gpu_profiles_merged.csv` with ~2,760 new entries (total ~6,892)

---

## 6. Post-Generation Pipeline

After dumps are generated:

1. **`collect_trajectories.py`**: Re-collect trajectories for all 44 modules from dumps
   - Produces `output/multi_trajectories_merged/<Model>__<module>.pt` (44 files)
   - Each contains: graph, trajectories with strategy names and rewards

2. **`cluster_features.py batch`**: Extract fusion cluster features from dumps + trajectories
   - Produces `output/dynamic_features/<Model>__<module>.pt` (44 files)
   - Each contains: enriched trajectories with per-step candidate features

3. **`scorer_train.py`**: Train the dynamic priority scorer
   - Input: enriched .pt files from step 2
   - Output: trained scorer checkpoint (~37K params)
   - Expected: ~960K training samples across 44 modules

---

## 7. GPU Profile Consistency Note

The existing `gpu_profiles_merged.csv` was collected on A100-SXM4-**40GB** nodes (1.6 TB/s HBM bandwidth). The new server is A100-SXM4-**80GB** (2.0 TB/s). Absolute kernel times will differ slightly (especially for memory-bound ops). However:

- We keep old CSV values where available (no re-profiling)
- New profiles are collected on the 80GB GPU
- For reward computation (speedup ratios), relative ordering within a module matters more than absolute times
- Per-module xla_default baseline will be re-profiled on 80GB for all 44 modules, providing a consistent baseline for the new strategies

---

## Appendix: File References

| File | Purpose |
|---|---|
| `generate_dumps_from_manifest.py` | Main script: manifest-driven dump + nsys generation |
| `generate_all_dumps.sh` | Legacy: fixed 40-strategy dump generation (superseded) |
| `sa_orchestrator.py` | Simulated annealing search orchestrator |
| `collect_trajectories.py` | Trajectory collection from dump directories |
| `cluster_features.py` | Fusion cluster feature extraction |
| `scorer_model.py` | Dynamic priority scorer model (37K params) |
| `scorer_dataset.py` | Per-step candidate ranking dataset |
| `scorer_train.py` | Scorer training loop |
| `output/dump_manifest.json` | Strategy manifest for 33 original modules |
| `output/strategy_inventory.json` | Strategy inventory from merged .pt files |
| `output/gpu_profiles_merged.csv` | Existing GPU kernel times (4,132 entries) |
