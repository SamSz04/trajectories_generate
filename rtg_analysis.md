# Dense Proxy RTG: Analysis and Design

## 1. The Problem: Flat RTG Kills Credit Assignment

In a Decision Transformer, Return-to-Go (RTG) serves as a **goal-conditioning signal**: it tells the model "this much reward remains to be collected." The DT learns to produce actions that achieve the specified remaining reward. If RTG is wrong or uninformative, the model cannot distinguish early critical decisions from late inconsequential ones.

### Current Implementation (3 modes)

| Mode | RTG formula | Training | Inference |
|---|---|---|---|
| `gpu` (default) | `R_t = gpu_speedup / max_gpu` for all t | Flat constant | Constant (`--target-rtg`) |
| `gpu_hybrid` | `R_t = (cost_rtg_t / cost_r0) × gpu_speedup / max_gpu` | Shaped: decays with cost-model cumsum | Constant (mismatch!) |
| `cost_model` | `R_t = Σ_{i=t}^{T} max(0, us_unfused_i - us_fused_i) / max_cost_r0` | Pure cost-model backward cumsum | Constant (mismatch!) |

**Key file locations:**
- Training RTG: `dt_dataset_multi.py:186-208`
- Model consumption: `dt_model.py:129` (`nn.Linear(1, embed_dim)`) → line 255 (`return_embed`)
- Greedy inference: `dt_eval_multi.py:110` (constant `target_rtg` every step)
- Beam inference: `dt_eval_multi.py:225-232` (linear decay `target_rtg * (1 - s/max_steps)`)

### Three fundamental failures:

**F1. Flat GPU RTG = zero temporal signal.** In `gpu` mode, every timestep gets the same RTG value. The model receives identical conditioning whether it's at step 1 (deciding which high-priority producer to fuse first) or step 100 (fusing a trivial leftover). This violates the core DT assumption that RTG should decay as rewards are collected.

**F2. Train/inference RTG mismatch.** Even when training uses `gpu_hybrid` (which decays), inference uses a flat constant. The model was trained to expect a specific decay curve — at inference it sees a completely different signal. This is analogous to training an image classifier on color images and testing on grayscale.

**F3. Cost-model RTG collapses to zero rapidly.** Empirical analysis shows:

| Module | Steps | RTG at 25% | RTG at 50% | RTG at 75% |
|---|---|---|---|---|
| Llama gqa_layer | 104 | 19.3% of initial | 8.0% | 0.4% |
| Griffin attention_block | 84 | 24.7% | 7.2% | 2.2% |
| SigLIP vision_attention | 13 | 100% | 74.7% | 29.7% |

For longer trajectories, the cost-model RTG becomes effectively zero in the second half. The model gets a near-zero RTG signal for ~50% of its decisions, making all late-trajectory actions indistinguishable from a conditioning perspective.

---

## 2. Empirical Evidence: The Zero-Reward Problem

Analysis of all 33 modules' trajectory data reveals pervasive sparsity in per-step cost-model rewards:

### 2.1 Zero-reward step frequency

| Module | Zero-reward steps | Impact |
|---|---|---|
| SigLIP/vision_attention | 39.9% | Many fusions have us_fused ≥ us_unfused |
| DeepSeek/moe_unpermute | 35.6% | Cost model sees no benefit |
| Mamba/mamba_conv_ssm | 33.8% | Despite large trajectory-level speedup |
| Griffin/attention_block | 29.1% | |
| Average across all modules | ~25% | 1 in 4 steps gives zero signal |

### 2.2 Catastrophic modules have degenerate cost-model signals

| Module | GPU result (V3) | Reward range | Zero-reward % | All-zero trajs |
|---|---|---|---|---|
| moe_permute | +87.6% worse | [-1140.9, 141.7] | High | Many |
| mamba_block | +7.1% worse | [-6778.0, 10950.6] | 33%+ | Some |
| mamba_conv_ssm | +7.3% worse | [-6784.7, 10950.6] | 33.8% | Some |

**For moe_permute, all initial priorities are 0.0** — the cost model provides literally zero signal about fusion ordering. Any RTG based on cost-model decomposition is identically zero for this module.

### 2.3 Negative rewards are clamped away

The current implementation uses `max(0.0, us_unfused - us_fused)`, discarding all negative rewards (fusions that increase cost). This means:
- Roughly 25% of steps have zero reward (post-clamp)
- The remaining rewards are concentrated in early high-priority fusions
- Late trajectory steps contribute near-zero to RTG

---

## 3. Literature: How Other DTs Handle Sparse/Delayed Rewards

### 3A. Q-learning Decision Transformer (QDT, Yamagata+ 2023)
- Train an offline Q-function (CQL/IQL) on the same data
- Use `R_t = V(s_t)` as RTG (learned value function replaces handcrafted RTG)
- **Pros**: Dense signal, learned from data, no cost-model dependency
- **Cons**: Requires training a separate critic; critic quality limited by same data

### 3B. Elastic Decision Transformer (EDT, Wu+ 2023)
- Allow variable-length history (elastic context)
- RTG is still trajectory-level but the model learns to adjust its own context window
- **Relevance**: Orthogonal to RTG design, but the insight that "not all history is equally useful" applies

### 3C. RUDDER (Arjona-Medina+ 2019) / ARES (attention-based)
- Train a return decomposition model: learn `r_t` such that `Σ r_t = R_final`
- Use LSTM or attention to attribute the terminal reward to individual steps
- **Pros**: Produces dense proxy reward from sparse terminal signal
- **Cons**: Requires training additional model; quality depends on decomposition accuracy

### 3D. Critic-Guided DT (CGDT, Wei+ 2024)
- Train value function V(s) or Q(s,a) on offline data
- Replace flat RTG with `R_t = Q(s_t, a_t)` or adjust RTG based on critic predictions
- **Pros**: State-dependent RTG; adapts to current trajectory state
- **Cons**: Same critic quality concerns as QDT

### 3E. Decoupled DT (DDT)
- Separate "which direction" from "how far" — decouple trajectory type from RTG magnitude
- Use categorical trajectory labels (good/medium/bad) instead of continuous RTG
- **Relevance**: Could classify trajectories by GPU speedup quartile

---

## 4. Design Options for FusionDT

### Option A: Fix the Mismatch (Minimal Change)

**What**: Keep `gpu_hybrid` training but fix inference to use matching decay.

**Implementation**:
```python
# In greedy_generate: decay RTG using cost-model shape
# Load the module's average cost-model decay curve from training data
# R_t = target_rtg * decay_curve[step]
```

**Pros**: Zero retraining needed, just fix inference RTG decay
**Cons**: Doesn't solve F1 (flat GPU mode) or F3 (cost-model collapse). Doesn't help modules with zero cost-model signal (moe_permute).
**Expected impact**: Small improvement for modules where gpu_hybrid was trained.

### Option B: Priority-Weighted Dense Reward (XLA-Native Signal)

**What**: Use XLA's actual priority queue scores as dense per-step rewards.

XLA maintains `us_unfused - us_fused` for every producer at every step. This is the exact signal XLA uses to decide fusion ordering. We already capture it in trajectories (`s["us_unfused"]`, `s["us_fused"]`).

**RTG formula**:
```
r_t = us_unfused_t - us_fused_t  (allow negatives!)
R_t = Σ_{i=t}^{T} r_i          (backward cumsum, not clamped)
R_t_normalized = R_t / max(|R_0|, 1.0)
```

**Key change**: Allow negative rewards. A fusion that increases cost (us_fused > us_unfused) should produce negative RTG, signaling "bad decisions ahead."

**Pros**: Uses XLA's actual decision signal; dense; non-zero for most steps
**Cons**: Still zero for moe_permute (all priorities are 0.0); rapid decay problem remains
**Expected impact**: Moderate — helps modules with good cost-model signal, doesn't help degenerate modules.

### Option C: GPU-Anchored Progressive Decay (Recommended for Training)

**What**: Create a dense RTG that (1) starts at the true GPU reward, (2) decays progressively, and (3) uses a smooth schedule that doesn't collapse to zero.

**RTG formula**:
```
# Step progress ratio
p_t = t / T

# Smooth decay (several options):
# Linear:     decay_t = 1 - p_t
# Sqrt:       decay_t = sqrt(1 - p_t)      ← slower early decay
# Cosine:     decay_t = cos(π/2 × p_t)     ← smooth S-curve
# Log-linear: decay_t = 1 - log(1 + p_t × (e-1)) / 1  ← gentle

R_t = gpu_reward × decay_t
```

**Why this works**:
1. Every step gets a non-zero RTG (no degenerate modules)
2. The decay communicates "how much reward remains" — early steps see high RTG, late steps see low
3. The GPU terminal reward anchors the scale, so the model learns trajectory quality discrimination
4. Unlike cost-model, this doesn't require per-step cost signals to be meaningful

**Pros**: Works for ALL modules including moe_permute; smooth decay; GPU-calibrated
**Cons**: The decay shape is assumed, not learned from data. Doesn't capture step-specific difficulty.
**Expected impact**: High — uniformly applicable, fixes F1 and partially fixes F3.

### Option D: Cost-Model-Shaped GPU Decay (Best of Both Worlds)

**What**: Use cost-model shape where available, fall back to smooth synthetic decay where not.

**RTG formula**:
```python
# Per-step cost-model reward (allow negatives)
r_t = us_unfused_t - us_fused_t

# Backward cumsum
cost_rtg = cumsum_backward(r_t)

if max(|cost_rtg|) > threshold:
    # Module has meaningful cost-model signal
    # Normalize cost_rtg to [0, 1] range, then scale by GPU reward
    cost_shape = cost_rtg / cost_rtg[0]
    R_t = gpu_reward × cost_shape
else:
    # Degenerate module (e.g., moe_permute) — use synthetic decay
    R_t = gpu_reward × sqrt(1 - t/T)
```

This is essentially `gpu_hybrid` with two improvements:
1. **Allow negative cost-model rewards** (don't clamp to 0)
2. **Fallback for degenerate modules** (synthetic decay when cost signal is zero)

**Pros**: Best available signal when present; graceful fallback; GPU-calibrated
**Cons**: Two different RTG shapes in the same training set (may confuse the model); threshold is a hyperparameter
**Expected impact**: High — addresses all three failures.

### Option E: Learned Return Decomposition (RUDDER-style)

**What**: Train a separate model to decompose the terminal GPU reward into per-step contributions.

**Architecture**: Small LSTM or 1-layer transformer that takes the trajectory (states + actions) and outputs per-step reward attributions summing to the terminal GPU reward.

```python
class ReturnDecomposer(nn.Module):
    def __init__(self, state_dim, hidden_dim=64):
        self.lstm = nn.LSTM(state_dim + 1, hidden_dim, batch_first=True)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, states, actions):
        # states: [B, T, D], actions: [B, T, 1]
        x = torch.cat([states, actions_embed], dim=-1)
        h, _ = self.lstm(x)
        rewards = self.head(h).squeeze(-1)  # [B, T]
        return rewards  # should sum to GPU terminal reward

    def loss(self, predicted_rewards, terminal_reward):
        # Sum constraint: predicted must sum to terminal
        pred_sum = predicted_rewards.sum(dim=-1)
        return F.mse_loss(pred_sum, terminal_reward)
```

**Training pipeline**:
1. Train ReturnDecomposer on (trajectory, GPU_reward) pairs
2. Use decomposed per-step rewards to compute dense RTG
3. Retrain DT with decomposed RTG

**Pros**: Learns the actual per-step credit from data; dense; principled
**Cons**: Adds training complexity; decomposition may be unstable with limited data (4,280 trajectories); potential circular dependency (need good decomposition to train good DT, need good DT to evaluate decomposition)
**Expected impact**: Potentially highest, but risky with current data size.

### Option F: Trajectory-Quality Conditioning (DDT-style)

**What**: Replace continuous RTG with discrete trajectory quality labels.

**Implementation**:
```python
# Discretize GPU speedup into quality tiers
quality_bins = [0.95, 0.98, 1.0, 1.02, 1.05]  # relative to xla_default
quality_label = digitize(gpu_speedup, quality_bins)  # 0-5

# Replace return_embed: nn.Linear(1, D) → nn.Embedding(6, D)
self.quality_embed = nn.Embedding(6, embed_dim)

# At inference: condition on label=5 ("best quality trajectory")
```

**Pros**: No decay needed; robust to reward scale; simple to implement
**Cons**: Loses granularity; 6-8 bins may not distinguish subtle quality differences; still flat conditioning (F1 remains)
**Expected impact**: Moderate — simpler but less expressive.

---

## 5. Recommendation: Phased Approach

### Phase 1 (Immediate, no retraining): Fix Inference RTG Decay

Modify `dt_eval_multi.py` to use matching decay at inference for the current `gpu_hybrid`-trained model:

```python
# Instead of constant target_rtg every step:
# Compute decay based on step progress
decay = math.sqrt(1.0 - step / max_steps)  # sqrt for gentler decay
current_rtg = target_rtg * decay
```

Also try linear decay and cosine decay. Profile all three against constant RTG.

**Effort**: ~10 lines changed. **Expected**: Small improvement from fixing train/inference mismatch.

### Phase 2 (Quick retrain): Option C or D

Implement Option C (GPU-anchored progressive decay) as new RTG mode `gpu_progressive`:

```python
# In dt_dataset_multi.py
elif reward_mode == "gpu_progressive":
    gpu_r = traj["reward_gpu"] / max_gpu_reward
    # Try cost-model shape first
    if cost_rtg[0] > threshold:
        rtg = [(cr / cost_r0) * gpu_r for cr in cost_rtg]
    else:
        # Synthetic sqrt decay for degenerate modules
        rtg = [gpu_r * math.sqrt(1.0 - t/T) for t in range(T)]
```

Match at inference:
```python
# In dt_eval_multi.py greedy_generate
if rtg_decay == "progressive":
    current_rtg = target_rtg * math.sqrt(1.0 - step / max_steps)
```

**Effort**: ~30 lines. Retrain: 2-3 hours on A100. **Expected**: Significant improvement, especially on degenerate modules.

### Phase 3 (Experimental): Option E (RUDDER decomposition)

Only if Phase 2 shows that RTG shape matters significantly. Train the decomposer as a side experiment.

---

## 6. Implementation Details for Phase 2

### 6.1 New RTG Mode: `gpu_progressive`

**File: `dt_dataset_multi.py`**

Add to the RTG computation block (after line 208):

```python
elif reward_mode == "gpu_progressive" and traj["reward_gpu"] is not None:
    gpu_r = traj["reward_gpu"] / max_gpu_reward
    # Use cost-model shape if signal is strong enough
    if cost_r0 > 1.0:  # threshold: at least 1 microsecond total cost-model reward
        rtg = [(cr / cost_r0) * gpu_r for cr in cost_rtg]
    else:
        # Synthetic progressive decay for degenerate modules
        rtg = [gpu_r * math.sqrt(1.0 - t / max(T - 1, 1)) for t in range(T)]
```

### 6.2 RTG Decay at Inference

**File: `dt_eval_multi.py`**

Add `--rtg-decay` CLI param with options: `constant` (current), `linear`, `sqrt`, `cosine`, `cost_model`.

In `greedy_generate`:
```python
# Replace line 110: all_rtg.append(target_rtg)
if rtg_decay == "constant":
    current_rtg = target_rtg
elif rtg_decay == "linear":
    current_rtg = target_rtg * (1.0 - step / max_steps)
elif rtg_decay == "sqrt":
    current_rtg = target_rtg * math.sqrt(1.0 - step / max_steps)
elif rtg_decay == "cosine":
    current_rtg = target_rtg * math.cos(math.pi / 2 * step / max_steps)
all_rtg.append(current_rtg)
```

### 6.3 Negative Cost-Model Rewards

**File: `dt_dataset_multi.py`**

Change line 187 from:
```python
rewards = [max(0.0, s["us_unfused"] - s["us_fused"]) for s in steps]
```
to:
```python
rewards = [s["us_unfused"] - s["us_fused"] for s in steps]  # allow negatives
```

This gives the model information about bad fusions, not just good ones.

### 6.4 Normalization

For `gpu_progressive`, normalization is straightforward:
- GPU reward is already normalized by `max_gpu_reward` (computed from training data)
- Decay curve is in [0, 1]
- Final RTG range: [0, 1] (assuming GPU speedup ≥ 0)

For negative cost-model rewards, normalization needs updating:
- Use `max(|cost_r0|)` across all trajectories as the normalizer
- RTG range becomes [-1, 1] approximately

---

## 7. Evaluation Plan

### Metrics to Track

1. **GPU kernel time** (primary): Compare DT plans vs XLA default on all 24 modules
2. **Rank correlation**: Spearman correlation between DT fusion order and XLA default order
3. **Per-module breakdown**: Especially catastrophic modules (moe_permute, mamba_block, attention_block)
4. **RTG utilization**: Does the model actually use RTG signal? (Ablation: random RTG at inference)

### Experiments

| Experiment | Training RTG | Inference RTG | Expected vs Current |
|---|---|---|---|
| Baseline | `gpu` (flat) | constant | Current result (+0.35%) |
| Fix mismatch only | `gpu_hybrid` | sqrt decay | Small improvement |
| Progressive (no neg) | `gpu_progressive` | sqrt decay | Moderate improvement |
| Progressive (with neg) | `gpu_progressive` + negatives | sqrt decay | Best improvement |
| Quality labels | DDT-style | top label | Moderate improvement |
| Ablation: random RTG | `gpu` | random | Should be worse (validates RTG usage) |

---

## 8. Key Insights Summary

1. **The #1 problem is flat GPU RTG**: Every step gets identical conditioning → no credit assignment. This is worse than having noisy-but-decaying RTG.

2. **Cost-model RTG collapses too fast**: For 100+ step trajectories, the second half gets near-zero RTG. The model treats all late decisions as equally unimportant.

3. **~25% of steps have zero cost-model reward**: Clamping negative rewards discards information. Negative rewards (bad fusions) are just as informative as positive ones.

4. **Degenerate modules exist**: moe_permute has ALL priorities = 0.0. Any cost-model-based RTG is identically zero. These modules need synthetic decay.

5. **Train/inference mismatch**: Even if training RTG is perfect, using constant RTG at inference undoes all the benefit.

6. **Recommended path**: `gpu_progressive` mode with cost-model shape fallback + matching inference decay. This addresses all three failures (F1, F2, F3) with minimal implementation complexity.
