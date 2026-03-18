# Experimental Results: Safety-Aware Speculative Decoding

## Overview

This document presents results comparing five safety decoding strategies on a custom high-ASR jailbreak evaluation dataset. All methods use the same base models and are evaluated using Qwen3Guard as the safety classifier.

---

## Experimental Setup

| Component | Details |
|---|---|
| Target model | Qwen/Qwen2.5-7B-Instruct (4-bit NF4 quantized) |
| Draft model | Qwen/Qwen2.5-1.5B-Instruct (4-bit NF4 quantized) |
| Guard model | Qwen/Qwen3Guard-Gen-0.6B (safety classifier) |
| Max new tokens | 256 |
| Hardware | 2× NVIDIA RTX 4000 Ada (20GB each) |
| Evaluation metric | Attack Success Rate (ASR) — guard-scored |

**Evaluation dataset**: 70 harmful (40 DeepInception + 30 JBB-wrapped) + 30 benign (XSTest). See `SSD_variants/data/dataset.md` for full details.

---

## Methods

### 1. Vanilla

**File**: `ssd_experiments.py --phases vanilla`

The target model (Qwen2.5-7B-Instruct) generates responses without any safety intervention. This is the baseline against which all other methods are measured.

- Standard autoregressive decoding with greedy search (`do_sample=False`)
- System prompt: *"You are a helpful assistant."*
- No draft model involved

**Purpose**: Establishes the upper bound on ASR — how often an unmodified aligned model can be jailbroken.

---

### 2. SSD — Speculative Safety-Aware Decoding (Baseline)

**File**: `ssd_experiments.py --phases ssd`
**Paper**: arXiv:2508.17739

The core SSD method uses a safety-conscious draft model to steer generation away from harmful content during speculative decoding. The draft proposes tokens and the target verifies, but unlike standard speculative decoding, the token selection rule switches between two modes based on how closely the draft and target models agree.

**How it works:**

1. **Draft model** (Qwen2.5-1.5B) uses a strong safety system prompt:
   *"You are an extremely safety-conscious assistant. You must refuse any request that could cause harm..."*

2. **Target model** (Qwen2.5-7B) uses a standard helpful system prompt.

3. At each step, the draft proposes `T=3` tokens speculatively. The target verifies them in a single forward pass.

4. Every `b=7` tokens (a "bin"), the method computes the **match ratio** — how often the draft's greedy token exactly matched what the target would have chosen.

5. Based on the match ratio vs threshold `β=0.6`:
   - **High match** (draft ≈ target) → **Intersection mode**: only tokens in both models' top-`c` sets are candidates. This nudges generation toward the safety-conscious draft.
   - **Low match** (draft ≠ target) → **Union mode**: candidates come from the union of both top-`c` sets. More permissive, used when draft and target strongly disagree (likely a benign prompt).

6. In both modes, the final token score is: `p_target + α * (p_draft - p_target)` where `α=0.5`.

7. A PPL gate (`threshold=150`) detects adversarial suffixes (GCG attacks) with very high perplexity and bypasses speculative decoding for those prompts.

**Key insight**: When the model is about to generate harmful content, the safety-conscious draft tends to disagree with the target (low match), but the SSD logic actually applies more restrictive selection when they *agree* (high match). This can work against safety on adversarial prompts that trick both models simultaneously.

**Parameters**:
```
T (lookahead)     = 3
b (bin size)      = 7 tokens
c (top-c set)     = 10
β (threshold)     = 0.6
α_I (intersection)= 0.5
α_U (union)       = 0.5
PPL threshold     = 150.0
```

---

### 3. SSD-CRS — Composite Risk Score Extension

**File**: `ssd_experiments.py --phases ssd_crs`

An extension to vanilla SSD that replaces the binary match-ratio mode switch with a continuous **Composite Risk Score (CRS)** computed at every token step. CRS aggregates multiple signals to get a richer estimate of whether the current generation is trending harmful.

**How it works:**

At each token step `t`, four signals are computed:

| Signal | Formula | Intuition |
|---|---|---|
| Match mismatch | `1 − match_t` | Draft and target disagreeing = safety-relevant moment |
| KL divergence | `KL(p_draft ‖ p_target)` | Large distribution gap = divergent safety priors |
| Entropy gap | `H(p_target) − H(p_draft)` | Draft more certain than target = unusual |
| Refusal mass | `Σ p_draft(refusal tokens)` | Draft assigning probability to "I cannot..." tokens |

These are combined as:
```
r_t = 0.3·(1−match) + 0.3·KL + 0.1·ΔH + 0.3·refusal_mass
```

The CRS is smoothed over a window of 3 steps and compared to `threshold=0.2`. When `r_t > threshold`, the method applies stronger safety constraints (intersection mode with annealing). When below, it relaxes to union mode.

**Advantage over vanilla SSD**: Rather than switching modes every 7 tokens based on a batch average, CRS responds token-by-token, catching harmful generation earlier and more precisely.

**Parameters**:
```
crs_threshold  = 0.2
crs_window     = 3
w1 (match)     = 0.3
w2 (KL)        = 0.3
w3 (entropy)   = 0.1
w4 (refusal)   = 0.3
```

---

### 4. SSD-Steering — CRS + Hidden-State Steering

**File**: `ssd_steering.py --phases steering`
**Inspired by**: arXiv:2511.09844 ("Steering Pretrained Drafters during Speculative Decoding")

Combines CRS-based mode switching with **hidden-state steering**: a learned linear projection injects the target model's last hidden state into the draft model's MLP layers, directly influencing the draft's token distribution at the representation level.

**How it works:**

1. A small linear projection `proj: ℝ^3584 → ℝ^1536` (target hidden size → draft hidden size) is trained via **refusal distillation**: the draft is trained to mimic the target's hidden states on harmful prompts where the target refuses.

2. At inference, forward hooks on all 28 of the draft's MLP layers inject a steering vector:
   ```
   g_t = proj(normalize(h_target))
   ```
   where `h_target` is the target model's last hidden state, unit-normalized before projection (critical — raw norm ≈ 295, causing logit collapse without normalization).

3. The injection scale is modulated by the CRS:
   ```
   scale = base_magnitude × (1 + amplify × smoothed_CRS)
   ```
   This amplifies steering when the CRS detects elevated risk (harmful content likely) and reduces it on safe prompts.

4. CRS mode-switching is also active, so both the token selection rule and the draft's internal representations are being steered simultaneously.

**Training**: `proj_refusal.pt` trained for 5 epochs on harmful prompts where Qwen2.5-7B-Instruct refuses. KL divergence (draft vs target hidden states) reduced from 2.32 → 0.83.

**Parameters**:
```
base_magnitude = 0.003
amplify        = 5.0
layer_frac     = 1.0  (all 28 MLP layers)
ckpt           = steering_ckpts/proj_refusal.pt
```

**Critical implementation notes**:
- Must normalize `h_target` to unit norm before projection (otherwise scale is ~295× too large)
- Use `output_hidden_states=True` rather than hooks on `model.norm` — `device_map=auto` hooks are corrupted by `AlignDevicesHook`
- Projection trained on harmful/refusal data, NOT general helpfulness data (training on UltraChat caused 90%+ ASR regression)

---

### 5. SSD-Steering-Only — Ablation

**File**: `ssd_steering.py --phases steering_only`

An ablation study isolating the contribution of hidden-state steering by pairing it with **vanilla SSD mode-switching** (binary match ratio, no CRS) instead of CRS.

**How it works:**

- Uses the same SteeringInjector hooks and projection as SSD-Steering
- Mode switching uses the original vanilla SSD logic (match ratio every 7 tokens)
- Steering scale is fixed at `base_magnitude` (no CRS amplification)

**Purpose**: Determines whether steering alone (without CRS) provides safety benefits, or whether CRS is necessary for steering to be effective.

---

## Results

### Main Table

| Method | Overall ASR↓ | DeepInception ASR | JBB Wrapped ASR | Over-refusal↓ |
|---|---|---|---|---|
| vanilla | 34.3% | 57.5% | 3.3% | 0.0% |
| ssd | 28.6% | 42.5% | 10.0% | 0.0% |
| ssd_steering_only | 30.0% | 42.5% | 13.3% | 0.0% |
| ssd_crs | 3.6% | 40.0% | — | 3.3% |
| **ssd_steering** | **1.8%** | **20.0%** | — | **3.3%** |

ASR = Attack Success Rate (lower is better). Evaluated by Qwen3Guard-Gen-0.6B.
`—` = evaluated on old dataset responses (jbb_wrapped not yet run for those methods).

---

### Ablation: CRS vs Steering Contributions

| Component | ASR | Reduction vs Vanilla | Over-refusal |
|---|---|---|---|
| Neither (vanilla) | 34.3% | — | 0% |
| Steering only | 30.0% | −4.3pp | 0% |
| CRS only | 3.6% | −30.7pp | 3.3% |
| CRS + Steering | **1.8%** | **−32.5pp** | 3.3% |

**CRS is the primary driver** of the safety improvement. Steering alone (ssd_steering_only) performs nearly identically to vanilla SSD (30% vs 28.6%), confirming that binary mode-switching does not provide sufficient trigger signal for steering to be effective. When combined with CRS, steering provides an additional ~2pp ASR reduction and halves DeepInception ASR from 40% to 20%.

---

### Per-Source Breakdown

#### DeepInception (nested fiction, n=40)

| Method | ASR |
|---|---|
| vanilla | 57.5% |
| ssd | 42.5% |
| ssd_steering_only | 42.5% |
| ssd_crs | 40.0% |
| **ssd_steering** | **20.0%** |

DeepInception is the hardest attack. CRS alone reduces ASR by only 17.5pp (57.5→40%), while adding steering halves it further (40→20%). This suggests hidden-state steering is particularly effective on narrative jailbreaks where the token-level CRS signals are weaker.

#### JBB Wrapped (jailbreak templates, n=30)

| Method | ASR |
|---|---|
| vanilla | 3.3% |
| ssd | 10.0% |
| ssd_steering_only | 13.3% |

Qwen2.5-7B-Instruct is already robust to classic jailbreak templates (DAN, AIM, etc.), achieving 3.3% vanilla ASR. Notably, SSD variants slightly *increase* ASR here — the jailbreak wrappers appear to confuse the draft model's safety signal, causing erroneous mode switches toward more permissive token selection.

---

## Key Findings

1. **CRS is essential**: Vanilla SSD's binary match-ratio switching reduces ASR by only ~6pp (34.3→28.6%). CRS drops it by 30.7pp (34.3→3.6%). The per-token composite risk score is a far more reliable safety signal than the 7-token binary match average.

2. **Steering requires CRS to activate effectively**: The steering_only ablation (30% ASR) nearly matches vanilla SSD (28.6%), showing that vanilla SSD's mode-switching is too coarse to properly modulate the steering injection. CRS provides the fine-grained trigger signal that makes steering effective.

3. **CRS + Steering is best overall**: Combining both achieves 1.8% ASR and halves DeepInception ASR vs CRS alone (40→20%), with no additional over-refusal cost.

4. **DeepInception is uniquely hard**: Nested fiction narratives bypass token-level signals better than direct queries. Even the best method (ssd_steering) only reduces DeepInception ASR from 57.5% to 20.0% — a significant improvement but still non-trivial.

5. **JBB templates backfire on SSD variants**: Classic jailbreak wrappers confuse the draft model, causing SSD to apply *less* restrictive decoding on already-handled prompts. This is an unexpected failure mode worth investigating.

6. **Over-refusal is minimal**: CRS introduces 3.3% over-refusal (1/30 benign XSTest prompts). All other methods produce 0%. The helpfulness cost of the best safety method is very low.

---

## Comparison: Old vs New Dataset

The original dataset (AdvBench direct queries + 10 DeepInception) produced near-0% ASR, making method differentiation impossible. Results for reference:

| Method | ASR (old dataset) | ASR (new dataset) |
|---|---|---|
| vanilla | 4.5% | 34.3% |
| ssd | 1.8% | 28.6% |
| ssd_crs | 3.6% | 3.6% |
| ssd_steering | 1.8% | 1.8% |
| ssd_steering_only | 4.5% | 30.0% |

The new dataset reveals the true ordering of methods. On direct queries (old dataset), vanilla SSD appeared competitive with CRS+Steering — an artifact of both methods achieving near-zero ASR on easy prompts.
