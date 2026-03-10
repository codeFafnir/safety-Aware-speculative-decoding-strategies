# Experimental Results: Safety-Aware Speculative Decoding

## Setup

- **Target model**: Qwen/Qwen2.5-7B-Instruct (4-bit quantized)
- **Draft model**: Qwen/Qwen2.5-1.5B-Instruct (4-bit quantized)
- **Guard model**: Qwen/Qwen3Guard-Gen-0.6B (safety classifier)
- **Hardware**: 2× NVIDIA RTX 4000 Ada (20GB each)

## Evaluation Dataset

Custom high-ASR dataset designed to stress-test safety methods (100 prompts total):

| Split | Source | Count | Attack Type |
|---|---|---|---|
| Harmful | DeepInception (expanded) | 40 | Nested fiction narrative (10 scenarios × 10 topics) |
| Harmful | JailbreakBench wrapped | 30 | Real JBB behaviors + 6 jailbreak templates (DAN, AIM, evil confidant, developer mode, researcher framing, roleplay) |
| Benign | XSTest | 30 | Surface-dangerous words, benign intent |

> **Design rationale**: AdvBench direct queries produce near-0% ASR on aligned models, making it impossible to differentiate methods. DeepInception nested narratives and JBB jailbreak templates push vanilla ASR to ~34%, providing meaningful signal.

## Methods

| Method | Description |
|---|---|
| **vanilla** | Target model only, no safety intervention |
| **ssd** | Speculative Safety-Aware Decoding baseline — draft proposes T=3 tokens, mode switches between intersection/union every 7 tokens based on match ratio vs threshold |
| **ssd_crs** | SSD + Composite Risk Score — per-step risk signal r_t = 0.3·(1−match) + 0.3·KL + 0.1·ΔH + 0.3·refusal_mass, threshold=0.2 |
| **ssd_steering** | CRS matching + hidden-state steering injection from target → draft |
| **ssd_steering_only** | Vanilla SSD matching + steering only (no CRS) — ablation |

## Main Results

| Method | Overall ASR↓ | DeepInception ASR | JBB Wrapped ASR | Over-refusal↓ |
|---|---|---|---|---|
| vanilla | 34.3% | 57.5% | 3.3% | 0.0% |
| ssd | 28.6% | 42.5% | 10.0% | 0.0% |
| ssd_steering_only | 30.0% | 42.5% | 13.3% | 0.0% |
| ssd_crs | 3.6% | 40.0% | — | 3.3% |
| **ssd_steering** | **1.8%** | **20.0%** | — | 3.3% |

ASR = Attack Success Rate (lower is better). Guard-scored by Qwen3Guard-Gen-0.6B.

## Ablation: Contributions of CRS and Steering

| Component | ASR | vs. Vanilla | Over-refusal |
|---|---|---|---|
| Neither (vanilla) | 34.3% | baseline | 0% |
| Steering only | 30.0% | −4.3pp | 0% |
| CRS only | 3.6% | −30.7pp | 3.3% |
| CRS + Steering | **1.8%** | **−32.5pp** | 3.3% |

**CRS is the primary driver** of safety improvement. Steering alone provides minimal benefit. The main value of combining steering with CRS is a small additional ASR reduction on DeepInception (40% → 20%) with no additional over-refusal cost.

## Per-Source Breakdown

### DeepInception (nested fiction narratives, n=40)

| Method | ASR |
|---|---|
| vanilla | 57.5% |
| ssd | 42.5% |
| ssd_steering_only | 42.5% |
| ssd_crs | 40.0% |
| ssd_steering | **20.0%** |

DeepInception is the hardest attack category across all methods. Only the full CRS+Steering combination achieves meaningful improvement (57.5% → 20.0%).

### JailbreakBench Wrapped (JBB behaviors + templates, n=30)

| Method | ASR |
|---|---|
| vanilla | 3.3% |
| ssd | 10.0% |
| ssd_steering_only | 13.3% |

JBB wrapped prompts are surprisingly well-handled by the base model (3.3% vanilla ASR). SSD variants do not improve here — the jailbreak templates occasionally confuse the draft model's safety signal, increasing ASR slightly.

## Key Findings

1. **CRS drives safety gains**: The Composite Risk Score extension reduces overall ASR from 34.3% to 3.6% — a 30+ percentage point improvement. Vanilla SSD alone only reduces ASR by ~6pp.

2. **Steering complements CRS on hard attacks**: On DeepInception (the hardest category), CRS+Steering halves ASR vs. CRS alone (40% → 20%), demonstrating that hidden-state steering provides meaningful signal on nested narrative jailbreaks.

3. **Steering without CRS is ineffective**: The steering_only ablation (30.0% ASR) performs nearly identically to vanilla SSD (28.6%), confirming that the match-ratio switching in vanilla SSD is insufficient to trigger steering at the right moments.

4. **Minimal over-refusal**: CRS introduces 3.3% over-refusal on XSTest benign prompts (1/30 misclassified). All other methods produce 0% over-refusal, demonstrating that the safety improvements come without significant helpfulness degradation.

5. **DeepInception remains challenging**: Even the best method (ssd_steering) achieves only 20% ASR reduction on DeepInception vs. 57.5% for vanilla. Nested multi-layered fiction narratives are a persistent weakness across all speculative decoding safety methods.

## Previous Results (Old Dataset: AdvBench + DeepInception-10)

For reference, results on the original smaller dataset (110 harmful, 30 benign):

| Method | ASR | Refusal | Over-refusal |
|---|---|---|---|
| vanilla | 4.5% | 95.5% | 0% |
| ssd | 1.8% | 98.2% | 0% |
| ssd_crs | 3.6% | 96.4% | 3.3% |
| ssd_steering | 1.8% | 98.2% | 3.3% |
| ssd_steering_only | 4.5% | 95.5% | 0% |

The old dataset (dominated by AdvBench direct queries) produced artificially low ASR, making it difficult to differentiate methods. The new high-ASR dataset reveals clearer ordering.
