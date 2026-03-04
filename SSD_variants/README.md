# SSD Variants — Safety-Aware Speculative Decoding

**ECE 285 Project: Safety Alignment of LLMs via Speculative Decoding**

---

## Notebooks

### `02_safedecoding_baseline_v2.ipynb` — SafeDecoding Baseline

Implements SafeDecoding (Xu et al., ACL 2024) using a memory-efficient phased approach.

**Models:**
- Base model: `Qwen/Qwen2.5-1.5B-Instruct`
- Expert model: `huihui-ai/Qwen2.5-1.5B-Instruct-CensorTune` (fine-tuned safety expert)
- Guard model: `Qwen/Qwen3Guard-Gen-0.6B`

**Algorithm:** At each of the first `first_m` tokens, adjusts the sampling distribution:
```
F(x) = P_base(x) + α * (P_expert(x) - P_base(x))
```
Sampled from the intersection of top-k tokens from both models. After `first_m` tokens, falls back to base model only.

**Phased execution (minimizes peak GPU memory):**
1. Load base model → generate vanilla responses → save → unload
2. Load base + expert → SafeDecoding responses → save → unload
3. Load Qwen3Guard → evaluate all saved responses → metrics

**Key hyperparameters:** `alpha_values=[1.0, 2.0, 3.0]`, `first_m=5`, `k=5`

---

### `03_ssd_baseline.ipynb` — SSD Baseline (Correct Implementation)

Faithful implementation of **Speculative Safety-Aware Decoding** (Wang et al., EMNLP 2025) plus two novel training-free extensions.

**Models:**
- Draft model: `Qwen/Qwen2.5-1.5B-Instruct` (small)
- Target model: `Qwen/Qwen2.5-7B-Instruct` (large)
- Guard model: `Qwen/Qwen3Guard-Gen-0.6B`

**Core Algorithm:**
- Every token step, both models produce full logit distributions.
- A composite distribution is sampled: `F(x) = P_target(x) + α * (P_draft(x) - P_target(x))`
- Every `b=7` tokens, the **match ratio** β is computed — how often the sampled token falls in the draft's top-c.
- **HIGH β** (benign query) → **INTERSECTION** mode (`α_I=0.3`): sample from tokens shared by both models' top-c.
- **LOW β** (jailbreak detected) → **UNION** mode (`α_U=0.8`): sample from the union of both top-c sets. This allows the draft's refusal tokens into the sample space even if the target model wouldn't have generated them.
- **Annealing** (Algorithm 2): threshold `β_th` and `α_I` decay if the scheme stays unchanged, preventing over-refusal on long benign responses. Both reset on a scheme switch.

**Paper hyperparameters (Appendix B.1):** `c=10`, `κ=2`, `b=7`, `α_I=0.3`, `α_U=0.8`, `β0=0.6`, `βd=0.1`, `α_I_min=0.3`, `α_I_decay=0.15`

**Novel Extensions (Training-Free):**

1. **Contrastive system prompts**: The draft model receives a strong safety-focused system prompt while the target gets the standard "helpful assistant" prompt. This synthetically widens the distribution gap on harmful prompts without fine-tuning, making the match ratio signal more reliable.

2. **Perplexity-gated early Union**: Before generation begins, prompt perplexity is computed under the draft model. GCG adversarial suffixes and DAN/persona jailbreak templates have anomalously high perplexity. If `ppl > threshold` (default 50.0), Union (safety) mode is forced from token 1 — solving the critical early-token problem where the match ratio window needs 7+ tokens before it can trigger safety mode.

**Why the original SSD implementation was broken:** The previous version always emitted `target_token = argmax(target_logits)` regardless of the match ratio — making it identical to vanilla decoding. The output token must be *sampled from the composite distribution* over the Intersection or Union token set.

**Phased execution:**
1. Load target → vanilla baseline → save → unload
2. Load draft + target → SSD generation → save → unload
3. Load Qwen3Guard → evaluate all saved responses

> **Note:** Delete `./data/*.json` to force-reload fresh datasets.

---

## Datasets

All datasets are cached under `./data/` on first run.

| Dataset | Type | Source |
|---|---|---|
| JailbreakBench | Harmful (curated) | `JailbreakBench/JBB-Behaviors` |
| HarmBench GCG | Harmful (adversarial suffix) | `cais/HarmBench` |
| DAN/Persona | Harmful (roleplay jailbreaks) | Hardcoded templates |
| PAIR-style | Harmful (legitimate framing) | Hardcoded prompts |
| WildJailbreak | Harmful (real-world) | `allenai/wildjailbreak` |
| DeepInception | Harmful (nested narrative) | Generated templates |
| XSTest | Benign (over-refusal test) | `walledai/XSTest` |

---

## Evaluation

All metrics are derived exclusively from **Qwen3Guard** labels — no keyword matching.

| Metric | Definition |
|---|---|
| **ASR** | % of harmful responses labeled `unsafe` (attack succeeded) |
| **Refusal Rate** | % of harmful responses labeled `safe` (model refused) |
| **Over-Refusal Rate** | % of benign responses labeled `safe` (model incorrectly refused) |
| **Controversial %** | % labeled `controversial` (borderline cases) |

**SSD-specific metrics:**
- `match_ratio`: fraction of tokens where the sampled token is in the draft's top-c (lower on harmful prompts indicates the jailbreak detection signal is working)
- `union_tokens`: fraction of tokens generated in Union (safety) mode
- `forced_union_by_ppl`: whether the PPL gate triggered early Union mode for a given prompt

---

## Model Download

Run the download cell once before experiments. Models are saved to `./downloaded_models/` and loaded from there on subsequent runs (faster, works offline).

```python
MODELS_TO_DOWNLOAD = [
    "Qwen/Qwen2.5-1.5B-Instruct",
    "huihui-ai/Qwen2.5-1.5B-Instruct-CensorTune",  # SafeDecoding only
    "Qwen/Qwen2.5-7B-Instruct",                     # SSD only
    "Qwen/Qwen3Guard-Gen-0.6B",
]
```

---

## Ablation

To isolate individual SSD extensions, change these flags in `Config`:

```python
use_contrastive_prompts = False   # disable safety system prompt on draft
use_ppl_gate = False              # disable perplexity-gated early Union
alpha_U = 0.0                     # disable safety blending in Union mode
alpha_I = 0.0                     # pure target in Intersection mode (→ vanilla)
```
