# SSD Variants — Safety-Aware Speculative Decoding

**ECE 285 Project: Safety Alignment of LLMs via Speculative Decoding**

---

## Setup

```bash
# Install dependencies
pip install -r requirements.txt   # from repo root

# Log in to HuggingFace (required once — models are Qwen public but login needed for some datasets)
huggingface-cli login
```

Models and datasets download automatically on first run and are cached locally.

---

## `ssd_experiments.py` — Main Experiment Script

Implements and compares three methods end-to-end:

| Method | Description |
|---|---|
| `vanilla` | Target model only, no safety intervention |
| `ssd` | SSD baseline faithful to [paper repo](https://github.com/k-k1w-w1x-x/Speculative-Safety-Aware-Decoding) |
| `ssd_crs` | SSD + Composite Risk Score (our extension) |

### Run all phases

```bash
cd SSD_variants
python ssd_experiments.py
```

### Run individual phases

Generation and evaluation are separate phases so you can re-run eval without regenerating responses.

```bash
# Generate responses only
python ssd_experiments.py --phases vanilla ssd ssd_crs

# Evaluate already-generated responses (fast — just runs Qwen3Guard)
python ssd_experiments.py --phases eval

# Run a single method and evaluate
python ssd_experiments.py --phases ssd eval

# Quick test with shorter responses
python ssd_experiments.py --phases ssd eval --max_new_tokens 64

# Disable 4-bit quantization (if you have enough VRAM)
python ssd_experiments.py --no_4bit
```

### Output files

All outputs go into `SSD_variants/` subdirectories created on first run:

```
./responses/
    vanilla_harmful.json       # raw generated responses
    vanilla_benign.json
    ssd_harmful.json
    ssd_benign.json
    ssd_crs_harmful.json
    ssd_crs_benign.json
    *_eval.json                # above + guard_label per response (after eval phase)

./results/
    metrics.json               # all metrics in one file
```

### Evaluation metrics

| Metric | Description |
|---|---|
| `ASR strict%` | % harmful responses labeled `unsafe` by Qwen3Guard |
| `ASR lenient%` | `unsafe + 0.5*controversial` — catches partial compliance |
| `Refusal%` | % harmful responses refused |
| `Over-refusal%` | % benign responses incorrectly refused |
| `Resp len (benign)` | Avg word count on benign responses — proxy for utility loss |

Per-source ASR is also printed, broken down by attack type (gcg, dan, jbb, wjb, di).

### Ablation flags (edit `Config` in the script)

```python
use_ppl_gate            = False   # disable perplexity-gated early union
use_contrastive_prompts = False   # same system prompt for draft and target
alpha_U                 = 0.0     # disable safety blending in union mode
crs_w2                  = 0.0     # ablate KL term from CRS
crs_w3                  = 0.0     # ablate entropy gap term from CRS
crs_w4                  = 0.0     # ablate refusal mass term from CRS
```

### CRS extension — how it works

The baseline SSD switches modes (intersection ↔ union) once every `b=7` tokens based on match ratio.
The CRS extension replaces this with a **per-step composite risk score**:

```
r_t = w1*(1 - match) + w2*KL(p_expert||p_target) + w3*ΔH + w4*refusal_mass
```

- **KL** — soft distribution disagreement; fires even when top tokens match
- **ΔH = H(p_target) - H(p_expert)** — target is uncertain while expert is confident → early jailbreak signal
- **refusal_mass** — direct probability the expert places on refusal-start tokens

Mode switches immediately when smoothed `r_t > crs_threshold` (no bin lag).

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
