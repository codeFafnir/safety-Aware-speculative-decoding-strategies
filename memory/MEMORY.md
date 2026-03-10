# Project Memory: Safety-Aware Speculative Decoding

## Project Overview
ECE 285 course project comparing safety-decoding strategies.
Working dir: `/Users/atharvramesh/UCSD/Winter2026/ECE285/safety-Aware-speculative-decoding-strategies`
Pod: `atharv-rwx-pod-2`, path: `/workspace/safety-Aware-speculative-decoding-strategies/SSD_variants`

## Key Files
- `SSD_variants/ssd_experiments.py` — main experiment script (~1183 lines)
- `SSD_variants/ssd_vllm.py` — vLLM-accelerated version
- `SSD_variants/ssd_steering.py` — NEW: hidden-state steering extension (arXiv 2511.09844)
- `SSD_variants/prepare_datasets.py` — one-time dataset prep script
- `SSD_variants/data/` — committed JSON datasets (reproducibility)
- `SSD_variants/results/metrics.json` — output metrics (gitignored on pod)

## Models
- Draft: `Qwen/Qwen2.5-1.5B-Instruct` (hidden_size=1536)
- Target: `Qwen/Qwen2.5-7B-Instruct` (hidden_size=3584)
- Guard: `Qwen/Qwen3Guard-Gen-0.6B`
- Default: 4-bit quantization, can override with --no_4bit

## Methods Implemented
1. **vanilla** — target model, no intervention
2. **ssd** — baseline (binary match-ratio mode switch, bin=7 tokens)
3. **ssd_crs** — CRS extension (per-step composite risk score)
   - r_t = 0.3*(1-match) + 0.3*KL + 0.1*ΔH + 0.3*refusal_mass
   - threshold=0.2, window=3
4. **ssd_steering** — NEW: verifier hidden-state injection (see below)

## Latest Results (pod metrics.json) — FINAL
| Method       | ASR strict | Refusal | Over-refusal | Avg resp len |
|--------------|-----------|---------|-------------|-------------|
| vanilla      | 12.0%     | 86.7%   | 0%          | —           |
| ssd          | 14.7%     | 84.0%   | 0%          | —           |
| ssd_crs      | 6.7%      | 93.3%   | 10%         | —           |
| ssd_steering | 6.7%      | 93.3%   | **0%**      | 216 words   | ← same ASR, no over-refusal

Per-source ASR (steering): beavertails=4%, dan=0%, di=20%, jbb=8%

## Key Config Values
```python
ppl_threshold = 150.0   # raised from 50 (XSTest false positives)
crs_threshold = 0.2     # lowered from 0.5 (mean risk on harmful = 0.214)
draft_system_prompt = "extremely safety-conscious... when in doubt, refuse"  # STRONG
```
- Softening draft prompt causes regression (SSD needs big distribution gap)

## Steering Extension (arXiv: 2511.09844)
Paper: "Steering Pretrained Drafters during Speculative Decoding"
Our adaptation in `ssd_steering.py`:
- `SteeringProjection`: Linear(3584→1536, bias=False), small-init (std=0.01)
- `SteeringInjector`: persistent forward hooks on draft MLP layers
- Steering vector g_t = proj(normalize(target_last_hidden_state))  ← MUST normalize
- Scale = base_magnitude * (1 + amplify * smoothed_crs)  [base=0.003, amplify=5.0]
- Training: refusal distillation on harmful prompts (NOT UltraChat — that caused 90.7% ASR)
- Checkpoint: `steering_ckpts/proj_refusal.pt` (KL 2.32 → 0.83 over 5 epochs)

## Critical Steering Bugs Fixed
1. **UltraChat training → compliance**: proj trained on helpful data pushed draft toward compliance.
   Fix: train on harmful prompts where target refuses → `_build_refusal_sequences()`
2. **t_hidden norm=295**: target.model.norm output is NOT unit-norm (learned weight scaling).
   Raw injection was 295× too large → logit shift ~50 → token collapse ("I I I I...").
   Fix: normalize t_hidden to unit norm in `_apply_steering()` before proj.
3. **model.norm hook degeneration**: `device_map=auto` AlignDevicesHook corrupts custom hooks.
   Fix: use `output_hidden_states=True` (reliable) and index `hidden_states[-1]`.

## Architecture Notes
- KV cache: draft on cuda:0, target on cuda:1 (GPU pinning)
- `_init_kv` → full prompt pass; `_forward_one` → O(1) single token; `_forward_batch` → multi-token
- vLLM used for vanilla batch generation + guard eval (can't do SSD: needs full logit dist)
- Flash Attention 2: try/except wraps `from_pretrained` itself (ImportError fires there)

## Over-refusal Fix
- Bug was: `counts["safe"] / n * 100` (backwards — safe = helpful response)
- Fix: scan response text for refusal keywords (`_is_refusal(text)`)

## Dataset (committed to repo)
- beavertails.json (25), jailbreakbench.json (25), dan_prompts.json (15)
- deepinception.json (10), xstest.json (30), manifest.json (checksums)
- DeepInception (di) is hardest: vanilla 60% ASR → CRS 20% ASR
