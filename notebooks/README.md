# Notebooks — Early Experimentation & Baselines

These notebooks document the early exploration phase of the project: baseline implementations, initial evaluations, and the groundwork that motivated the CRS-SSD design. They are self-contained and can be run independently of the main experiment scripts.

| Notebook | Description |
|---|---|
| `02_safedecoding_baseline_v2.ipynb` | SafeDecoding (Xu et al., ACL 2024) baseline using `Qwen2.5-1.5B` + CensorTune expert |
| `03_ssd_baseline.ipynb` | Faithful SSD re-implementation (Wang et al., EMNLP 2025) with contrastive prompts and PPL-gated early Union extensions |
| `04_ssd_aasd_baseline.ipynb` | SSD + AASD integration experiments; documents the generation-collapse failure with frozen λ and motivates the decaying-λ fix |

These were used to establish baseline numbers, debug the SSD implementation, and understand failure modes before the full CRS pipeline was built in `SSD_variants/ssd_experiments.py`.
