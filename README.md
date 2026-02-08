# Safety Alignment of Quantized Models

**ECE 285 Project** - Implementing SafeLoRA for improving safety alignment in quantized Large Language Models.

This project adapts [SafeLoRA](https://github.com/IBM/SafeLoRA) (NeurIPS 2024) for quantized models, enabling safety preservation when fine-tuning memory-efficient LLMs.

## Overview

When fine-tuning LLMs using LoRA (Low-Rank Adaptation), safety alignment can degrade - making models respond to harmful prompts. SafeLoRA addresses this by projecting LoRA weights onto a safety-aligned subspace, maintaining safety without retraining.

This implementation extends SafeLoRA to support:
- **BitsAndBytes 4-bit/8-bit quantization**
- **GPTQ quantized models**
- **AWQ quantized models**
- **QLoRA (quantized base + FP16 LoRA adapters)**

## Installation

```bash
# Clone the repository
cd SafetyAlignmentOfQuantizedModels

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Requirements
- Python 3.10+
- PyTorch 2.1+
- CUDA 11.8+ (for GPU acceleration)
- ~16GB GPU memory for 4-bit 7B models

## Project Structure

```
SafetyAlignmentOfQuantizedModels/
├── README.md
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── config.py                 # SafeLoRA configuration
│   ├── safe_lora_quantized.py    # Main SafeLoRA implementation
│   ├── quantization_utils.py     # Quantization helpers
│   └── evaluation.py             # Safety evaluation utilities
├── experiments/
│   ├── run_baseline.py           # Evaluate baseline safety
│   ├── run_safelora.py           # Evaluate with SafeLoRA
│   └── compare_results.py        # Compare and visualize results
└── notebooks/
    └── 01_exploration.ipynb      # Interactive exploration
```

## Quick Start

### 1. Basic Usage

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from src import SafeLoRAQuantized, SafeLoRAQuantizedConfig

# Load your QLoRA fine-tuned model
bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-chat-hf",
    quantization_config=bnb_config,
    device_map="auto"
)
peft_model = PeftModel.from_pretrained(base_model, "path/to/your/lora/adapter")

# Configure SafeLoRA
config = SafeLoRAQuantizedConfig.for_4bit_bnb(
    base_model_path="meta-llama/Llama-2-7b-hf",
    aligned_model_path="meta-llama/Llama-2-7b-chat-hf",
    num_proj_layers=10,  # Project top 10 least-aligned layers
)

# Apply SafeLoRA projection
safelora = SafeLoRAQuantized(peft_model, config)
safe_model = safelora.model  # Use this for inference
```

### 2. Run Experiments

```bash
# Evaluate baseline (before SafeLoRA)
python experiments/run_baseline.py \
    --base_model meta-llama/Llama-2-7b-chat-hf \
    --peft_model path/to/adapter \
    --harmful_prompts data/advbench_harmful_behaviors.csv \
    --load_in_4bit \
    --output_dir results/baseline

# Evaluate with SafeLoRA
python experiments/run_safelora.py \
    --base_model meta-llama/Llama-2-7b-hf \
    --aligned_model meta-llama/Llama-2-7b-chat-hf \
    --peft_model path/to/adapter \
    --harmful_prompts data/advbench_harmful_behaviors.csv \
    --load_in_4bit \
    --num_proj_layers 10 \
    --output_dir results/safelora

# Compare results
python experiments/compare_results.py \
    --baseline_results "results/baseline/*.json" \
    --safelora_results "results/safelora/*.json" \
    --output_dir results/comparison
```

## Configuration Options

### Layer Selection Methods

| Method | Description |
|--------|-------------|
| `number` | Project the N layers with lowest safety alignment (recommended) |
| `threshold` | Project layers with cosine similarity below threshold |
| `all` | Project all LoRA layers |
| `auto` | Adaptive selection using median cosine as threshold |

### Key Parameters

```python
SafeLoRAQuantizedConfig(
    base_model_path="...",           # Path to unaligned base model
    aligned_model_path="...",         # Path to safety-aligned model
    select_layers_type="number",      # Layer selection method
    num_proj_layers=10,               # Number of layers to project
    threshold=0.5,                    # Cosine threshold (for threshold method)
    projection_strength=1.0,          # 0.0-1.0, interpolate projection
    compute_safety_subspace_in_fp=True,  # Recommended for accuracy
)
```

## Evaluation Metrics

The project includes three safety evaluation methods:

1. **Template Safety Check**: Keyword-based refusal detection
2. **Llama Guard**: Meta's safety classifier
3. **GPT-4 Judge**: LLM-based safety judgment (requires API key)

Plus utility metrics:
- **ROUGE scores** for summarization tasks

## References

- **SafeLoRA Paper**: [Safe LoRA: the Silver Lining of Reducing Safety Risks when Fine-tuning Large Language Models](https://arxiv.org/abs/2405.16833) (NeurIPS 2024)
- **SafeLoRA Code**: [IBM/SafeLoRA](https://github.com/IBM/SafeLoRA)
- **LoRA Paper**: [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- **QLoRA Paper**: [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)

## License

Apache 2.0 License (following original SafeLoRA)

## Acknowledgments

- Original SafeLoRA implementation by IBM Research
- ADV-LLM evaluation framework for safety testing
