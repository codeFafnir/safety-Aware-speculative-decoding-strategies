#!/usr/bin/env python
"""
Baseline experiment: Evaluate safety of quantized model BEFORE SafeLoRA projection.

This script:
1. Loads a quantized model with LoRA fine-tuning
2. Evaluates safety using multiple methods
3. Saves results for comparison

Usage:
    # Option 1: Run with default config (modify BASELINE_CONFIG in src/experiment_config.py)
    python experiments/run_baseline.py
    
    # Option 2: Import and customize in Python
    from experiments.run_baseline import run_baseline
    from src.experiment_config import create_baseline_config
    
    config = create_baseline_config(
        base_model="meta-llama/Llama-2-7b-chat-hf",
        peft_model="path/to/adapter",
        num_samples=50,
    )
    run_baseline(config)
"""

import os
import sys
import json
from datetime import datetime

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.experiment_config import BASELINE_CONFIG, BaselineExperimentConfig
from src.evaluation import SafetyEvaluator, load_harmful_prompts


def run_baseline(config: BaselineExperimentConfig = None):
    """
    Run baseline safety evaluation.
    
    Args:
        config: Experiment configuration. If None, uses BASELINE_CONFIG from experiment_config.py
    """
    if config is None:
        config = BASELINE_CONFIG
    
    print("=" * 60)
    print("BASELINE SAFETY EVALUATION")
    print("=" * 60)
    print(f"Base model: {config.model.base_model}")
    print(f"PEFT model: {config.model.peft_model}")
    print(f"Quantization: 4-bit={config.quantization.load_in_4bit}, 8-bit={config.quantization.load_in_8bit}")
    print("=" * 60)
    
    # Setup quantization config
    quantization_config = None
    if config.quantization.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=getattr(torch, config.quantization.bnb_4bit_compute_dtype),
            bnb_4bit_quant_type=config.quantization.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=config.quantization.use_double_quant,
        )
    elif config.quantization.load_in_8bit:
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    
    # Load model
    print("\nLoading model...")
    model = AutoModelForCausalLM.from_pretrained(
        config.model.base_model,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(config.model.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load PEFT adapter if provided
    if config.model.peft_model:
        print(f"Loading PEFT adapter from {config.model.peft_model}...")
        model = PeftModel.from_pretrained(model, config.model.peft_model, torch_dtype=torch.float16)
    
    model.eval()
    
    # Load harmful prompts
    print(f"\nLoading harmful prompts from {config.evaluation.harmful_prompts_path}...")
    prompts = load_harmful_prompts(config.evaluation.harmful_prompts_path)
    
    if config.evaluation.num_samples:
        prompts = prompts[:config.evaluation.num_samples]
    
    print(f"Evaluating on {len(prompts)} prompts")
    
    # Create evaluator
    evaluator = SafetyEvaluator(
        model=model,
        tokenizer=tokenizer,
        device="cuda" if torch.cuda.is_available() else "cpu",
        max_new_tokens=config.evaluation.max_new_tokens,
    )
    
    # Run evaluation
    results = evaluator.full_evaluation(
        harmful_prompts=prompts,
        use_llama_guard=config.evaluation.use_llama_guard,
        use_gpt4=config.evaluation.use_gpt4,
        verbose=config.output.verbose,
    )
    
    # Save results
    os.makedirs(config.output.output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(config.output.output_dir, f"baseline_results_{timestamp}.json")
    
    # Save config along with results
    run_config = {
        "base_model": config.model.base_model,
        "peft_model": config.model.peft_model,
        "load_in_4bit": config.quantization.load_in_4bit,
        "load_in_8bit": config.quantization.load_in_8bit,
        "num_samples": len(prompts),
        "timestamp": timestamp,
    }
    
    output = {"config": run_config, "results": {}}
    for name, result in results.items():
        output["results"][name] = {
            "metric_name": result.metric_name,
            "score": result.score,
            "num_samples": result.num_samples,
        }
    
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to {output_path}")
    
    # Print final summary
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    for name, result in results.items():
        print(f"{result.metric_name}: {result.score:.4f}")
    print("=" * 60)
    
    return results


def main():
    """Run baseline experiment with default configuration."""
    run_baseline(BASELINE_CONFIG)


if __name__ == "__main__":
    main()
