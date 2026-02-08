#!/usr/bin/env python
"""
SafeLoRA experiment: Apply SafeLoRA projection and evaluate safety.

This script:
1. Loads a quantized model with LoRA fine-tuning
2. Applies SafeLoRA projection to restore safety alignment
3. Evaluates safety using multiple methods
4. Compares with baseline results

Usage:
    # Option 1: Run with default config (modify SAFELORA_CONFIG in src/experiment_config.py)
    python experiments/run_safelora.py
    
    # Option 2: Import and customize in Python
    from experiments.run_safelora import run_safelora
    from src.experiment_config import create_safelora_config
    
    config = create_safelora_config(
        base_model="meta-llama/Llama-2-7b-hf",
        aligned_model="meta-llama/Llama-2-7b-chat-hf",
        peft_model="path/to/adapter",
        num_proj_layers=15,
    )
    run_safelora(config)
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

from src.experiment_config import SAFELORA_CONFIG, SafeLoRAExperimentConfig
from src.safe_lora_quantized import SafeLoRAQuantized
from src.config import SafeLoRAQuantizedConfig
from src.evaluation import SafetyEvaluator, load_harmful_prompts


def run_safelora(config: SafeLoRAExperimentConfig = None):
    """
    Run SafeLoRA safety evaluation.
    
    Args:
        config: Experiment configuration. If None, uses SAFELORA_CONFIG from experiment_config.py
    """
    if config is None:
        config = SAFELORA_CONFIG
    
    print("=" * 60)
    print("SAFELORA SAFETY EVALUATION")
    print("=" * 60)
    print(f"Base model: {config.model.base_model}")
    print(f"Aligned model: {config.model.aligned_model}")
    print(f"PEFT model: {config.model.peft_model}")
    print(f"Quantization: 4-bit={config.quantization.load_in_4bit}, 8-bit={config.quantization.load_in_8bit}")
    print(f"Layer selection: {config.projection.select_layers_type}")
    print(f"Num proj layers: {config.projection.num_proj_layers}")
    print(f"Projection strength: {config.projection.projection_strength}")
    print("=" * 60)
    
    # Create SafeLoRA config
    safelora_config = SafeLoRAQuantizedConfig(
        base_model_path=config.model.base_model,
        aligned_model_path=config.model.aligned_model,
        quantization_method=config.quantization.quantization_method,
        load_in_4bit=config.quantization.load_in_4bit,
        load_in_8bit=config.quantization.load_in_8bit,
        select_layers_type=config.projection.select_layers_type,
        num_proj_layers=config.projection.num_proj_layers,
        threshold=config.projection.threshold,
        projection_strength=config.projection.projection_strength,
        compute_safety_subspace_in_fp=config.projection.compute_in_fp,
        safety_subspace_cache_path=config.projection.cache_subspace_path,
        verbose=config.output.verbose,
    )
    
    # Setup quantization config for model loading
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
    
    # Load base model (use aligned model as base for PEFT)
    print("\nLoading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        config.model.aligned_model,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(config.model.aligned_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load PEFT adapter
    if config.model.peft_model:
        print(f"Loading PEFT adapter from {config.model.peft_model}...")
        peft_model = PeftModel.from_pretrained(base_model, config.model.peft_model, torch_dtype=torch.float16)
    else:
        print("WARNING: No PEFT model specified. Using base model without LoRA.")
        peft_model = base_model
    
    # Apply SafeLoRA projection
    print("\nApplying SafeLoRA projection...")
    safelora = SafeLoRAQuantized(peft_model, safelora_config)
    model = safelora.model
    model.eval()
    
    # Save projected adapter if requested
    if config.output.save_adapter:
        adapter_path = os.path.join(config.output.output_dir, "projected_adapter")
        safelora.save_projected_adapter(adapter_path)
    
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
    output_path = os.path.join(config.output.output_dir, f"safelora_results_{timestamp}.json")
    
    # Get projection stats
    proj_stats = safelora.get_projection_stats()
    
    # Save config along with results
    run_config = {
        "base_model": config.model.base_model,
        "aligned_model": config.model.aligned_model,
        "peft_model": config.model.peft_model,
        "load_in_4bit": config.quantization.load_in_4bit,
        "load_in_8bit": config.quantization.load_in_8bit,
        "select_layers_type": config.projection.select_layers_type,
        "num_proj_layers": config.projection.num_proj_layers,
        "threshold": config.projection.threshold,
        "projection_strength": config.projection.projection_strength,
        "num_samples": len(prompts),
        "timestamp": timestamp,
    }
    
    output = {
        "config": run_config,
        "projection_stats": {
            "num_projected": proj_stats["num_projected"],
            "total_layers": proj_stats["total_layers"],
            "threshold_used": proj_stats["threshold"],
            "mean_cosine": proj_stats["mean_cosine"],
            "mean_pdst": proj_stats["mean_pdst"],
        },
        "results": {}
    }
    
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
    print("FINAL RESULTS (WITH SAFELORA)")
    print("=" * 60)
    print(f"Layers projected: {proj_stats['num_projected']}/{proj_stats['total_layers']}")
    print(f"Mean Pdst: {proj_stats['mean_pdst']:.4f}")
    print("-" * 60)
    for name, result in results.items():
        print(f"{result.metric_name}: {result.score:.4f}")
    print("=" * 60)
    
    return results, proj_stats


def main():
    """Run SafeLoRA experiment with default configuration."""
    run_safelora(SAFELORA_CONFIG)


if __name__ == "__main__":
    main()
