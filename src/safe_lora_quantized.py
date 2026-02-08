# Copyright 2024 SafeLoRA Quantized Project
# Adapted from IBM SafeLoRA: https://github.com/IBM/SafeLoRA
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""
SafeLoRA adapted for Quantized Models

This module implements SafeLoRA projection for quantized LLMs, supporting:
- BitsAndBytes 4-bit and 8-bit quantization
- GPTQ quantization
- AWQ quantization
- QLoRA (quantized base + FP LoRA adapters)

The key insight is that we can:
1. Compute the safety subspace from full-precision aligned/base models
2. Apply the projection to LoRA weights (which remain in FP16)
3. The quantized base model is unchanged
"""

import copy
import os
from typing import Optional, List, Tuple, Dict, Any

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM
from peft import PeftModel

from .config import SafeLoRAQuantizedConfig
from .quantization_utils import (
    load_quantized_model,
    dequantize_layer_weights,
    is_quantized_layer,
    get_layer_by_name,
)


class SafeLoRAQuantized:
    """
    SafeLoRA implementation for quantized models.
    
    This class projects LoRA weights onto a safety-aligned subspace to maintain
    model safety after fine-tuning, specifically adapted for quantized models.
    
    Key Features:
    - Computes safety subspace from full-precision models (recommended) or 
      dequantized quantized models
    - Projects LoRA adapter weights to preserve safety alignment
    - Supports caching of safety subspace for efficiency
    - Works with QLoRA, GPTQ-LoRA, and AWQ-LoRA setups
    
    Example Usage:
    ```python
    from transformers import AutoModelForCausalLM
    from peft import PeftModel
    from src import SafeLoRAQuantized, SafeLoRAQuantizedConfig
    
    # Load your QLoRA fine-tuned model
    base_model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-2-7b-hf",
        load_in_4bit=True,
        device_map="auto"
    )
    peft_model = PeftModel.from_pretrained(base_model, "path/to/lora/adapter")
    
    # Configure SafeLoRA
    config = SafeLoRAQuantizedConfig.for_4bit_bnb(
        base_model_path="meta-llama/Llama-2-7b-hf",
        aligned_model_path="meta-llama/Llama-2-7b-chat-hf",
        num_proj_layers=10,
    )
    
    # Apply SafeLoRA projection
    safelora = SafeLoRAQuantized(peft_model, config)
    safe_model = safelora.model  # Use this model for inference
    ```
    """
    
    def __init__(
        self,
        peft_model: torch.nn.Module,
        config: SafeLoRAQuantizedConfig,
    ):
        """
        Initialize SafeLoRA for quantized models.
        
        Args:
            peft_model: A PeftModel (LoRA fine-tuned model)
            config: SafeLoRAQuantizedConfig with paths and settings
        """
        self.peft_model = peft_model
        self.config = config
        self.peft_config = peft_model.peft_config.get("default", list(peft_model.peft_config.values())[0])
        
        # Store original model for comparison
        self.model_ori = copy.deepcopy(peft_model)
        
        # Compute or load safety projection matrices
        if config.verbose:
            print("Computing safety-aligned projection matrices...")
        
        self.projection_matrices = self._get_projection_matrices()
        
        # Apply projection based on layer selection method
        if config.verbose:
            print(f"Applying projection with method: {config.select_layers_type}")
        
        self.model, self.projection_stats = self._apply_projection()
        
        if config.verbose:
            self._print_summary()
    
    def _get_projection_matrices(self) -> List[torch.Tensor]:
        """
        Compute projection matrices for safety-aligned subspace.
        
        Returns:
            List of projection matrices, one per target module layer
        """
        # Check for cached matrices
        if self.config.safety_subspace_cache_path and os.path.exists(self.config.safety_subspace_cache_path):
            if self.config.verbose:
                print(f"Loading cached projection matrices from {self.config.safety_subspace_cache_path}")
            return torch.load(self.config.safety_subspace_cache_path)
        
        # Compute fresh projection matrices
        if self.config.compute_safety_subspace_in_fp:
            projection_matrices = self._compute_fp_projection_matrices()
        else:
            projection_matrices = self._compute_quantized_projection_matrices()
        
        # Cache if path provided
        if self.config.safety_subspace_cache_path:
            os.makedirs(os.path.dirname(self.config.safety_subspace_cache_path), exist_ok=True)
            torch.save(projection_matrices, self.config.safety_subspace_cache_path)
            if self.config.verbose:
                print(f"Cached projection matrices to {self.config.safety_subspace_cache_path}")
        
        return projection_matrices
    
    def _compute_fp_projection_matrices(self) -> List[torch.Tensor]:
        """
        Compute projection matrices using full-precision models.
        
        This is the recommended approach as it avoids quantization noise
        in the safety subspace computation.
        """
        if self.config.verbose:
            print("Loading full-precision base model...")
        
        base_model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model_path,
            return_dict=True,
            load_in_8bit=False,
            device_map="cpu",
            low_cpu_mem_usage=True,
            torch_dtype=torch.float32,
        )
        
        if self.config.verbose:
            print("Loading full-precision aligned model...")
        
        aligned_model = AutoModelForCausalLM.from_pretrained(
            self.config.aligned_model_path,
            return_dict=True,
            load_in_8bit=False,
            device_map="cpu",
            low_cpu_mem_usage=True,
            torch_dtype=torch.float32,
        )
        
        return self._compute_projection_from_models(base_model, aligned_model)
    
    def _compute_quantized_projection_matrices(self) -> List[torch.Tensor]:
        """
        Compute projection matrices by dequantizing quantized models.
        
        This approach may introduce quantization noise but uses less memory.
        """
        if self.config.verbose:
            print("Loading and dequantizing base model...")
        
        base_model, _ = load_quantized_model(
            self.config.base_model_path,
            quantization_method=self.config.quantization_method,
            device_map="cpu",
        )
        
        if self.config.verbose:
            print("Loading and dequantizing aligned model...")
        
        aligned_model, _ = load_quantized_model(
            self.config.aligned_model_path,
            quantization_method=self.config.quantization_method,
            device_map="cpu",
        )
        
        return self._compute_projection_from_models(
            base_model, 
            aligned_model,
            dequantize=True
        )
    
    def _compute_projection_from_models(
        self,
        base_model: torch.nn.Module,
        aligned_model: torch.nn.Module,
        dequantize: bool = False,
    ) -> List[torch.Tensor]:
        """
        Compute projection matrices from base and aligned models.
        
        The projection matrix for each layer is computed as:
        P = V @ V^T / ||V||^2
        where V = W_aligned - W_base (the safety direction)
        
        Args:
            base_model: Base (unaligned) model
            aligned_model: Safety-aligned model
            dequantize: Whether to dequantize weights before computation
        
        Returns:
            List of projection matrices
        """
        projection_matrices = []
        
        # Get target modules from PEFT config
        target_modules = self.config.target_modules or list(self.peft_config.target_modules)
        
        if self.config.verbose:
            print(f"Computing projection matrices for modules: {target_modules}")
        
        # Iterate over parameters
        base_params = list(base_model.named_parameters())
        aligned_params = list(aligned_model.named_parameters())
        
        layer_idx = 0
        for (b_name, b_param), (a_name, a_param) in tqdm(
            zip(base_params, aligned_params),
            total=len(base_params),
            desc="Computing projections",
            disable=not self.config.verbose
        ):
            # Check if this parameter belongs to a target module
            if any(module in a_name for module in target_modules):
                assert b_param.shape == a_param.shape, \
                    f"Shape mismatch: {b_name} {b_param.shape} vs {a_name} {a_param.shape}"
                
                # Skip excluded layers
                if self.config.exclude_layers and layer_idx in self.config.exclude_layers:
                    layer_idx += 1
                    continue
                
                # Compute safety direction vector
                if dequantize:
                    b_weight = dequantize_layer_weights(
                        get_layer_by_name(base_model, b_name.replace('.weight', '')),
                        b_name,
                        self.config.quantization_method
                    )
                    a_weight = dequantize_layer_weights(
                        get_layer_by_name(aligned_model, a_name.replace('.weight', '')),
                        a_name,
                        self.config.quantization_method
                    )
                else:
                    b_weight = b_param.data.float()
                    a_weight = a_param.data.float()
                
                # V = W_aligned - W_base (safety direction)
                V = a_weight - b_weight
                
                # P = V @ V^T / ||V||^2 (projection matrix)
                V = V.to(self.config.device)
                P = torch.mm(V, V.t()) / torch.norm(V)
                
                projection_matrices.append(P.detach().cpu())
                layer_idx += 1
        
        # Clean up
        del base_model, aligned_model
        torch.cuda.empty_cache()
        
        return projection_matrices
    
    def _apply_projection(self) -> Tuple[torch.nn.Module, Dict[str, Any]]:
        """
        Apply SafeLoRA projection to LoRA weights.
        
        Returns:
            Tuple of (projected model, projection statistics)
        """
        if self.config.select_layers_type == "threshold":
            return self._projected_weighted(
                threshold=self.config.threshold,
            )
        
        elif self.config.select_layers_type == "number":
            # First pass: compute all cosine similarities
            _, cos_values = self._projected_weighted(threshold=-1, dry_run=True)
            
            # Select threshold based on number of layers
            sorted_cos = np.sort(cos_values)
            threshold = sorted_cos[:self.config.num_proj_layers][-1] if len(sorted_cos) >= self.config.num_proj_layers else sorted_cos[0]
            
            # Second pass: apply projection
            return self._projected_weighted(threshold=threshold)
        
        elif self.config.select_layers_type == "all":
            return self._projected_weighted(threshold=float('inf'))
        
        elif self.config.select_layers_type == "auto":
            # Adaptive selection based on cosine similarity distribution
            _, cos_values = self._projected_weighted(threshold=-1, dry_run=True)
            
            # Use median as threshold (project layers with below-median alignment)
            threshold = np.median(cos_values)
            return self._projected_weighted(threshold=threshold)
        
        else:
            raise ValueError(f"Unknown select_layers_type: {self.config.select_layers_type}")
    
    def _projected_weighted(
        self,
        threshold: float,
        dry_run: bool = False,
    ) -> Tuple[torch.nn.Module, Any]:
        """
        Apply projection to LoRA weights based on cosine similarity threshold.
        
        Args:
            threshold: Cosine similarity threshold (layers with cos <= threshold are projected)
            dry_run: If True, only compute statistics without modifying weights
        
        Returns:
            Tuple of (projected model, cosine similarities or full stats)
        """
        proj_matrices = self.projection_matrices
        
        idx = 0
        num_projected = 0
        cos_values = []
        pdst_values = []
        projected_layers = []
        
        # Get LoRA rank
        lora_r = self.peft_config.r
        
        # Temporary storage for B matrices (LoRA down projection)
        B_matrix = None
        
        for (name, param), (name_ori, param_ori) in zip(
            self.peft_model.named_parameters(),
            self.model_ori.named_parameters()
        ):
            if 'lora' in name.lower():
                # LoRA has two matrices: A (down) and B (up)
                # A: (r, in_features), B: (out_features, r)
                # The full LoRA update is: ΔW = B @ A
                
                if param.shape[0] == lora_r:
                    # This is matrix A (down projection)
                    B_matrix = copy.deepcopy(param_ori)
                    continue
                
                if param.shape[0] != lora_r and B_matrix is not None:
                    # This is matrix B (up projection) - apply SafeLoRA here
                    P = proj_matrices[idx].to(param.device)
                    
                    # Compute projected LoRA update
                    W_projected = torch.mm(P, param_ori.data)  # Project B matrix
                    
                    # Compute full LoRA updates for comparison
                    full_update_projected = torch.mm(W_projected, B_matrix)
                    full_update_original = torch.mm(param_ori.data, B_matrix)
                    
                    # Compute cosine similarity between projected and original updates
                    cos = torch.nn.functional.cosine_similarity(
                        full_update_projected.reshape(1, -1),
                        full_update_original.reshape(1, -1)
                    ).item()
                    cos = np.round(cos, 5)
                    cos_values.append(cos)
                    
                    # Compute projection distance metric
                    pdst = 1 / (1 + torch.norm(
                        full_update_projected.reshape(1, -1) - full_update_original.reshape(1, -1)
                    ))
                    pdst_values.append(pdst.item())
                    
                    # Apply projection if below threshold (and not dry run)
                    if cos <= threshold and not dry_run:
                        # Apply with projection strength
                        if self.config.projection_strength < 1.0:
                            # Interpolate between original and projected
                            param.data = (
                                self.config.projection_strength * W_projected +
                                (1 - self.config.projection_strength) * param_ori.data
                            )
                        else:
                            param.data = W_projected
                        
                        num_projected += 1
                        projected_layers.append(name)
                    else:
                        if not dry_run:
                            param.data = param_ori.data
                    
                    idx += 1
                    B_matrix = None  # Reset for next layer
        
        if dry_run:
            return self.peft_model, cos_values
        
        stats = {
            "num_projected": num_projected,
            "total_layers": len(cos_values),
            "threshold": threshold,
            "mean_cosine": np.mean(cos_values) if cos_values else 0,
            "mean_pdst": np.mean(pdst_values) if pdst_values else 0,
            "cosine_values": cos_values,
            "pdst_values": pdst_values,
            "projected_layers": projected_layers,
        }
        
        return self.peft_model, stats
    
    def _print_summary(self):
        """Print summary of SafeLoRA projection."""
        stats = self.projection_stats
        print("\n" + "=" * 60)
        print("SafeLoRA Projection Summary")
        print("=" * 60)
        print(f"Layers projected: {stats['num_projected']} / {stats['total_layers']}")
        print(f"Threshold: {stats['threshold']:.4f}")
        print(f"Mean cosine similarity: {stats['mean_cosine']:.4f}")
        print(f"Mean Pdst (> 0.8 is better): {stats['mean_pdst']:.4f}")
        print(f"Projection strength: {self.config.projection_strength}")
        print("=" * 60 + "\n")
    
    def get_projection_stats(self) -> Dict[str, Any]:
        """Get detailed projection statistics."""
        return self.projection_stats
    
    def save_projected_adapter(self, save_path: str):
        """
        Save the projected LoRA adapter.
        
        Args:
            save_path: Directory to save the adapter
        """
        self.model.save_pretrained(save_path)
        if self.config.verbose:
            print(f"Saved projected adapter to {save_path}")
    
    @classmethod
    def from_pretrained(
        cls,
        base_model_path: str,
        peft_model_path: str,
        config: SafeLoRAQuantizedConfig,
        **model_kwargs
    ) -> "SafeLoRAQuantized":
        """
        Load a PEFT model and apply SafeLoRA projection.
        
        Args:
            base_model_path: Path to the base model
            peft_model_path: Path to the PEFT adapter
            config: SafeLoRA configuration
            **model_kwargs: Additional arguments for model loading
        
        Returns:
            SafeLoRAQuantized instance with projected model
        """
        from transformers import BitsAndBytesConfig
        
        # Prepare quantization config
        quantization_config = None
        if config.load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=getattr(torch, config.bnb_4bit_compute_dtype),
                bnb_4bit_quant_type=config.bnb_4bit_quant_type,
                bnb_4bit_use_double_quant=config.use_double_quant,
            )
        elif config.load_in_8bit:
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        
        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            quantization_config=quantization_config,
            device_map=config.device_map,
            trust_remote_code=True,
            **model_kwargs
        )
        
        # Load PEFT adapter
        peft_model = PeftModel.from_pretrained(
            base_model,
            peft_model_path,
            torch_dtype=torch.float16,
        )
        
        return cls(peft_model, config)

