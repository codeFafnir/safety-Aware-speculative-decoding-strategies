# Copyright 2024 SafeLoRA Quantized Project
# Adapted from IBM SafeLoRA: https://github.com/IBM/SafeLoRA
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from dataclasses import dataclass, field
from typing import Optional, List, Literal


@dataclass
class SafeLoRAQuantizedConfig:
    """
    Configuration class for SafeLoRA adapted for quantized models.
    
    This extends the original SafeLoRA config to support:
    - Quantized model loading (4-bit, 8-bit via bitsandbytes, GPTQ, AWQ)
    - Dequantization strategies for safety subspace computation
    - Additional layer selection methods
    """
    
    # Model paths
    base_model_path: str = field(
        default=None,
        metadata={"help": "Path to the base (unaligned) model for computing safety direction"},
    )
    
    aligned_model_path: str = field(
        default=None,
        metadata={"help": "Path to the safety-aligned model for computing safety direction"},
    )
    
    peft_model_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the fine-tuned PEFT/LoRA adapter (if loading separately)"},
    )
    
    # Quantization settings
    quantization_method: Literal["none", "bitsandbytes_4bit", "bitsandbytes_8bit", "gptq", "awq"] = field(
        default="none",
        metadata={"help": "Quantization method used for the model"},
    )
    
    load_in_4bit: bool = field(
        default=False,
        metadata={"help": "Whether to load the quantized model in 4-bit (bitsandbytes)"},
    )
    
    load_in_8bit: bool = field(
        default=False,
        metadata={"help": "Whether to load the quantized model in 8-bit (bitsandbytes)"},
    )
    
    bnb_4bit_compute_dtype: str = field(
        default="float16",
        metadata={"help": "Compute dtype for 4-bit quantization (float16, bfloat16)"},
    )
    
    bnb_4bit_quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization type for 4-bit (nf4 or fp4)"},
    )
    
    use_double_quant: bool = field(
        default=True,
        metadata={"help": "Whether to use nested quantization for 4-bit"},
    )
    
    # Safety subspace computation
    compute_safety_subspace_in_fp: bool = field(
        default=True,
        metadata={"help": "Compute safety subspace using full-precision models (recommended)"},
    )
    
    safety_subspace_cache_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to cache/load precomputed safety subspace matrices"},
    )
    
    # Layer selection
    select_layers_type: Literal["threshold", "number", "all", "auto"] = field(
        default="number",
        metadata={"help": "Method to select layers for projection: threshold, number, all, or auto"},
    )
    
    threshold: float = field(
        default=0.5,
        metadata={"help": "Cosine similarity threshold for layer selection (used if select_layers_type='threshold')"},
    )
    
    num_proj_layers: int = field(
        default=10,
        metadata={"help": "Number of layers to project (used if select_layers_type='number')"},
    )
    
    target_modules: Optional[List[str]] = field(
        default=None,
        metadata={"help": "List of module names to apply SafeLoRA projection. If None, uses PEFT config."},
    )
    
    exclude_layers: Optional[List[int]] = field(
        default=None,
        metadata={"help": "List of layer indices to exclude from projection"},
    )
    
    # Projection settings
    projection_strength: float = field(
        default=1.0,
        metadata={"help": "Strength of projection (0.0 = no projection, 1.0 = full projection)"},
    )
    
    # Device settings
    device: str = field(
        default="cuda",
        metadata={"help": "Device for computation (cuda or cpu)"},
    )
    
    device_map: str = field(
        default="auto",
        metadata={"help": "Device map for model loading (auto, balanced, sequential)"},
    )
    
    # Logging
    verbose: bool = field(
        default=True,
        metadata={"help": "Whether to print detailed logs during projection"},
    )
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.base_model_path is None:
            raise ValueError("base_model_path cannot be None")
        if self.aligned_model_path is None:
            raise ValueError("aligned_model_path cannot be None")
        
        if self.load_in_4bit and self.load_in_8bit:
            raise ValueError("Cannot set both load_in_4bit and load_in_8bit to True")
        
        if self.select_layers_type == "threshold" and not (0.0 <= self.threshold <= 1.0):
            raise ValueError("threshold must be between 0.0 and 1.0")
        
        if self.projection_strength < 0.0 or self.projection_strength > 1.0:
            raise ValueError("projection_strength must be between 0.0 and 1.0")
    
    @classmethod
    def for_4bit_bnb(
        cls,
        base_model_path: str,
        aligned_model_path: str,
        **kwargs
    ) -> "SafeLoRAQuantizedConfig":
        """Factory method for 4-bit bitsandbytes configuration."""
        return cls(
            base_model_path=base_model_path,
            aligned_model_path=aligned_model_path,
            quantization_method="bitsandbytes_4bit",
            load_in_4bit=True,
            **kwargs
        )
    
    @classmethod
    def for_8bit_bnb(
        cls,
        base_model_path: str,
        aligned_model_path: str,
        **kwargs
    ) -> "SafeLoRAQuantizedConfig":
        """Factory method for 8-bit bitsandbytes configuration."""
        return cls(
            base_model_path=base_model_path,
            aligned_model_path=aligned_model_path,
            quantization_method="bitsandbytes_8bit",
            load_in_8bit=True,
            **kwargs
        )
    
    @classmethod  
    def for_gptq(
        cls,
        base_model_path: str,
        aligned_model_path: str,
        **kwargs
    ) -> "SafeLoRAQuantizedConfig":
        """Factory method for GPTQ quantization configuration."""
        return cls(
            base_model_path=base_model_path,
            aligned_model_path=aligned_model_path,
            quantization_method="gptq",
            **kwargs
        )
    
    @classmethod
    def for_awq(
        cls,
        base_model_path: str,
        aligned_model_path: str,
        **kwargs
    ) -> "SafeLoRAQuantizedConfig":
        """Factory method for AWQ quantization configuration."""
        return cls(
            base_model_path=base_model_path,
            aligned_model_path=aligned_model_path,
            quantization_method="awq",
            **kwargs
        )

